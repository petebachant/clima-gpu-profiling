#!/usr/bin/env python3
"""Parse Nsight Systems .nsys-rep file and generate interactive timeline with
Plotly.

This was generated with Claude and hasn't been tested/fixed yet.
"""

import argparse
import sqlite3
from pathlib import Path

import plotly.graph_objects as go


def get_string_from_id(cursor, string_id):
    """Resolve string ID to actual string value"""
    if string_id is None:
        return "Unknown"
    result = cursor.execute(
        "SELECT value FROM StringIds WHERE id = ?", (string_id,)
    ).fetchone()
    return result[0] if result else f"ID_{string_id}"


def parse_kernels(cursor):
    """Extract GPU kernel events"""
    events = []
    try:
        kernels = cursor.execute("""
            SELECT start, end, shortName, demangledName, gridX, gridY, gridZ,
                   blockX, blockY, blockZ, streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY start
        """).fetchall()

        for k in kernels:
            (
                start,
                end,
                short_name,
                demangled_name,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                stream,
            ) = k
            name = (
                get_string_from_id(cursor, demangled_name)
                or get_string_from_id(cursor, short_name)
                or "Kernel"
            )

            # Truncate long names
            if len(name) > 50:
                name = name[:47] + "..."

            events.append(
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "type": "CUDA Kernel",
                    "category": "GPU",
                    "details": f"Grid: ({gx},{gy},{gz}) Block: ({bx},{by},{bz}) Stream: {stream}",
                }
            )
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not parse kernels: {e}")

    return events


def parse_memcpy(cursor):
    """Extract memory copy events"""
    events = []
    try:
        memcpys = cursor.execute("""
            SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            ORDER BY start
        """).fetchall()

        copy_kinds = {
            1: "HtoD",
            2: "DtoH",
            3: "HtoA",
            4: "AtoH",
            5: "AtoA",
            6: "AtoD",
            7: "DtoA",
            8: "DtoD",
            10: "PtoP",
        }

        for m in memcpys:
            start, end, bytes_copied, copy_kind, src_kind, dst_kind, stream = m
            kind_str = copy_kinds.get(copy_kind, f"Type{copy_kind}")

            events.append(
                {
                    "name": f"cudaMemcpy {kind_str}",
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "type": "Memory Copy",
                    "category": "GPU",
                    "details": f"Size: {bytes_copied:,} bytes, Stream: {stream}",
                }
            )
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not parse memcpy: {e}")

    return events


def parse_cuda_runtime(cursor):
    """Extract CUDA runtime API calls"""
    events = []
    try:
        runtime_calls = cursor.execute("""
            SELECT start, end, nameId, globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            ORDER BY start
            LIMIT 10000
        """).fetchall()

        for call in runtime_calls:
            start, end, name_id, thread_id = call
            name = get_string_from_id(cursor, name_id)

            events.append(
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "type": "CUDA Runtime",
                    "category": f"CPU Thread {thread_id}",
                    "details": f"Thread: {thread_id}",
                }
            )
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not parse runtime calls: {e}")

    return events


def parse_nvtx(cursor):
    """Extract NVTX markers and ranges"""
    events = []
    try:
        nvtx_events = cursor.execute("""
            SELECT start, end, textId, globalTid
            FROM NVTX_EVENTS
            WHERE end IS NOT NULL
            ORDER BY start
        """).fetchall()

        for nvtx in nvtx_events:
            start, end, text_id, thread_id = nvtx
            name = get_string_from_id(cursor, text_id)

            events.append(
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "type": "NVTX Range",
                    "category": f"CPU Thread {thread_id}",
                    "details": f"Thread: {thread_id}",
                }
            )
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not parse NVTX events: {e}")

    return events


def create_timeline_plot(events, output_file="timeline.html"):
    """Create interactive Plotly timeline"""
    if not events:
        print("No events found to plot!")
        return

    # Convert timestamps to milliseconds for readability
    min_time = min(e["start"] for e in events)
    for event in events:
        event["start_ms"] = (event["start"] - min_time) / 1e6
        event["end_ms"] = (event["end"] - min_time) / 1e6
        event["duration_ms"] = event["duration"] / 1e6

    # Group events by category for swim lanes
    categories = sorted(set(e["category"] for e in events))

    # Assign colors by type
    type_colors = {
        "CUDA Kernel": "#1f77b4",
        "Memory Copy": "#ff7f0e",
        "CUDA Runtime": "#2ca02c",
        "NVTX Range": "#d62728",
    }

    fig = go.Figure()

    # Add events as rectangles
    for event in events:
        color = type_colors.get(event["type"], "#7f7f7f")

        fig.add_trace(
            go.Bar(
                x=[event["duration_ms"]],
                y=[event["category"]],
                base=[event["start_ms"]],
                orientation="h",
                name=event["type"],
                marker=dict(color=color),
                hovertemplate=(
                    f"<b>{event['name']}</b><br>"
                    f"Type: {event['type']}<br>"
                    f"Start: {event['start_ms']:.3f} ms<br>"
                    f"Duration: {event['duration_ms']:.3f} ms<br>"
                    f"{event['details']}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
                width=0.8,
            )
        )

    # Update layout
    fig.update_layout(
        title="Nsight Systems Timeline",
        xaxis_title="Time (ms)",
        yaxis_title="Category",
        height=max(400, len(categories) * 40),
        hovermode="closest",
        bargap=0.1,
        barmode="overlay",
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"Timeline saved to {output_file}")
    print(f"Total events plotted: {len(events)}")
    print(f"Categories: {len(categories)}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Nsight Systems report and create timeline"
    )
    parser.add_argument("nsys_file", type=str, help="Path to .nsys-rep file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="timeline.html",
        help="Output HTML file (default: timeline.html)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit total events to parse (default: 10000)",
    )

    args = parser.parse_args()

    if not Path(args.nsys_file).exists():
        print(f"Error: File not found: {args.nsys_file}")
        return

    print(f"Parsing {args.nsys_file}...")

    # Connect to SQLite database
    conn = sqlite3.connect(args.nsys_file)
    cursor = conn.cursor()

    # List available tables
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    print(f"Available tables: {[t[0] for t in tables]}")

    # Parse different event types
    all_events = []

    print("Parsing GPU kernels...")
    all_events.extend(parse_kernels(cursor))

    print("Parsing memory copies...")
    all_events.extend(parse_memcpy(cursor))

    print("Parsing CUDA runtime calls...")
    all_events.extend(parse_cuda_runtime(cursor))

    print("Parsing NVTX events...")
    all_events.extend(parse_nvtx(cursor))

    # Limit events if needed
    if len(all_events) > args.limit:
        print(f"Limiting to {args.limit} events (found {len(all_events)})")
        all_events = sorted(all_events, key=lambda x: x["start"])[: args.limit]

    # Create timeline
    create_timeline_plot(all_events, args.output)

    conn.close()


if __name__ == "__main__":
    main()
