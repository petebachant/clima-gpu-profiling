# Will `julia script.jl` get as far as running this script?
#
# Two checks, each added after it cost a GPU reservation on 2026-11-09:
#
#  1. INCREMENTAL PARSE, which is what Julia actually does -- it parses and
#     evaluates one top-level expression at a time, so a syntax error late in
#     the file surfaces only after everything before it has already run.
#     `Meta.parseall` parses the whole file as one block and ACCEPTS input that
#     the incremental parse rejects: `() -> CUDA.@cuda launch = false f(x)` as
#     a call argument, where `launch = false` reads as a keyword of the
#     enclosing call. parseall said OK; the job died 38 minutes in.
#
#  2. QUALIFIED MACRO REFERENCES: does every `Module.@macro` exist in that
#     module? Parsing never resolves macro names, and expansion happens when
#     the enclosing definition is evaluated -- after the model is built. That
#     is how `Base.@printf`, which has never existed, cost 42 minutes.
#
#  3. UNQUALIFIED CALLS TO AN IMPORTED MODULE'S EXPORTS, added 2026-11-15. A
#     name a module exports is reachable unqualified only after `using`; under
#     `import` it is an undefined name. `measure-radiation-phase-registers.jl`
#     imports CUDA and wrote `threadIdx()` in seven device stubs, so all seven
#     failed after the warmup step had already built the model -- and the error
#     reads from the wrong end, leading with `gpu_gc_pool_alloc` and a dynamic
#     call, which are consequences, and naming the undefined name several
#     lines down.
#
# `Meta.parseall` does not expand macros, so a parse check accepts
# `Base.@printf` -- which does not exist, since `@printf` lives in `Printf`.
# That cost a 42-minute GPU reservation on 2026-11-09: the script built a
# CoupledSimulation, ran its warmup step, and only then failed expanding a
# macro name that was wrong before the job was ever submitted.
#
# This resolves each qualified macro reference against the loaded module, so
# it catches the whole class in a second and without a GPU. Run it on a
# measurement script before submitting:
#
#   julia --project=<env> scripts/check-macro-references.jl <script.jl>...
#
# Modules the script imports must be loadable here, which for the measurement
# scripts means running it under the same project. A module that cannot be
# resolved is reported rather than skipped silently.

# Drop comments, so a note *about* a bad macro name is not reported as one.
# This is a line-level approximation: a `#` inside a string literal would end
# the line early, which can only hide a reference, never invent one.
function strip_comment(line)
    i = findfirst('#', line)
    return isnothing(i) ? line : line[1:prevind(line, i)]
end

function qualified_macro_references(src)
    refs = Tuple{String, String, Int}[]
    for (i, line) in enumerate(split(src, '\n'))
        for m in eachmatch(r"\b([A-Z][A-Za-z0-9_]*)\.@([A-Za-z0-9_!]+)", strip_comment(line))
            push!(refs, (m.captures[1], m.captures[2], i))
        end
    end
    return refs
end

# Load the modules the script itself imports, so the check is self-sufficient
# rather than depending on what the caller happened to import. Only plain
# `import X` / `using X` lines are honored; a module that fails to load is
# reported by the reference check below rather than aborting here.
function import_script_modules(src)
    for raw in split(src, '\n')
        line = strip_comment(raw)
        # `import A`, `using A`, and the aliased form `import A.B as C`, which
        # is how these scripts reach submodules (`import ClimaCore.MatrixFields
        # as MF`). The alias is the name a macro reference will be written with.
        m = match(r"^\s*(?:import|using)\s+([A-Za-z][A-Za-z0-9_.]*)(?:\s+as\s+([A-Za-z][A-Za-z0-9_]*))?\s*$", line)
        isnothing(m) && continue
        path, alias = m.captures[1], m.captures[2]
        name = isnothing(alias) ? first(split(path, '.')) : alias
        isdefined(Main, Symbol(name)) && continue
        try
            Core.eval(Main, Meta.parse(isnothing(alias) ? "import $path" : "import $path as $alias"))
        catch err
            println("  note: could not import $path ($(typeof(err)))")
        end
    end
    return nothing
end

# Resolve a dotted module path through Main, for the export checks below.
function resolve_module(name)
    mod = Main
    for part in split(name, '.')
        sym = Symbol(part)
        isdefined(mod, sym) || return nothing
        mod = getfield(mod, sym)
        mod isa Module || return nothing
    end
    return mod
end

# Which modules arrived by `using` (exports reachable unqualified) and which by
# `import` (exports reachable only through the module).
function module_import_forms(src)
    usings, imports = String[], String[]
    for raw in split(src, '\n')
        m = match(
            r"^\s*(import|using)\s+([A-Za-z][A-Za-z0-9_.]*)(?:\s+as\s+([A-Za-z][A-Za-z0-9_]*))?\s*$",
            strip_comment(raw),
        )
        isnothing(m) && continue
        kind, path, alias = m.captures[1], m.captures[2], m.captures[3]
        push!(kind == "using" ? usings : imports, isnothing(alias) ? path : alias)
    end
    return usings, imports
end

# Names the script binds itself, which shadow an import and are not the bug.
# Over-collecting here only costs a missed report, never a false one.
function script_defined_names(src)
    defs = Set{Symbol}()
    for raw in split(src, '\n')
        line = strip_comment(raw)
        for re in (
            r"^\s*function\s+([A-Za-z_][A-Za-z0-9_!]*)",
            r"^\s*(?:const\s+|local\s+|global\s+)?([A-Za-z_][A-Za-z0-9_!]*)\s*(?:=|\()",
        )
            m = match(re, line)
            isnothing(m) || push!(defs, Symbol(m.captures[1]))
        end
    end
    return defs
end

# Device intrinsics that are never valid unqualified under `import CUDA`, kept
# here so the check still fires when CUDA itself cannot be loaded in the
# checker's project -- which is the common case, and is what made the first
# version of this check silently inert against its own regression case.
const GATED_FALLBACK = Dict(
    "CUDA" => Symbol.([
        "threadIdx", "blockIdx", "blockDim", "gridDim", "warpsize", "laneid",
        "sync_threads", "sync_warp", "threadfence", "threadfence_block",
        "shfl_sync", "shfl_up_sync", "shfl_down_sync", "shfl_xor_sync",
        "vote_all_sync", "vote_any_sync", "vote_ballot_sync", "active_mask",
    ]),
)

# A bare `name(` where `name` is exported by an import-only module, is not
# bound in this file, and does not resolve in Main -- so it cannot resolve at
# run time either. The `(?<![.\w])` is what keeps a qualified `CUDA.name(`
# out of the match. Returns the hits and the import-only modules it could not
# resolve, because a check that cannot see a module is UNCHECKED, not clean.
function unqualified_import_calls(src)
    usings, imports = module_import_forms(src)
    reachable = Set{Symbol}()
    for u in usings
        mod = resolve_module(u)
        isnothing(mod) || union!(reachable, names(mod))
    end
    here = script_defined_names(src)
    gated = Dict{Symbol, String}()
    unresolved = String[]
    for i in imports
        mod = resolve_module(i)
        candidates = if isnothing(mod)
            haskey(GATED_FALLBACK, i) || push!(unresolved, i)
            get(GATED_FALLBACK, i, Symbol[])
        else
            names(mod)
        end
        for n in candidates
            (n in reachable || n in here || isdefined(Main, n)) && continue
            get!(gated, n, i)
        end
    end
    hits = Tuple{Symbol, String, Int}[]
    isempty(gated) && return hits, unresolved
    for (ln, raw) in enumerate(split(src, '\n'))
        for m in eachmatch(r"(?<![.\w])([A-Za-z_][A-Za-z0-9_!]*)\s*\(", strip_comment(raw))
            sym = Symbol(m.captures[1])
            haskey(gated, sym) && push!(hits, (sym, gated[sym], ln))
        end
    end
    return hits, unresolved
end

# A reference is only a failure when the module resolves and the macro is
# absent -- that is the `Base.@printf` case. A module this session cannot see
# (one bound in an included file, or aliased inside a function) is UNCHECKED,
# not wrong, and conflating the two would make the gate unusable.
# Parse the file the way `julia file.jl` parses it: one top-level expression at
# a time, each one evaluated before the next is read.
function incremental_parse(src)
    pos, n = 1, 0
    while pos <= lastindex(src)
        local ex
        try
            (ex, pos) = Meta.parse(src, pos; greedy = true, raise = true)
        catch err
            return (n, first(sprint(showerror, err), 400))
        end
        ex === nothing && break
        n += 1
    end
    return (n, nothing)
end

function check(path)
    src = read(path, String)
    (nexprs, parse_err) = incremental_parse(src)
    if !isnothing(parse_err)
        println("$path: INCREMENTAL PARSE FAILED after $nexprs expressions")
        return (["$path  incremental parse failed after $nexprs expressions:\n$parse_err"], String[])
    end
    println("$path: $nexprs top-level expressions parse incrementally")
    import_script_modules(src)
    refs = qualified_macro_references(src)
    problems, unchecked = String[], String[]
    for (modname, macroname, line) in refs
        mod = try
            getfield(Main, Symbol(modname))
        catch
            push!(unchecked, "$path:$line  $modname.@$macroname (module not visible here)")
            continue
        end
        mod isa Module || continue
        isdefined(mod, Symbol("@", macroname)) ||
            push!(problems, "$path:$line  $modname.@$macroname is NOT defined in $modname")
    end
    (bare, unresolved) = unqualified_import_calls(src)
    for (sym, modname, line) in bare
        push!(problems,
            "$path:$line  `$sym(` is an undefined name: $modname is imported, \
             not used, so write $modname.$sym")
    end
    for modname in unique(unresolved)
        push!(unchecked,
            "$path  bare calls to $modname's exports (module not visible here)")
    end
    checked = length(refs) - length(unchecked)
    println("$path: $checked/$(length(refs)) references checked, \
             $(length(problems)) bad, $(length(unchecked)) unchecked")
    return problems, unchecked
end

function main(paths)
    all_problems, all_unchecked = String[], String[]
    for p in paths
        problems, unchecked = check(p)
        append!(all_problems, problems)
        append!(all_unchecked, unchecked)
    end
    if !isempty(all_unchecked)
        println("\nunchecked (module not visible from this project):")
        foreach(u -> println("  ", u), all_unchecked)
    end
    if !isempty(all_problems)
        println("\nBAD:")
        foreach(p -> println("  ", p), all_problems)
        error("$(length(all_problems)) reference(s) will not resolve at run time")
    end
    println("\nevery qualified macro exists and no bare call needs a module prefix")
    return nothing
end

main(ARGS)
