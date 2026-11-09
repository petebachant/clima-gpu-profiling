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
        error("$(length(all_problems)) macro reference(s) name a macro their module does not define")
    end
    println("\nno qualified macro reference names a macro its module lacks")
    return nothing
end

main(ARGS)
