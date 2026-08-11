"""
    AMIPWarmup

Precompilation workload that caches ClimaCore's native code across jobs.

## Why this package exists

An AMIP profiling job spends ~88% of its wall time in host-side Julia JIT
(measured: 1715 s of 1942 s, to produce 0.6 s of steady-state stepping). A
`--trace-compile` of that job shows where it goes: **89.4% of the methods
compiled at run time involve ClimaCore** -- spaces, fields, broadcasting,
operators. ClimaAtmos accounts for 0.8% and CloudMicrophysics for 0.4%.

Julia caches native code into a *package's* precompilation image, and the AMIP
experiment is a script, not a package, so none of that work is ever kept. This
package gives it somewhere to live.

## Why the dependencies are what they are

See the policy comment in `Project.toml`. In short: ClimaAtmos and
CloudMicrophysics are under active development here, and a package is
invalidated whenever a dependency changes, so depending on either would throw
this cache away on every edit. ClimaCore is stable and accounts for the
overwhelming majority of the compilation, which is what makes this worthwhile.

## Matching types is the whole game

Cached specializations are keyed on *types*, so the workload must construct the
same ones AMIP does. Element counts do not appear in any type, so `h_elem` is
free and `z_elem` is bounded only by the stretching rule being solvable. These
DO appear in types, and were read out of a real AMIP trace rather than guessed:

  * `Float32`                       (320k mentions in the trace, vs 5.8k Float64)
  * `Quadratures.GLL{4}`            (i.e. nh_poly = 3)
  * `Hypsography.SLEVEAdaption`     (4517 mentions; the config sets
                                     `topography: "Earth"`, and a flat space is
                                     a *different type* that would cache the
                                     wrong specializations)
  * `GeneralizedExponentialStretching` -- `IntervalMesh{S, ...}` carries the
                                     stretching rule as `S`, so a Uniform
                                     vertical mesh is likewise a different type

The elevation field is synthetic. Only its type matters for caching, so this
avoids reading the Earth topography artifact during precompilation.
"""
module AMIPWarmup

import ClimaComms
# Required: ClimaComms refuses to hand out a CUDADevice unless the backend
# package is loaded, and CLIMACOMMS_DEVICE=CUDA is set for every job here.
# Without this, precompilation dies with "Loading CUDA.jl is required".
ClimaComms.@import_required_backends
import ClimaCore
using PrecompileTools: @setup_workload, @compile_workload

const CC = ClimaCore

"""
    amip_like_grid(FT; h_elem, z_elem, context)

Build a grid whose *type* matches the AMIP configuration: cubed sphere,
`GLL{4}` quadrature, stretched vertical mesh, SLEVE topography adaption.
Deliberately tiny -- resolution is not part of the type.
"""
function amip_like_grid(::Type{FT}; h_elem, z_elem, context) where {FT}
    return CC.CommonGrids.ExtrudedCubedSphereGrid(
        FT;
        z_elem,
        z_min = FT(0),
        z_max = FT(60000),
        radius = FT(6.371e6),
        h_elem,
        n_quad_points = 4,          # GLL{4}
        # The stretching rule IS type-bearing: `IntervalMesh{S, ...}` carries it
        # as `S`, so a Uniform mesh would produce a different space type and
        # cache the wrong specializations. Its *values* are not type-bearing, so
        # they need not match the config exactly -- but the rule must be
        # solvable for the chosen `z_elem`, which is why `z_elem` cannot be made
        # arbitrarily small here (4 levels over 60 km fails outright).
        stretch = CC.Meshes.GeneralizedExponentialStretching(FT(30), FT(8000)),
        # SLEVEAdaption over a synthetic surface: same type as the real run,
        # without touching the topography artifact.
        hypsography_fun = (h_grid, z_grid) -> begin
            h_space = CC.Spaces.SpectralElementSpace2D(h_grid)
            # NB: deliberately no `FT` inside this innermost closure. `FT` would
            # be captured through two closure levels, at which point inference
            # sees a `DataType` variable rather than a compile-time constant and
            # `FT(500)` becomes non-inferrable -- ClimaCore then rejects the
            # elevation function outright ("Concrete type of result could not be
            # inferred"). Integer literals promote against the coordinate's own
            # float type, so the result is Float32 with nothing captured.
            # A Field of ZPoint, not of bare floats: `ref_z_to_physical_z`
            # reads `adaption.surface.z`, so raw Float32 elevations fail with
            # "type Float32 has no field z".
            z_surface = map(CC.Fields.coordinate_field(h_space)) do coord
                CC.Geometry.ZPoint(500 * (1 + sind(coord.lat) * cosd(coord.long)))
            end
            CC.Hypsography.SLEVEAdaption(z_surface, FT(0.5), FT(10000))
        end,
        context,
    )
end

"""
    exercise!(grid)

Touch the ClimaCore paths an AMIP timestep leans on: field construction,
fused broadcasting, vertical and spectral operators, and column reductions.
"""
function exercise!(grid)
    cspace = CC.Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    # Dispatch on the float type so `FT` is a compile-time constant in the body.
    # Binding it as a local (`FT = undertype(space)`) makes it a `DataType`
    # *value*, and every `FT(0)` inside a closure or broadcast then infers as
    # `Any` -- ClimaCore rejects that outright ("Concrete type of result could
    # not be inferred").
    return _exercise!(grid, cspace, CC.Spaces.undertype(cspace))
end

function _exercise!(grid, cspace, ::Type{FT}) where {FT}
    fspace = CC.Spaces.FaceExtrudedFiniteDifferenceSpace(grid)

    # Plain constructors rather than `map` closures: fewer inference hazards,
    # and the field types are what matter for caching, not the values.
    ρ = CC.Fields.ones(cspace)
    e = CC.Fields.ones(cspace)
    q = CC.Fields.ones(cspace)
    @. e = FT(300)
    @. q = FT(1e-3)
    w = CC.Fields.zeros(fspace)

    # Fused pointwise broadcast over multiple fields -- the dominant kernel shape.
    @. ρ = ρ + FT(0) * e * q

    # Vertical interpolation both ways, plus a face-valued gradient: the
    # operator specializations a tendency evaluation instantiates.
    If2c = CC.Operators.InterpolateF2C()
    Ic2f = CC.Operators.InterpolateC2F(
        bottom = CC.Operators.Extrapolate(),
        top = CC.Operators.Extrapolate(),
    )
    ∂f = CC.Operators.GradientC2F(
        bottom = CC.Operators.SetGradient(CC.Geometry.WVector(FT(0))),
        top = CC.Operators.SetGradient(CC.Geometry.WVector(FT(0))),
    )
    divc = CC.Operators.DivergenceF2C()

    @. w = Ic2f(ρ) * FT(0)
    @. ρ += FT(0) * If2c(w)
    @. e += FT(0) * divc(∂f(e))

    # Spectral operators and DSS, used every stage on the horizontal.
    grad = CC.Operators.Gradient()
    wdiv = CC.Operators.WeakDivergence()
    @. e += FT(0) * wdiv(grad(e))
    CC.Spaces.weighted_dss!(e)

    # Column reductions feed diagnostics and conservation checks.
    colint = similar(CC.Fields.level(ρ, 1))
    CC.Operators.column_integral_definite!(colint, ρ)

    # FieldVector arithmetic: the state representation the timestepper mutates.
    Y = CC.Fields.FieldVector(; c = (; ρ, e, q), f = (; w))
    dY = similar(Y)
    fill!(parent(dY.c.ρ), 0)
    @. dY = FT(0) * Y
    Y .+= FT(0) .* dY

    return nothing
end

# Off-switch: set AMIPWARMUP_SKIP=1 to build the package without running the
# workload. Useful as an A/B control (does the workload actually cost anything at
# precompile time?) and to disable warming without editing code.
const SKIP_WORKLOAD = get(ENV, "AMIPWARMUP_SKIP", "false") in ("1", "true", "yes")

@setup_workload begin
    # Everything, including acquiring the device, sits inside the guard.
    # Precompilation must never fail the package: a workload that stops
    # building should degrade to "no speedup", not "cannot load". An earlier
    # version guarded only the workload body, so a failure in setup still
    # broke the build.
    try
        SKIP_WORKLOAD && error("AMIPWARMUP_SKIP set; skipping workload")
        device = ClimaComms.device()
        context = ClimaComms.context(device)
        # z_elem matches the real config (63): the stretching rule must be
        # solvable, and 63 levels on a 2-element sphere is still tiny.
        grid = amip_like_grid(Float32; h_elem = 2, z_elem = 63, context)
        @compile_workload begin
            exercise!(grid)
        end
    catch err
        if SKIP_WORKLOAD
            @info "AMIPWarmup workload skipped (AMIPWARMUP_SKIP set)"
        else
            @warn "AMIPWarmup workload failed; continuing without it" err
        end
    end
end

end # module
