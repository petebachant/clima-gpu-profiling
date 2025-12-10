# Figure out which project is currently activated
project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"

if contains(project_dir, "-mod")
    package_dir = "ClimaCore.jl-mod"
else
    package_dir = "ClimaCore.jl"
end

include(joinpath("..", package_dir, "test", "MatrixFields", "matrix_field_test_utils.jl"))

# Generate extruded finite difference spaces for testing. Include topography
# when possible.
function test_spaces(::Type{FT}) where {FT}
    velem = 63 # This should be big enough to test high-bandwidth matrices.
    helem = 16
    npoly = 3 # These should be small enough for the tests to be fast.

    comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
    hdomain = Domains.SphereDomain(FT(10))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(comms_ctx, hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_names=(:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems=velem)
    vtopology = Topologies.IntervalTopology(comms_ctx, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    sfc_coord = Fields.coordinate_field(hspace)
    hypsography =
        using_cuda ? Hypsography.Flat() :
        Hypsography.LinearAdaption(
            Geometry.ZPoint.(@. cosd(sfc_coord.lat) + cosd(sfc_coord.long) + 1),
        ) # TODO: FD operators don't currently work with hypsography on GPUs.
    center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

# Create a field matrix for a similar solve to ClimaAtmos's moist dycore + prognostic,
# EDMF + prognostic surface temperature with implicit acoustic waves and SGS fluxes
# also returns corresponding FieldVector
function dycore_prognostic_EDMF_FieldMatrix(
    ::Type{FT},
    center_space=nothing,
    face_space=nothing,
) where {FT}
    seed!(1) # For reproducibility with random fields
    if isnothing(center_space) || isnothing(face_space)
        center_space, face_space = test_spaces(FT)
    end
    surface_space = Spaces.level(face_space, half)
    surface_space = Spaces.level(face_space, half)
    sfc_vec = random_field(FT, surface_space)
    ᶜvec = random_field(FT, center_space)
    ᶠvec = random_field(FT, face_space)
    λ = 10
    ᶜᶜmat1 = random_field(DiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space) ./ λ
    ᶠᶜmat2 = random_field(BidiagonalMatrixRow{FT}, face_space) ./ λ
    ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
    ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)
    # Geometry.Covariant123Vector(1, 2, 3) * Geometry.Covariant12Vector(1, 2)'
    e¹² = Geometry.Covariant12Vector(1, 1)
    e₁₂ = Geometry.Contravariant12Vector(1, 1)
    e³ = Geometry.Covariant3Vector(1)
    e₃ = Geometry.Contravariant3Vector(1)

    ρχ_unit = (; ρq_tot=1, ρq_liq=1, ρq_ice=1, ρq_rai=1, ρq_sno=1)
    ρaχ_unit =
        (; ρaq_tot=1, ρaq_liq=1, ρaq_ice=1, ρaq_rai=1, ρaq_sno=1)

    ᶠᶜmat2_u₃_scalar = ᶠᶜmat2 .* (e³,)
    ᶜᶠmat2_scalar_u₃ = ᶜᶠmat2 .* (e₃',)
    ᶠᶠmat3_u₃_u₃ = ᶠᶠmat3 .* (e³ * e₃',)
    ᶜᶠmat2_ρχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit ⊠ e₃')), ᶜᶠmat2)
    ᶜᶜmat3_uₕ_scalar = ᶜᶜmat3 .* (e¹²,)
    ᶜᶜmat3_uₕ_uₕ =
        ᶜᶜmat3 .* (
            Geometry.Covariant12Vector(1, 0) *
            Geometry.Contravariant12Vector(1, 0)' +
            Geometry.Covariant12Vector(0, 1) *
            Geometry.Contravariant12Vector(0, 1)',
        )
    ᶜᶠmat2_uₕ_u₃ = ᶜᶠmat2 .* (e¹² * e₃',)
    ᶜᶜmat3_ρχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρχ_unit)), ᶜᶜmat3)
    ᶜᶜmat3_ρaχ_scalar = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit)), ᶜᶜmat3)
    ᶜᶠmat2_ρaχ_u₃ = map(Base.Fix1(map, Base.Fix2(⊠, ρaχ_unit ⊠ e₃')), ᶜᶠmat2)

    dry_center_gs_unit = (; ρ=1, ρe_tot=1, uₕ=e¹²)
    center_gs_unit = (; dry_center_gs_unit..., ρatke=1, ρχ=ρχ_unit)
    center_sgsʲ_unit = (; ρa=1, ρae_tot=1, ρaχ=ρaχ_unit)

    b = Fields.FieldVector(;
        sfc=sfc_vec .* ((; T=1),),
        c=ᶜvec .* ((; center_gs_unit..., sgsʲs=(center_sgsʲ_unit,)),),
        f=ᶠvec .* ((; u₃=e³, sgsʲs=((; u₃=e³),)),),
    )
    A = MatrixFields.FieldMatrix(
        # GS-GS blocks:
        (@name(c.ρe_tot), @name(c.ρe_tot)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.ρatke)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.ρχ)) => ᶜᶜmat3,
        (@name(c.uₕ), @name(c.uₕ)) => ᶜᶜmat3_uₕ_uₕ,
        (@name(f.u₃), @name(f.u₃)) => ᶠᶠmat3_u₃_u₃,
        # GS-SGS blocks:
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρae_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_tot), @name(c.sgsʲs.:(1).ρaχ.ρaq_tot)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_liq), @name(c.sgsʲs.:(1).ρaχ.ρaq_liq)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_ice), @name(c.sgsʲs.:(1).ρaχ.ρaq_ice)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_rai), @name(c.sgsʲs.:(1).ρaχ.ρaq_rai)) => ᶜᶜmat3,
        (@name(c.ρχ.ρq_sno), @name(c.sgsʲs.:(1).ρaχ.ρaq_sno)) => ᶜᶜmat3,
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρatke), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3,
        (@name(c.ρχ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_ρχ_scalar,
        (@name(c.uₕ), @name(c.sgsʲs.:(1).ρa)) => ᶜᶜmat3_uₕ_scalar,
        (@name(f.u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
        # SGS-SGS blocks:
        (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => ᶠᶠmat3_u₃_u₃,
    )
    return A, b
end


function test_field_matrix_solver(; test_name, alg, A, b, use_rel_error=false)
    # @testset "$test_name" begin
    x = similar(b)
    A′ = FieldMatrixWithSolver(A, b, alg)
    @test zero(A′) isa typeof(A′)
    solve_time =
        @benchmark ClimaComms.@cuda_sync comms_device ldiv!(x, A′, b)

    b_test = similar(b)
    # @test zero(b) isa typeof(b)
    mul_time =
        @benchmark ClimaComms.@cuda_sync comms_device mul!(b_test, A′, x)

    solve_time_rounded = round(solve_time; sigdigits=2)
    mul_time_rounded = round(mul_time; sigdigits=2)
    time_ratio = solve_time_rounded / mul_time_rounded
    time_ratio_rounded = round(time_ratio; sigdigits=2)

    error_vector = abs.(parent(b_test) .- parent(b))
    if use_rel_error
        rel_error = norm(error_vector) / norm(parent(b))
        rel_error_rounded = round(rel_error; sigdigits=2)
        error_string = "Relative Error = $rel_error_rounded"
    else
        max_error = maximum(error_vector)
        max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))
        error_string = "Maximum Error = $max_eps_error eps"
    end

    @info "$test_name:\n\tSolve Time = $solve_time_rounded s, \
           Multiplication Time = $mul_time_rounded s (Ratio = \
           $time_ratio_rounded)\n\t$error_string"

    if use_rel_error
        @test rel_error < 1e-5
    else
        @test max_eps_error <= 3
    end

    # TODO: fix broken test when Nv is added to the type space
    using_cuda || @test @allocated(ldiv!(x, A′, b)) ≤ 1536
    using_cuda || @test @allocated(mul!(b_test, A′, x)) == 0
    # end
end

# @testset "FieldMatrixSolver Unit Tests" begin
FT = Float64
center_space, face_space = test_spaces(FT)
surface_space = Spaces.level(face_space, half)

seed!(1) # ensures reproducibility

ᶜvec = random_field(FT, center_space)
ᶠvec = random_field(FT, face_space)
sfc_vec = random_field(FT, surface_space)

# Make each random square matrix diagonally dominant in order to avoid large
# large roundoff errors when computing its inverse. Scale the non-square
# matrices by the same amount as the square matrices.
λ = 10 # scale factor
ᶜᶜmat3 = random_field(TridiagonalMatrixRow{FT}, center_space) ./ λ .+ (I,)
ᶠᶠmat3 = random_field(TridiagonalMatrixRow{FT}, face_space) ./ λ .+ (I,)

for (vector, matrix, string1, string2) in (
    (ᶜvec, ᶜᶜmat3, "tri-diagonal matrix", "cell centers"),
    (ᶠvec, ᶠᶠmat3, "tri-diagonal matrix", "cell faces"),
)
    test_field_matrix_solver(;
        test_name="$string1 solve on $string2",
        alg=MatrixFields.BlockDiagonalSolve(),
        A=MatrixFields.FieldMatrix((@name(_), @name(_)) => matrix),
        b=Fields.FieldVector(; _=vector),
    )
end

# Test a more complex FieldMatrix similar to that used in ClimaAtmos's
# dycore + prognostic, EDMF + prognostic surface temperature solve.
A, b = dycore_prognostic_EDMF_FieldMatrix(FT, center_space, face_space)

keyname = keys(A).values[1]
keyname1 = @name(var1)

A1 = MatrixFields.FieldMatrix((keyname1, keyname1) => A[keyname])
b1_entry = MatrixFields.get_field(b, keyname[1])
b1 = Fields.FieldVector(; var1=b1_entry)

test_field_matrix_solver(;
    test_name="Dycore + prognostic, EDMF + prognostic surface temperature \
               solve",
    alg=MatrixFields.BlockDiagonalSolve(),
    A=A1,
    b=b1,
)
# end

# Test batched tri-diagonal solver
if package_dir == "ClimaCore.jl-mod"
    for (vector, matrix, string1, string2) in (
        (ᶜvec, ᶜᶜmat3, "tri-diagonal matrix", "cell centers"),
        (ᶠvec, ᶠᶠmat3, "tri-diagonal matrix", "cell faces"),
    )
        test_field_matrix_solver(;
            test_name="Batched $string1 solve on $string2",
            alg=MatrixFields.BatchedTridiagonalSolve(),
            A=MatrixFields.FieldMatrix((@name(_), @name(_)) => matrix),
            b=Fields.FieldVector(; _=vector),
        )
    end
end
