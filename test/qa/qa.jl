using SciMLTesting, ExponentialUtilities, JET, Test, LinearAlgebra
using AllocCheck: check_allocs
using ExplicitImports

# ExplicitImports only sees a package extension once its trigger weakdep is
# loaded (`Base.get_extension` returns `nothing` otherwise), so loading
# StaticArrays here is what puts ExponentialUtilitiesStaticArraysExt under QA.
using StaticArrays

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(
        ExponentialUtilities, :ExponentialUtilitiesStaticArraysExt
    ) !== nothing
end

run_qa(
    ExponentialUtilities;
    ei_kwargs = (;
        # `Base.promote_op` infers the exponential! workspace type at cache
        # construction without allocating one.
        # `arithmetic_closure` (owned by StaticArrays) is the only spelling of
        # "type you get from doing arithmetic on this eltype", which the
        # StaticArrays extension needs to pick the working eltype for `expv` on
        # an integer-valued `SMatrix`. It is documented -- its own docstring
        # demonstrates `import StaticArrays.arithmetic_closure` -- but StaticArrays
        # has not declared it `public`, and there is no public equivalent.
        # `_mul` is the package-internal product hook the StaticArrays extension
        # overrides.
        all_qualified_accesses_are_public = (;
            ignore = (:arithmetic_closure, :promote_op, :_mul),
        ),
        # ArrayInterface.parameterless_type is not declared public but is the
        # standard way to adapt a host array to the GPU array type of `w`.
        all_explicit_imports_are_public = (;
            ignore = (:diagview, :parameterless_type),
        ),
    ),
)
run_explicit_imports(
    Base.get_extension(ExponentialUtilities, :ExponentialUtilitiesStaticArraysExt), ExplicitImports;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (:arithmetic_closure, :_mul),
        ),
    ),
)

@testset "AllocCheck static analysis of the Krylov hot path" begin
    # AllocCheck's `check_allocs` runs a whole-method static allocation analysis.
    # It CANNOT report zero for these entry points, and that is expected: the
    # first call for a given subspace size lazily builds the `exponential!`
    # workspace via `LinearSolve.init` (hundreds of static allocation sites), and
    # the Padé denominator solve goes through `LinearSolve.solve!`; inlining
    # attributes those to the caller and they cannot be filtered from the truly
    # per-call code. Static analysis therefore cannot certify the reuse -- the
    # runtime size-independence testset in the Core group is the authoritative
    # allocation guard. This testset keeps AllocCheck wired in and documents the
    # situation; the `broken = true` markers record that a clean static zero is
    # not achievable while LinearSolve is on the path. It lives in the QA group
    # (Julia lts and 1 only) so AllocCheck is not exercised on the `pre` channel.
    m = 30
    n = 120
    A = collect(-2.0I(n) + 0.05 .* [1.0 / (1 + abs(i - j)) for i in 1:n, j in 1:n])
    b = [1.0 / i for i in 1:n]
    Ks = arnoldi(A, b; m = m)

    w = Matrix{Float64}(undef, n, 4)
    pcache = ExponentialUtilities.PhivCache(b, m, 4)
    phiv!(w, 0.1, Ks, 3; cache = pcache)  # warm the workspace
    phiv_warm(w, Ks, c) = phiv!(w, 0.1, Ks, 3; cache = c)
    phiv_allocs = check_allocs(
        phiv_warm, (typeof(w), typeof(Ks), typeof(pcache)); ignore_throw = true
    )
    @test isempty(phiv_allocs) broken = true

    wv = zeros(n)
    ecache = ExponentialUtilities.ExpvCache{Float64}(m)
    expv!(wv, 0.1, Ks; cache = ecache)  # warm
    expv_warm(wv, Ks, c) = expv!(wv, 0.1, Ks; cache = c)
    expv_allocs = check_allocs(
        expv_warm, (typeof(wv), typeof(Ks), typeof(ecache)); ignore_throw = true
    )
    @test isempty(expv_allocs) broken = true
end
