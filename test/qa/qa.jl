using SciMLTesting, ExponentialUtilities, JET, Test, LinearAlgebra
using AllocCheck: check_allocs

run_qa(
    ExponentialUtilities;
    ei_kwargs = (;
        # Non-public LinearAlgebra/BLAS/LAPACK names used by the non-allocating
        # LAPACK balancing wrapper `gebal_noalloc!` (which keeps the CPU matrix
        # exponential allocation-free), plus `Base.promote_op` used to infer the
        # exponential! workspace type at cache construction without allocating.
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :BlasInt, :chkfinite, :chklapackerror,
                :chkstride1, :libblastrampoline, :promote_op,
            ),
        ),
        # `chkstride1` (owned by LinearAlgebra) and `libblastrampoline` (owned by
        # libblastrampoline_jll) are reached through the `LinearAlgebra.BLAS`
        # re-exporter in the same balancing wrapper.
        all_qualified_accesses_via_owners = (;
            ignore = (:chkstride1, :libblastrampoline),
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
