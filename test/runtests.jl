using Test, LinearAlgebra, Random, SparseArrays, ExponentialUtilities
using ExponentialUtilities: getH, getV

@testset "Phi" begin
    # Scalar phi
    K = 4
    z = 0.1
    P = fill(0., K+1); P[1] = exp(z)
    for i = 1:K
        P[i+1] = (P[i] - 1/factorial(i-1))/z
    end
    @test phi(z, K) ≈ P

    # Matrix phi (dense)
    A = [0.1 0.2; 0.3 0.4]
    P = Vector{Matrix{Float64}}(undef, K+1); P[1] = exp(A)
    for i = 1:K
        P[i+1] = (P[i] - 1/factorial(i-1)*I) / A
    end
    @test phi(A, K) ≈ P

    # Matrix phi (Diagonal)
    A = Diagonal([0.1, 0.2, 0.3, 0.4])
    Afull = Matrix(A)
    P = phi(A, K)
    Pfull = phi(Afull, K)
    for i = 1:K+1
        @test Matrix(P[i]) ≈ Pfull[i]
    end
end

@testset "Arnoldi & Krylov" begin
    Random.seed!(0)
    n = 20; m = 5; K = 4
    A = randn(n, n)
    t = 1e-2
    b = randn(n)
    direct = exp(t * A) * b
    @test direct ≈ expv(t, A, b; m=m)
    @test direct ≈ kiops(t, A, b)[1]
    P = phi(t * A, K)
    W = fill(0., n, K+1)
    for i = 1:K+1
        W[:,i] = P[i] * b
    end
    Ks = arnoldi(A, b; m=m)
    W_approx = phiv(t, Ks, K)
    @test W ≈ W_approx
    W_approx_kiops3 = kiops(t, A, hcat([b*inv(t)^i for i in 0:K-1]...))
    @test sum(W[:, 1:K], dims=2) ≈ W_approx_kiops3[1]
    @test_skip begin
        W_approx_kiops4 = kiops(t, A, hcat([b*inv(t)^i for i in 0:K]...))
        @test_broken sum(W[:, 1:K+1], dims=2) ≈ W_approx_kiops4[1]
        @test sum(W[:, 1:K+1], dims=2) ≈ W_approx_kiops4[1] atol=1e-2
    end

    # Happy-breakdown in Krylov
    v = normalize(randn(n))
    A = v * v' # A is Idempotent
    Ks = arnoldi(A, b)
    @test Ks.m == 2

    # Arnoldi vs Lanczos
    A = Hermitian(randn(n, n))
    Aperm = A + 1e-10 * randn(n, n) # no longer Hermitian
    w = expv(t, A, b; m=m)
    wperm = expv(t, Aperm, b; m=m)
    wkiops = kiops(t, A, b; m=m)[1]
    @test w ≈ wperm
    @test w ≈ wkiops
end

@testset "Complex Value" begin
    n = 20; m = 10;
    for A in [Hermitian(rand(ComplexF64, n, n)), Hermitian(rand(n, n)), rand(ComplexF64, n, n), rand(n, n)]
        for b in [rand(ComplexF64, n), rand(n)], t in [1e-2, 1e-2im, 1e-2 + 1e-2im]
            @test exp(t * A) * b ≈ expv(t, A, b; m=m)
        end
    end
end

@testset "Adaptive Krylov" begin
    # Internal time-stepping for Krylov (with adaptation)
    n = 100
    K = 4
    t = 5.0
    tol = 1e-7
    A = spdiagm(-1=>ones(n-1), 0=>-2*ones(n), 1=>ones(n-1))
    B = randn(n, K+1)
    Phi_half = phi(t/2 * A, K)
    Phi = phi(t * A, K)
    uhalf_exact = sum((t/2)^i * Phi_half[i+1] * B[:,i+1] for i = 0:K)
    u_exact = sum(t^i * Phi[i+1] * B[:,i+1] for i = 0:K)
    U = phiv_timestep([t/2, t], A, B; adaptive=true, tol=tol)
    @test norm(U[:,1] - uhalf_exact) / norm(uhalf_exact) < tol
    @test norm(U[:,2] - u_exact) / norm(u_exact) < tol
    # p = 0 special case (expv_timestep)
    u_exact = Phi[1] * B[:, 1]
    u = expv_timestep(t, A, B[:, 1]; adaptive=true, tol=tol)
    @test norm(u - u_exact) / norm(u_exact) < tol
end

@testset "Krylov for Hermitian matrices" begin
    # Hermitian matrices have real spectra. Ensure that the subspace
    # matrix is representable as a real matrix.

    n = 100
    m = 15
    tol = 1e-14

    e = ones(n)
    p = -im*Tridiagonal(-e[2:end], 0e, e[2:end])

    KsA = KrylovSubspace{ComplexF64}(n, m)
    KsL = KrylovSubspace{ComplexF64, Float64}(n, m)

    v = rand(ComplexF64, n)

    arnoldi!(KsA, p, v)
    lanczos!(KsL, p, v)

    AH = view(KsA.H,1:KsA.m,1:KsA.m)
    LH = view(KsL.H,1:KsL.m,1:KsL.m)

    @test norm(AH-LH)/norm(AH) < tol
end

@testset "Alternative Lanczos expv Interface" begin
    n = 300
    m = 30

    A = Hermitian(rand(n,n))
    b = rand(ComplexF64, n)
    dt = 0.1

    atol=1e-10
    rtol=1e-10
    w = expv(-im, dt*A, b, m=m, tol=atol, rtol=rtol, mode=:error_estimate)

    function fullexp(A, v)
        w = similar(v)
        eA = exp(A)
        mul!(w, eA, v)
        w
    end

    w′ = fullexp(-im*dt*A, b)

    δw = norm(w-w′)
    @test δw < atol
    @test δw/abs(1e-16+norm(w)) < rtol
end

struct MatrixFreeOperator{T} <: AbstractMatrix{T}
    A::Matrix{T}
end
Base.eltype(A::MatrixFreeOperator{T}) where T = T
LinearAlgebra.mul!(y::AbstractVector, A::MatrixFreeOperator, x::AbstractVector) = mul!(y, A.A, x)
Base.size(A::MatrixFreeOperator, dim) = size(A.A, dim)
struct OpnormFunctor end
(::OpnormFunctor)(A::MatrixFreeOperator, p::Real) = opnorm(A.A, p)
@testset "Matrix-free Operator" begin
    Random.seed!(123)
    n = 20
    for ishermitian in (false, true)
        A = rand(ComplexF64, n, n)
        M = ishermitian ? A'A : A
        Op = MatrixFreeOperator(M)
        b = rand(ComplexF64, n)
        Ks = arnoldi(Op, b; ishermitian=ishermitian, opnorm=OpnormFunctor(), tol=1e-12)
        pv = phiv(0.01, Ks, 2)
        pv′ = hcat(map(A->A*b, phi(0.01Op.A, 2))...)

        @test pv ≈ pv′ atol=1e-12
    end
end
