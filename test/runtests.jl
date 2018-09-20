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
  # Krylov
  n = 20; m = 5; K = 4
  Random.seed!(0)
  A = randn(n, n)
  t = 1e-2
  b = randn(n)
  @test exp(t * A) * b ≈ expv(t, A, b; m=m)
  P = phi(t * A, K)
  W = fill(0., n, K+1)
  for i = 1:K+1
    W[:,i] = P[i] * b
  end
  Ks = arnoldi(A, b; m=m)
  W_approx = phiv(t, Ks, K)
  @test W ≈ W_approx

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
  @test w ≈ wperm
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
