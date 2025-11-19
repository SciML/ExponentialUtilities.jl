using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using ExponentialUtilities: inplace_add!,
                            exponential!, ExpMethodHigham2005, expv,
                            expv_timestep

using Random: Xoshiro

@testset "GPU-safe inplace_add!" begin
    A_d = cu(zeros(Float32, 8, 8))

    λ = 2.0f0 * I

    # Make sure scalar indexing isn't happening with the GPU
    @test_nowarn inplace_add!(A_d, λ)
    @test collect(A_d) == λ
end

@testset "GPU Exponential" begin
    n = 256
    rng = Xoshiro(0x0451)
    A = randn(rng, Float32, (n, n))
    eA = exp(A)
    A_d = cu(A)

    exponential!(copy(A_d)) # Make sure simple command works

    # Iterate over GPU-compatible methods
    for m in (ExpMethodHigham2005(false),)
        @testset "GPU Exponential, $(string(m))" begin
            eA_d = exponential!(copy(A_d), m)

            @test collect(eA_d) ≈ eA
        end
    end
end

for t in [0.1, 0.1im]
    n = 1000
    A = sprand(ComplexF64, n, n, 10 / n)
    A = triu(A, 1) + sprand(ComplexF64, n, n, 1 / n)
    b = rand(ComplexF64, n)

    A_gpu = CuSparseMatrixCSR(A)
    b_gpu = CuArray(b)

    ts = Array(LinRange(0, 1, 300))

    E1 = expv(t, A, b)
    E2 = Array(expv(t, A_gpu, b_gpu))
    @show size(E1), size(E2)
    @test E1 ≈ E2
    E1 = expv_timestep(ts, A, b)
    E2 = Array(expv_timestep(ts, A_gpu, b_gpu))
    @test E1 ≈ E2
end