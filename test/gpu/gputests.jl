using CUDA: cu
using LinearAlgebra
using ExponentialUtilities: inplace_add!, exponential!, ExpMethodHigham2005

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

    # Iterate over GPU-compatible methods
    for m in (ExpMethodHigham2005(false),)
        @testset "GPU Exponential, $(string(m))" begin
            eA_d = exponential!(copy(A_d), m)

            @test collect(eA_d) ≈ eA
        end
    end
end
