using ExponentialUtilities, JET, Test

@testset "JET static analysis" begin
    # Test key entry points for type stability and correctness
    # Using report_call to check for runtime errors

    @testset "expv" begin
        rep = JET.report_call(expv, (Float64, Matrix{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "arnoldi" begin
        rep = JET.report_call(arnoldi, (Matrix{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "phi" begin
        rep = JET.report_call(phi, (Matrix{Float64}, Int))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "exponential!" begin
        rep = JET.report_call(ExponentialUtilities.exponential!, (Matrix{Float64},))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "phiv" begin
        rep = JET.report_call(phiv, (Float64, Matrix{Float64}, Vector{Float64}, Int))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "kiops" begin
        rep = JET.report_call(kiops, (Float64, Matrix{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "expv_timestep" begin
        rep = JET.report_call(expv_timestep, (Float64, Matrix{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0
    end
end
