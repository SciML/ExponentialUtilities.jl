using ExponentialUtilities, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(ExponentialUtilities)
    Aqua.test_ambiguities(ExponentialUtilities, recursive = false)
    Aqua.test_deps_compat(
        ExponentialUtilities,
        ignore = [:libblastrampoline_jll]
    )
    Aqua.test_piracies(ExponentialUtilities)
    Aqua.test_project_extras(ExponentialUtilities)
    Aqua.test_stale_deps(ExponentialUtilities)
    Aqua.test_unbound_args(ExponentialUtilities)
    Aqua.test_undefined_exports(ExponentialUtilities)
end

# Analyze only ExponentialUtilities' own code. Without this, JET on Julia 1.12 traces
# into LinearAlgebra/Base internals (e.g. `norm(::Vector)` -> `norm_recursive_check`,
# and the broadcast `unalias`/`copyto_unaliased!` path over `Adjoint{T, Union{}}`) and
# reports abstract-interpretation artifacts there that are not under this package's
# control. Scoping to `ExponentialUtilities` keeps full coverage of this package's code
# (it still flags real `may be undefined` findings here) without asserting that all of
# the stdlib is JET-clean.
const JET_TARGET = (ExponentialUtilities,)

@testset "JET static analysis" begin
    @testset "expv" begin
        rep = JET.report_call(
            expv, (Float64, Matrix{Float64}, Vector{Float64}); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "arnoldi" begin
        rep = JET.report_call(
            arnoldi, (Matrix{Float64}, Vector{Float64}); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "phi" begin
        rep = JET.report_call(phi, (Matrix{Float64}, Int); target_modules = JET_TARGET)
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "exponential!" begin
        rep = JET.report_call(
            ExponentialUtilities.exponential!, (Matrix{Float64},); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "phiv" begin
        rep = JET.report_call(
            phiv, (Float64, Matrix{Float64}, Vector{Float64}, Int); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "kiops" begin
        rep = JET.report_call(
            kiops, (Float64, Matrix{Float64}, Vector{Float64}); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "expv_timestep" begin
        rep = JET.report_call(
            expv_timestep, (Float64, Matrix{Float64}, Vector{Float64}); target_modules = JET_TARGET
        )
        @test length(JET.get_reports(rep)) == 0
    end
end
