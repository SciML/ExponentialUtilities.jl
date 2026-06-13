using SafeTestsets, Test
using SciMLTesting

run_tests(;
    core = () -> begin
        @time @safetestset "Quality Assurance" include("qa.jl")
        @time @safetestset "Basic Tests" include("basictests.jl")
    end,
    groups = Dict(
        "GPU" => (; env = joinpath(@__DIR__, "gpu"), body = () -> begin
            @time @safetestset "GPU Tests" include("gpu/gputests.jl")
        end),
        "JET" => (; env = joinpath(@__DIR__, "jet"), body = () -> begin
            @time @safetestset "JET Tests" include("jet/jet.jl")
        end),
    ),
)
