using Pkg
using SafeTestsets, Test
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Quality Assurance" include("qa.jl")
        @time @safetestset "Basic Tests" include("basictests.jl")
    end

    if GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "GPU Tests" include("gpu/gputests.jl")
    end
end
