using Pkg
using SafeTestsets, Test
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

function activate_env(name)
    Pkg.activate(name)
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("basictests.jl")
    end

    if GROUP == "QA"
        activate_env("qa")
        @time @safetestset "Quality Assurance" include("qa/qa.jl")
    end

    if GROUP == "GPU"
        activate_env("gpu")
        @time @safetestset "GPU Tests" include("gpu/gputests.jl")
    end
end
