using ExponentialUtilities, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(ExponentialUtilities)
    Aqua.test_ambiguities(ExponentialUtilities, recursive = false)
    Aqua.test_deps_compat(ExponentialUtilities,
        ignore = [:libblastrampoline_jll])
    Aqua.test_piracies(ExponentialUtilities)
    Aqua.test_project_extras(ExponentialUtilities)
    Aqua.test_stale_deps(ExponentialUtilities)
    Aqua.test_unbound_args(ExponentialUtilities)
    Aqua.test_undefined_exports(ExponentialUtilities)
end
