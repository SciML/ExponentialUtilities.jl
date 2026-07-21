using SciMLTesting, ExponentialUtilities, JET, Test

run_qa(
    ExponentialUtilities;
    api_docs_kwargs = (; rendered = true),
    explicit_imports = true,
    aqua_kwargs = (; deps_compat = (; ignore = [:libblastrampoline_jll])),
    ei_kwargs = (;
        # Names owned elsewhere but reached through LinearAlgebra.BLAS.
        all_qualified_accesses_via_owners = (;
            ignore = (:BlasFloat, :chkstride1, :libblastrampoline),
        ),
        # Non-public names of Base / LinearAlgebra(.BLAS/.LAPACK) accessed qualified.
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@aliasscope"), Symbol("@assume_effects"),
                Symbol("@blasfunc"), Symbol("@propagate_inbounds"),
                :BlasFloat, :Cartesian, :Const, :Experimental,
                :checksquare, :chkfinite, :chklapackerror, :chkstride1,
                :gebal!, :gesv!, :libblastrampoline, :rcswap!, :stegr!,
            ),
        ),
        # Non-public names explicitly imported from LinearAlgebra(.BLAS/.LAPACK,
        # incl. the Stegr submodule) / ArrayInterface / Base.
        all_explicit_imports_are_public = (;
            ignore = (
                :BlasInt, :checksquare, :allowed_setindex!, :ismutable, :typename,
                Symbol("@blasfunc"), :stegr!,
            ),
        ),
    ),
    # Heavy `using LinearAlgebra, SparseArrays, Printf, PrecompileTools` brings ~31
    # names implicitly; making them explicit is a large refactor tracked separately.
    ei_broken = (:no_implicit_imports,)
) # SciML/ExponentialUtilities.jl#231
