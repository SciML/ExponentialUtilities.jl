using SciMLTesting, ExponentialUtilities, JET, Test

run_qa(
    ExponentialUtilities;
    ei_kwargs = (;
        # Non-public LinearAlgebra/BLAS/LAPACK names used by the non-allocating
        # LAPACK balancing wrapper `gebal_noalloc!` (which keeps the CPU matrix
        # exponential allocation-free), plus `Base.promote_op` used to infer the
        # exponential! workspace type at cache construction without allocating.
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :BlasInt, :chkfinite, :chklapackerror,
                :chkstride1, :libblastrampoline, :promote_op,
            ),
        ),
        # `chkstride1` (owned by LinearAlgebra) and `libblastrampoline` (owned by
        # libblastrampoline_jll) are reached through the `LinearAlgebra.BLAS`
        # re-exporter in the same balancing wrapper.
        all_qualified_accesses_via_owners = (;
            ignore = (:chkstride1, :libblastrampoline),
        ),
    ),
)
