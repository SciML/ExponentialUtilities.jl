for expmeth in [
        ExpMethodDiagonalization, ExpMethodGeneric,
        ExpMethodHigham2005, ExpMethodHigham2005Base, ExpMethodNative,
    ]
    @eval function exponential!(A::AbstractSparseArray, method::$expmeth, cache = nothing)
        throw(
            ErrorException(
                "exp(A) on a sparse matrix is generally dense. This operation is " *
                    "not allowed with exponential. If you wished to compute exp(At)*v, see expv. " *
                    "Otherwise to override this error, densify the matrix before calling, " *
                    "i.e. exponential!(Matrix(A))"
            )
        )
    end
end
