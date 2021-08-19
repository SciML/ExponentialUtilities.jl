#


# Fallback
function alloc_mem(A,method)
    return nothing;
end


## The diagonalization based
"""
    ExpMethodDiagonalization()

Matrix exponential method corresponding to the diagonalization with `eigen`.

"""
struct ExpMethodDiagonalization
end
function _exp!(A,method::ExpMethodDiagonalization,cache)
    F=eigen!(A)
    copyto!(A,F.vectors*Diagonal(exp.(F.values))/F.vectors)
    return A
end



"""
    ExpMethodNative()

Matrix exponential method corresponding to calling `Base.exp`.

"""
struct ExpMethodNative
end
function _exp!(A,method::ExpMethodNative,cache)
    return exp(A)
end
