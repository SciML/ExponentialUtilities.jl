#

# Fallback
"""
    cache=alloc_mem(A,method)

Pre-allocates memory associated with matrix exponential function `method` and matrix `A`. To be used in combination with [`exponential!`](@ref).
"""
function alloc_mem(A, method)
    return nothing
end

@deprecate _exp! exponential!
@deprecate exp_generic exponential!
exponential!(A) = exponential!(A, ExpMethodHigham2005(A));
function exponential!(A::GPUArraysCore.AbstractGPUArray)
    exponential!(A, ExpMethodHigham2005(false))
end;

## The diagonalization based
"""
    ExpMethodDiagonalization(enforce_real=true)

Matrix exponential method corresponding to the diagonalization with `eigen` possibly by removing imaginary part introduced by the numerical approximation.
"""
struct ExpMethodDiagonalization
    enforce_real::Bool
end
ExpMethodDiagonalization() = ExpMethodDiagonalization(true);

"""
    E=exponential!(A,[method [cache]])

Computes the matrix exponential with the method specified in `method`. The contents of `A` are modified, allowing for fewer allocations. The `method` parameter specifies the implementation and implementation parameters, e.g. [`ExpMethodNative`](@ref), [`ExpMethodDiagonalization`](@ref), [`ExpMethodGeneric`](@ref), [`ExpMethodHigham2005`](@ref). Memory
needed can be preallocated and provided in the parameter `cache` such that the memory can be recycled when calling `exponential!` several times. The preallocation is done with the command [`alloc_mem`](@ref): `cache=alloc_mem(A,method)`.

Example

```julia-repl
julia> A = randn(50, 50);

julia> B = A * 2;

julia> method = ExpMethodHigham2005();

julia> cache = ExponentialUtilities.alloc_mem(A, method); # Main allocation done here

julia> E1 = exponential!(A, method, cache) # Very little allocation here

julia> E2 = exponential!(B, method, cache) # Very little allocation here

```
"""
function exponential!(A, method::ExpMethodDiagonalization, cache = nothing)
    F = eigen!(A)
    E = F.vectors * Diagonal(exp.(F.values)) / F.vectors
    if (method.enforce_real && isreal(A))
        E = real.(E)
    end
    copyto!(A, E)
    return A
end

"""
    ExpMethodNative()

Matrix exponential method corresponding to calling `Base.exp`.
"""
struct ExpMethodNative end
function exponential!(A, method::ExpMethodNative, cache = nothing)
    return exp(A)
end

function exponential!(A::AbstractSparseArray,method=nothing, cache=nothing)
    throw("exp(A) on a sparse matrix is generally dense. This operation is "*
    "not allowed with exponential. If you wished to compute exp(At)*v, see expv. "*
    "Otherwise to override this error, densify the matrix before calling, "*
    "i.e. exponential!(Matrix(A))")
end
