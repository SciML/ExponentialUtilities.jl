#

# Fallback
"""
    cache = alloc_mem(A, method)

Pre-allocates memory associated with matrix exponential function `method` and matrix `A`. To be used in combination with [`exponential!`](@ref).
"""
function alloc_mem(A, method)
    return nothing
end

# Immutable matrices (e.g. StaticArrays' SMatrix) cannot use the in-place
# ExpMethodHigham2005 default, so route them to the non-mutating generic method.
function exponential!(A)
    return if ismutable(A)
        exponential!(A, ExpMethodHigham2005(A))
    else
        exponential!(A, ExpMethodGeneric())
    end
end
function exponential!(A::GPUArraysCore.AbstractGPUArray)
    return exponential!(A, ExpMethodHigham2005(false))
end;

## The diagonalization based
"""
    ExpMethodDiagonalization(enforce_real = true)

Matrix-exponential method based on diagonalization with `eigen`.

# Arguments

  - `enforce_real`: when `true` (the default), discard small imaginary parts
    introduced by numerical diagonalization for real input matrices.

# Fields

  - `enforce_real::Bool`: whether real input receives a real-valued result.
"""
struct ExpMethodDiagonalization
    enforce_real::Bool
end
ExpMethodDiagonalization() = ExpMethodDiagonalization(true);

"""
    E = exponential!(A, [method [cache]])

Compute a matrix exponential, mutating `A` when the selected method supports
in-place evaluation.

# Arguments

  - `A`: square, non-sparse matrix. Mutable inputs are overwritten by most
    methods; immutable inputs use [`ExpMethodGeneric`](@ref) and are returned
    out of place.
  - `method`: exponential implementation, such as [`ExpMethodNative`](@ref),
    [`ExpMethodDiagonalization`](@ref), [`ExpMethodGeneric`](@ref), or
    [`ExpMethodHigham2005`](@ref).
  - `cache`: optional workspace from `alloc_mem(A, method)` for repeated calls.

# Returns

The matrix exponential. For mutable input and in-place methods, this is the
mutated `A`.

If no `method` is given, immutable matrices (e.g. StaticArrays' `SMatrix`) are
computed out-of-place with [`ExpMethodGeneric`](@ref) and the result is returned
without modifying `A`.

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

Matrix-exponential method that delegates to `Base.exp`.
"""
struct ExpMethodNative end
function exponential!(A, method::ExpMethodNative, cache = nothing)
    return exp(A)
end
