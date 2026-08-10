# Alternative Krylov phiv methods using error estimates of Saad to automatically
# terminate Arnoldi/Lanczos iterations.
# Currently only expv for Lanczos is implemented.

########################################
# Cache types
abstract type SubspaceCache{T} end
abstract type HermitianSubspaceCache{T} <: SubspaceCache{T} end

"""
    StegrCache(T, n::Integer)

Subspace-exponential cache for the error-estimate variant of [`expv!`](@ref)
(the `mode = :error_estimate` path) on Hermitian operators with element type
`T`. It is a concrete `HermitianSubspaceCache` sized for a Lanczos subspace of
dimension up to `n`, and it uses the symmetric-tridiagonal eigensolver to
compute the exponential of the tridiagonal subspace matrix on each iteration.
Construct one directly, or let
[`get_subspace_cache`](@ref) build the right cache for a given
`KrylovSubspace`.

# Arguments

  - `T`: element type of the propagated subspace vector.
  - `n::Integer`: maximum Lanczos subspace dimension.

# Returns

A `StegrCache` sized for subspaces through dimension `n`.

# Fields

  - `v::Vector{T}`: the subspace-propagated vector (length `n`) that holds the
    result of applying the subspace exponential.
  - `w::Vector{T}`: scratch vector (length `n`) for intermediate values.

# Examples

```julia
cache = StegrCache(ComplexF64, 30)
```
"""
mutable struct StegrCache{T, R <: Real} <: HermitianSubspaceCache{T}
    v::Vector{T} # Subspace-propagated vector
    w::Vector{T}
    function StegrCache(::Type{T}, n::Integer) where {T}
        return new{T, real(T)}(Vector{T}(undef, n), Vector{T}(undef, n))
    end
end

"""
    expT!(α, β, t, cache)

Calculate the subspace exponential `exp(t*T)` for a tridiagonal
subspace matrix `T` with `α` on the diagonal and `β` on the
super-/subdiagonal.
"""
function expT!(
        α::AbstractVector{R}, β::AbstractVector{R}, t::Number,
        cache::StegrCache{T, R}
    ) where {T, R <: Real}
    F = eigen!(SymTridiagonal(α, β))
    sel = 1:length(α)
    @inbounds for i in sel
        cache.w[i] = exp(t * F.values[i]) * F.vectors[1, i]
    end
    return mul!(@view(cache.v[sel]), @view(F.vectors[sel, sel]), @view(cache.w[sel]))
end

"""
    get_subspace_cache(Ks::KrylovSubspace) -> SubspaceCache

Construct the subspace-exponential cache appropriate for the Krylov subspace
`Ks`, for use with the error-estimate variant of [`expv!`](@ref). For a real
(Hermitian) subspace this returns a [`StegrCache`](@ref) sized to `Ks.maxiter`,
which uses a symmetric-tridiagonal eigensolver. Non-Hermitian
(complex) subspaces are not yet supported and raise an error.

# Arguments

  - `Ks`: a populated or reusable [`KrylovSubspace`](@ref) with real-valued
    Hessenberg coefficients.

# Returns

A [`StegrCache`](@ref) that can be supplied to the error-estimate `expv!`
method. This function is a cache constructor, not an extension point.

# Examples

```julia
Ks = KrylovSubspace{ComplexF64, Float64}(100, 30)
cache = get_subspace_cache(Ks)
```
"""
function get_subspace_cache(Ks::KrylovSubspace{T, U}) where {T, U <: Complex}
    error("Subspace exponential caches not yet available for non-Hermitian matrices.")
end
function get_subspace_cache(Ks::KrylovSubspace{T, U}) where {T, U <: Real}
    return StegrCache(T, Ks.maxiter)
end

########################################
# Phiv with error estimate as termination condition
"""
    expv!(w, t, A, b, Ks, cache)

Calculate the action of `exp(t*A)` on `b`, storing the result in `w`. The
Krylov iteration terminates when its subspace error estimate is below the
requested tolerance. This method currently supports Hermitian operators and a
cache returned by [`get_subspace_cache`](@ref).

# Arguments

  - `w`: output vector, overwritten in place.
  - `t`: scalar time or scale factor.
  - `A`: Hermitian matrix or matrix-free operator.
  - `b`: input vector.
  - `Ks`: reusable `KrylovSubspace` with real recurrence coefficients.
  - `cache`: subspace-exponential workspace from [`get_subspace_cache`](@ref).

# Keywords

  - `atol = 1e-8`: absolute stopping tolerance.
  - `rtol = 1e-4`: relative stopping tolerance, scaled by `norm(b)`.
  - `m`: maximum Lanczos dimension, defaulting to `min(Ks.maxiter, size(A, 1))`.
  - `ishermitian = LinearAlgebra.ishermitian(A)`: must be `true`; non-Hermitian
    error estimation is not implemented.
  - `verbose = false`: print each error estimate when `true`.
  - `expmethod`: reduced matrix-exponential method. Reserved for compatibility.

# Returns

The mutated `w`.

# Examples

```julia
using LinearAlgebra

A = Hermitian([-2.0 1.0; 1.0 -2.0])
b = ComplexF64[1, 0]
Ks = KrylovSubspace{ComplexF64, Float64}(length(b), 2)
cache = get_subspace_cache(Ks)
w = similar(b)
expv!(w, 0.1, A, b, Ks, cache; atol = 1.0e-10, rtol = 1.0e-8)
```
"""
function expv!(
        w::AbstractVector{T}, t::Number, A, b::AbstractVector{T},
        Ks::KrylovSubspace{T, B, B}, cache::HSC;
        atol::B = 1.0e-8, rtol::B = 1.0e-4,
        m = min(Ks.maxiter, size(A, 1)),
        ishermitian::Bool = LinearAlgebra.ishermitian(A),
        verbose::Bool = false,
        expmethod = ExpMethodHigham2005Base()
    ) where {
        B, T <: Number,
        HSC <: HermitianSubspaceCache,
    }
    # TODO: this only implements the Lanczos algorithm for Hermitian matrices
    # ks.H is tridiagonal, required for the expT! function above to call stegr!()
    if !ishermitian
        error("Error estimation not yet available for non-Hermitian matrices.")
    end

    if m > Ks.maxiter
        resize!(Ks, m)
    else
        Ks.m = m # might change if error estimate is below requested tolerance
    end

    V, H = getV(Ks), getH(Ks)
    Ks.beta = norm(b)
    if iszero(Ks.beta)
        Ks.m = 0
        w .= false
        return w
    end
    @. V[:, 1] = b / Ks.beta

    ε = atol + rtol * Ks.beta
    verbose && @printf("Initial norm: β₀ %e, stopping threshold: %e\n", Ks.beta, ε)

    α = @diagview(H)
    β = @diagview(H, -1)
    n = size(V, 1)

    for j in 1:m
        lanczos_step!(j, A, V, α, β)
        expT!(@view(α[1:j]), @view(β[1:j]), t, cache)

        # This is practical error estimate Er₂ from
        #
        #   Saad, Y. (1992). Analysis of some Krylov subspace
        #   approximations. SIAM Journal on Numerical Analysis.
        σ = β[j] * Ks.beta * abs(cache.v[j])
        verbose && @printf("iter %d, α[%d] %e, β[%d] %e, σ %e\n", j, j, α[j], j, β[j], σ)
        if σ < ε
            Ks.m = j
            break
        end
    end
    verbose && println("Krylov subspace size: ", Ks.m)

    return lmul!(Ks.beta, mul!(w, @view(Ks.V[:, 1:(Ks.m)]), @view(cache.v[1:(Ks.m)])))
end
