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
dimension up to `n`. Every iteration diagonalizes the tridiagonal subspace
matrix with a preallocated implicit-QL eigensolver, so applying the subspace
exponential allocates nothing. Construct one directly, or let
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
  - `d::Vector{R}`, `e::Vector{R}`: working copies (length `n`) of the
    tridiagonal diagonal and off-diagonal, overwritten by the eigensolver, where
    `R = real(T)`.
  - `Z::Matrix{R}`: `n × n` eigenvector workspace.

# Examples

```julia
cache = StegrCache(ComplexF64, 30)
```
"""
mutable struct StegrCache{T, R <: Real} <: HermitianSubspaceCache{T}
    v::Vector{T} # Subspace-propagated vector
    w::Vector{T}
    d::Vector{R}
    e::Vector{R}
    Z::Matrix{R}
    function StegrCache(::Type{T}, n::Integer) where {T}
        R = real(T)
        return new{T, R}(
            Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{R}(undef, n), Vector{R}(undef, n), Matrix{R}(undef, n, n)
        )
    end
end

"""
    symtridiag_eigen!(d, e, Z)

Diagonalize the real symmetric tridiagonal matrix with diagonal `d` and
off-diagonal `e` (`e[i]` couples rows `i` and `i + 1`; `e[end]` is scratch)
by implicit QL iteration with Wilkinson shifts. On return `d` holds the
eigenvalues, in no particular order, and `Z` holds the corresponding
eigenvectors in its columns, provided `Z` was the identity on entry. The
rotations are accumulated into whatever `Z` contains, so passing a
non-identity `Z` yields `Z * Q`. Works in place on the given buffers and
allocates nothing.
"""
function symtridiag_eigen!(
        d::AbstractVector{R}, e::AbstractVector{R}, Z::AbstractMatrix{R}
    ) where {R <: Real}
    n = length(d)
    n == 0 && return d, Z
    @inbounds e[n] = zero(R)
    @inbounds for l in 1:n
        iter = 0
        while true
            m = l
            while m < n
                abs(e[m]) <= eps(R) * (abs(d[m]) + abs(d[m + 1])) && break
                m += 1
            end
            m == l && break
            (iter += 1) > 30n && error("symtridiag_eigen!: no convergence after 30n QL iterations")
            g = (d[l + 1] - d[l]) / (2 * e[l])
            r = hypot(g, one(R))
            g = d[m] - d[l] + e[l] / (g + copysign(r, g))
            s = one(R)
            c = one(R)
            p = zero(R)
            underflow = false
            i = m - 1
            while i >= l
                f = s * e[i]
                b = c * e[i]
                r = hypot(f, g)
                e[i + 1] = r
                if iszero(r)
                    d[i + 1] -= p
                    e[m] = zero(R)
                    underflow = true
                    break
                end
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for k in 1:n
                    zk = Z[k, i + 1]
                    Z[k, i + 1] = s * Z[k, i] + c * zk
                    Z[k, i] = c * Z[k, i] - s * zk
                end
                i -= 1
            end
            underflow && continue
            d[l] -= p
            e[l] = g
            e[m] = zero(R)
        end
    end
    return d, Z
end

"""
    expT!(α, β, t, cache)

Calculate the subspace exponential `exp(t*T)` for a tridiagonal
subspace matrix `T` with `α` on the diagonal and `β` on the
super-/subdiagonal. `α` and `β` are copied into the cache first and are not
modified.
"""
function expT!(
        α::AbstractVector{R}, β::AbstractVector{R}, t::Number,
        cache::StegrCache{T, R}
    ) where {T, R <: Real}
    sel = 1:length(α)
    d = @view cache.d[sel]
    e = @view cache.e[sel]
    Z = @view cache.Z[sel, sel]
    copyto!(d, α)
    copyto!(e, β)
    fill!(Z, zero(R))
    @inbounds for i in sel
        Z[i, i] = one(R)
    end
    symtridiag_eigen!(d, e, Z)
    @inbounds for i in sel
        cache.w[i] = exp(t * d[i]) * Z[1, i]
    end
    return mul!(@view(cache.v[sel]), Z, @view(cache.w[sel]))
end

"""
    get_subspace_cache(Ks::KrylovSubspace) -> SubspaceCache

Construct the subspace-exponential cache appropriate for the Krylov subspace
`Ks`, for use with the error-estimate variant of [`expv!`](@ref). For a real
(Hermitian) subspace this returns a [`StegrCache`](@ref) sized to `Ks.maxiter`,
which diagonalizes the tridiagonal subspace matrix with a preallocated
implicit-QL eigensolver. Non-Hermitian
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
