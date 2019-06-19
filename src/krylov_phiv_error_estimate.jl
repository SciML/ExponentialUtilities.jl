# Alternative Krylov phiv methods using error estimates of Saad to automatically
# terminate Arnoldi/Lanczos iterations.
# Currently only expv for Lanczos is implemented.

########################################
# Cache types
abstract type SubspaceCache{T} end
abstract type HermitianSubspaceCache{T} <: SubspaceCache{T} end

mutable struct StegrCache{T,R<:Real} <: HermitianSubspaceCache{T}
    v::Vector{T} # Subspace-propagated vector
    w::Vector{T}
    sw::Stegr.StegrWork{R}
    StegrCache(::Type{T}, n::Integer) where T = new{T,real(T)}(
        Vector{T}(undef, n), Vector{T}(undef, n),
        Stegr.StegrWork(real(T), n))
end

"""
    expT!(α, β, t, cache)

Calculate the subspace exponential `exp(t*T)` for a tridiagonal
subspace matrix `T` with `α` on the diagonal and `β` on the
super-/subdiagonal, diagonalizing via `stegr!`.
"""
function expT!(α::AbstractVector{R}, β::AbstractVector{R}, t::Number,
               cache::StegrCache{T,R}) where {T,R<:Real}
    LAPACK.stegr!(α, β, cache.sw)
    sel = 1:length(α)
    @inbounds for i = sel
        cache.w[i] = exp(t*cache.sw.w[i])*cache.sw.Z[1,i]
    end
    mul!(@view(cache.v[sel]), @view(cache.sw.Z[sel,sel]), @view(cache.w[sel]))
end

get_subspace_cache(Ks::KrylovSubspace{B,T,U}) where {B,T,U<:Complex} =
    error("Subspace exponential caches not yet available for non-Hermitian matrices.")
get_subspace_cache(Ks::KrylovSubspace{B,T,U}) where {B,T,U<:Real} =
    StegrCache(T, Ks.maxiter)

########################################
# Phiv with error estimate as termination condition
"""
    expv!(w, t, A, b, Ks, cache)

Alternative interface for calculating the action of `exp(t*A)` on the
vector `b`, storing the result in `w`. The Krylov iteration is
terminated when an error estimate for the matrix exponential in the
generated subspace is below the requested tolerance. `Ks` is a
`KrylovSubspace` and `typeof(cache)<:HermitianSubspaceCache`, the
exact type decides which algorithm is used to compute the subspace
exponential.
"""
function expv!(w::AbstractVector{T}, t::Number, A, b::AbstractVector{T},
               Ks::KrylovSubspace{B, T, B}, cache::HSC;
               atol::B=1.0e-8, rtol::B=1.0e-4,
               m=min(Ks.maxiter, size(A,1)),
               verbose::Bool=false) where {B, T <: Number, HSC <: HermitianSubspaceCache}
    if m > Ks.maxiter
        resize!(Ks, m)
    else
        Ks.m = m # might change if error estimate is below requested tolerance
    end

    V, H = getV(Ks), getH(Ks)
    Ks.beta = norm(b)
    @. V[:, 1] = b / Ks.beta

    ε = atol + rtol * Ks.beta
    verbose && @printf("Initial norm: β₀ %e, stopping threshold: %e\n", Ks.beta, ε)

    α = @diagview(H)
    β = @diagview(H,-1)
    n = size(V, 1)

    for j ∈ 1:m
        lanczos_step!(j, A, V, α, β)
        expT!(@view(α[1:j]), @view(β[1:j]), t, cache)

        # This is practical error estimate Er₂ from
        #
        #   Saad, Y. (1992). Analysis of some Krylov subspace
        #   approximations. SIAM Journal on Numerical Analysis.
        σ = β[j]*Ks.beta*abs(cache.v[j])
        verbose && @printf("iter %d, α[%d] %e, β[%d] %e, σ %e\n",j, j, α[j], j, β[j], σ)
        if σ < ε
            Ks.m = j
            break
        end
    end
    verbose && println("Krylov subspace size: ", Ks.m)

    lmul!(Ks.beta, mul!(w, @view(Ks.V[:,1:Ks.m]), @view(cache.v[1:Ks.m])))
end
