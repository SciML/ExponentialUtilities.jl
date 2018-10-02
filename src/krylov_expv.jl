using Printf

abstract type SubspaceCache{T} end
abstract type HermitianSubspaceCache{T} <: SubspaceCache{T} end

include("stegr_cache.jl")

function expv!(w::AbstractVector{T}, t::Number, A, b::AbstractVector{T},
               Ks::KrylovSubspace{B, T, B}, cache::HSC;
               atol::B=1.0e-8, rtol::B=1.0e-4,
               m=min(Ks.maxiter, size(A,1)),
               verbose::Bool=false) where {B, T <: Number, HSC <: HermitianSubspaceCache}
    if m > Ks.maxiter
        resize!(Ks, m)
    else
        Ks.m = m # might change if happy-breakdown occurs
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
        lanczos_step!(j, m, n, A, V, α, β)
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
