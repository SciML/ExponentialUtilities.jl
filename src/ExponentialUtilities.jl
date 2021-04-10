module ExponentialUtilities
using LinearAlgebra, SparseArrays, Printf, Requires, ChainRulesCore

"""
    @diagview(A,d) -> view of the `d`th diagonal of `A`.
"""
macro diagview(A,d::Integer=0)
    s = d<=0 ? 1+abs(d) : :(m+$d)
    quote
        m = size($(esc(A)),1)
        @view($(esc(A))[($s):m+1:end])
    end
end

include("exp.jl")
include("phi.jl")
include("arnoldi.jl")
include("krylov_phiv.jl")
include("krylov_phiv_adaptive.jl")
include("kiops.jl")
include("StegrWork.jl")
include("krylov_phiv_error_estimate.jl")
include("krylov_phiv_chainrules.jl")

export phi, phi!, KrylovSubspace, arnoldi, arnoldi!, lanczos!, ExpvCache, PhivCache,
    expv, expv!, exp_generic, phiv, phiv!, kiops, expv_timestep, expv_timestep!, phiv_timestep, phiv_timestep!,
    StegrCache, get_subspace_cache

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        function ExponentialUtilities.expv!(w::CuArrays.CuVector{Tw},
                                    t::Real, Ks::KrylovSubspace{T, U};
                                    cache=nothing,
                                    dexpHe::CuArrays.CuVector = CuArrays.CuVector{U}(undef, Ks.m)) where {Tw, T, U}
            m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
            @assert length(w) == size(V, 1) "Dimension mismatch"
            if cache == nothing
                cache = Matrix{U}(undef, m, m)
            elseif isa(cache, ExpvCache)
                cache = get_cache(cache, m)
            else
                throw(ArgumentError("Cache must be an ExpvCache"))
            end
            copyto!(cache, @view(H[1:m, :]))
            if ishermitian(cache)
                # Optimize the case for symtridiagonal H
                F = eigen!(SymTridiagonal(cache))
                expHe = F.vectors * (exp.(lmul!(t,F.values)) .* @view(F.vectors[1, :]))
            else
                lmul!(t, cache); expH = cache
                _exp!(expH)
                expHe = @view(expH[:, 1])
            end

            copyto!(dexpHe, expHe)
            lmul!(beta, mul!(w, @view(V[:, 1:m]), dexpHe)) # exp(A) ≈ norm(b) * V * exp(H)e
        end
    end
    
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        function ExponentialUtilities.expv!(w::CUDA.CuVector{Tw},
                                    t::Real, Ks::KrylovSubspace{T, U};
                                    cache=nothing,
                                    dexpHe::CUDA.CuVector = CUDA.CuVector{U}(undef, Ks.m)) where {Tw, T, U}
            m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
            @assert length(w) == size(V, 1) "Dimension mismatch"
            if cache == nothing
                cache = Matrix{U}(undef, m, m)
            elseif isa(cache, ExpvCache)
                cache = get_cache(cache, m)
            else
                throw(ArgumentError("Cache must be an ExpvCache"))
            end
            copyto!(cache, @view(H[1:m, :]))
            if ishermitian(cache)
                # Optimize the case for symtridiagonal H
                F = eigen!(SymTridiagonal(cache))
                expHe = F.vectors * (exp.(lmul!(t,F.values)) .* @view(F.vectors[1, :]))
            else
                lmul!(t, cache); expH = cache
                _exp!(expH)
                expHe = @view(expH[:, 1])
            end

            copyto!(dexpHe, expHe)
            lmul!(beta, mul!(w, @view(V[:, 1:m]), dexpHe)) # exp(A) ≈ norm(b) * V * exp(H)e
        end
    end
end

end
