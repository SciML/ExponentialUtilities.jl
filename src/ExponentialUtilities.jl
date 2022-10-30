module ExponentialUtilities
using LinearAlgebra, SparseArrays, Printf
using ArrayInterfaceCore: ismutable, allowed_getindex, allowed_setindex!
using ArrayInterfaceGPUArrays
using SnoopPrecompile
import GenericSchur
import GPUArraysCore
import Adapt

using Base: typename

Base.@pure __parameterless_type(T) = typename(T).wrapper
parameterless_type(x) = __parameterless_type(typeof(x))
parameterless_type(::Type{T}) where {T} = __parameterless_type(T)

"""
    @diagview(A,d) -> view of the `d`th diagonal of `A`.
"""
macro diagview(A, d::Integer = 0)
    s = d <= 0 ? 1 + abs(d) : :(m + $d)
    quote
        m = size($(esc(A)), 1)
        @view($(esc(A))[($s):(m + 1):end])
    end
end

include("exp.jl")
include("exp_baseexp.jl")
include("exp_noalloc.jl")
include("exp_generic.jl")
include("phi.jl")
include("arnoldi.jl")
include("krylov_phiv.jl")
include("krylov_phiv_adaptive.jl")
include("kiops.jl")
include("StegrWork.jl")
include("krylov_phiv_error_estimate.jl")
# precompile script
include("precompile.jl")

export phi, phi!, KrylovSubspace, arnoldi, arnoldi!, lanczos!, ExpvCache, PhivCache,
       expv, expv!, phiv, phiv!, kiops, expv_timestep, expv_timestep!, phiv_timestep,
       phiv_timestep!,
       StegrCache, get_subspace_cache, exponential!
export ExpMethodHigham2005, ExpMethodHigham2005Base, ExpMethodGeneric, ExpMethodNative,
       ExpMethodDiagonalization

end
