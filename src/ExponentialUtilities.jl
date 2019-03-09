module ExponentialUtilities
using LinearAlgebra, SparseArrays, Printf
using LinearAlgebra: exp!, BlasInt
using LinearAlgebra.LAPACK: stegr!

include("utils.jl")
include("phi.jl")
include("arnoldi.jl")
include("krylov_phiv.jl")
include("krylov_phiv_adaptive.jl")
include("kiops.jl")
include("StegrWork.jl")
include("krylov_phiv_error_estimate.jl")

export phi, phi!, KrylovSubspace, arnoldi, arnoldi!, lanczos!, ExpvCache, PhivCache,
    expv, expv!, phiv, phiv!, kiops, expv_timestep, expv_timestep!, phiv_timestep, phiv_timestep!,
    StegrCache, get_subspace_cache
end
