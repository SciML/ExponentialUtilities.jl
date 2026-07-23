module ExponentialUtilities
import LinearAlgebra
import LinearAlgebra: BLAS, Diagonal, Hermitian, I, SingularException,
    SymTridiagonal, UniformScaling, axpy!, diagind, diagview, dot, eigen!,
    ishermitian, ldiv!, lmul!, lu!, mul!, norm, opnorm, rdiv!, rmul!
import SparseArrays
import SparseArrays: AbstractSparseArray, AbstractSparseMatrix, SparseMatrixCSC, nnz
import Printf
import Printf: @printf
using ArrayInterface: ismutable, allowed_setindex!, fast_scalar_indexing
import PrecompileTools
import PrecompileTools: @compile_workload, @setup_workload
import GenericSchur
import GPUArraysCore
import Adapt

const BlasFloat = Union{Float32, Float64, ComplexF32, ComplexF64}

function checksquare(A)
    n, m = size(A)
    n == m || throw(DimensionMismatch("matrix must be square, got size $(size(A))"))
    return n
end

function rcswap!(i::Integer, j::Integer, A::AbstractMatrix)
    i == j && return A
    @inbounds for k in axes(A, 2)
        A[i, k], A[j, k] = A[j, k], A[i, k]
    end
    @inbounds for k in axes(A, 1)
        A[k, i], A[k, j] = A[k, j], A[k, i]
    end
    return A
end

"""
    @diagview(A,d) -> view of the `d`th diagonal of `A`.
"""
macro diagview(A, d::Integer = 0)
    s = d <= 0 ? 1 + abs(d) : :(m + $d)
    return quote
        m = size($(esc(A)), 1)
        @view($(esc(A))[($s):(m + 1):end])
    end
end

include("exp.jl")
include("exp_baseexp.jl")
include("exp_noalloc.jl")
include("exp_generic.jl")
include("exp_sparse.jl")
include("phi.jl")
include("phi_almohy.jl")
include("arnoldi.jl")
include("krylov_phiv.jl")
include("krylov_phiv_adaptive.jl")
include("kiops.jl")
include("krylov_phiv_error_estimate.jl")
# precompile script
include("precompile.jl")

export phi, phi!, KrylovSubspace, arnoldi, arnoldi!, lanczos!, ExpvCache, PhivCache,
    expv, expv!, phiv, phiv!, kiops, expv_timestep, expv_timestep!, phiv_timestep,
    phiv_timestep!,
    StegrCache, PhiPadeCache, get_subspace_cache, exponential!
export ExpMethodHigham2005,
    ExpMethodHigham2005Base, ExpMethodGeneric, ExpMethodNative,
    ExpMethodDiagonalization

end
