"""
    ExpMethodHigham2005(A::AbstractMatrix);
    ExpMethodHigham2005(b::Bool=true);

Computes the matrix exponential using the algorithm Higham, N. J. (2005). "The scaling and squaring method for the matrix exponential revisited." SIAM J. Matrix Anal. Appl.Vol. 26, No. 4, pp. 1179–1193" based on generated code. If a matrix is specified, balancing is determined automatically.
"""
struct ExpMethodHigham2005
    do_balancing::Bool
end
ExpMethodHigham2005(A::AbstractMatrix) = ExpMethodHigham2005(A isa StridedMatrix)
ExpMethodHigham2005() = ExpMethodHigham2005(true)
ExpMethodHigham2005(A::GPUArraysCore.AbstractGPUArray) = ExpMethodHigham2005(false)

# Holds the generated-code memory slots plus, when the element type supports it,
# a cached LinearSolve workspace for the Padé denominator solve. `getmem` sees
# only `slots`, so the generated code is unchanged apart from passing `linsolve`.
struct Higham2005Cache{V <: AbstractVector, L}
    slots::V
    linsolve::L
end

# LinearSolve's default algorithm choice with alias_A/alias_b: size-selects the
# fastest LU (RecursiveFactorization when loaded), refactorizes in place. Only
# for dense strided BLAS matrices; other types (GPU, BigFloat, ...) keep the
# `lu!` fallback in `ldiv_for_generated!`.
function _pade_linsolve(A::StridedMatrix{<:LinearAlgebra.BlasFloat})
    A isa GPUArraysCore.AbstractGPUArray && return nothing
    Abuf = similar(A)
    Bbuf = similar(A)
    return LinearSolve.init(
        LinearProblem(Abuf, Bbuf);
        alias = LinearSolve.LinearAliasSpecifier(alias_A = true, alias_b = true)
    )
end
_pade_linsolve(A) = nothing

function alloc_mem(A, ::ExpMethodHigham2005)
    T = eltype(A)
    scale = T <: LinearAlgebra.BlasFloat ? similar(A, real(T), size(A, 1)) : nothing
    return Higham2005Cache([similar(A) for i in 1:5], _pade_linsolve(A)), scale
end

# Import the generated code
include("exp_generated/exp_1.jl")
include("exp_generated/exp_2.jl")
include("exp_generated/exp_3.jl")
include("exp_generated/exp_4.jl")
include("exp_generated/exp_5.jl")
include("exp_generated/exp_6.jl")
include("exp_generated/exp_7.jl")
include("exp_generated/exp_8.jl")
include("exp_generated/exp_9.jl")
include("exp_generated/exp_10.jl")
include("exp_generated/exp_11.jl")
include("exp_generated/exp_12.jl")
include("exp_generated/exp_13.jl")

getmem(cache, k) = cache[k - 1] # Called from generated code
getmem(cache::Higham2005Cache, k) = cache.slots[k - 1]

# C = A \ B. Called from generated code, threading the cache's `linsolve`.
function ldiv_for_generated!(C, A, B, linsolve) # cached LinearSolve path
    linsolve.A = A # alias the denominator slot: factorized in place
    linsolve.b = B
    sol = LinearSolve.solve!(linsolve)
    copyto!(C, sol.u)
    return C
end
function ldiv_for_generated!(C, A, B, ::Nothing) # lu! fallback (GPU, BigFloat, ...)
    F = lu!(A)
    ldiv!(F, B) # Result stored in B
    if (pointer_from_objref(C) != pointer_from_objref(B)) # Aliasing allowed
        copyto!(C, B)
    end
    return C
end

const RHO_V = (0.015, 0.25, 0.95, 2.1, 5.4, 10.8, 21.6, 43.2, 86.4, 172.8, 345.6, 691.2)

# From LinearAlgebra
const LIBLAPACK = BLAS.libblastrampoline
using LinearAlgebra: BlasInt, checksquare
for (gebal, gebak, elty, relty) in (
        (:dgebal_, :dgebak_, :Float64, :Float64),
        (:sgebal_, :sgebak_, :Float32, :Float32),
        (:zgebal_, :zgebak_, :ComplexF64, :Float64),
        (:cgebal_, :cgebak_, :ComplexF32, :Float32),
    )
    @eval begin
        #     SUBROUTINE DGEBAL( JOB, N, A, LDA, ILO, IHI, SCALE, INFO )
        #*     .. Scalar Arguments ..
        #      CHARACTER          JOB
        #      INTEGER            IHI, ILP, INFO, LDA, N
        #     .. Array Arguments ..
        #      DOUBLE PRECISION   A( LDA, * ), SCALE( * )
        function gebal_noalloc!(job::AbstractChar, A::AbstractMatrix{$elty}, scale)
            BLAS.chkstride1(A)
            n = checksquare(A)
            LAPACK.chkfinite(A) # balancing routines don't support NaNs and Infs
            ihi = Ref{BlasInt}()
            ilo = Ref{BlasInt}()
            info = Ref{BlasInt}()
            ccall(
                (BLAS.@blasfunc($gebal), LIBLAPACK), Cvoid,
                (
                    Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{BlasInt}, Clong,
                ),
                job, n, A, max(1, stride(A, 2)), ilo, ihi, scale, info, 1
            )
            LAPACK.chklapackerror(info[])
            return ilo[], ihi[], scale
        end
    end
end

# Inplace add of a UniformScaling object (support julia 1.6.2)
@inline function inplace_add!(A, B::UniformScaling) # Called from generated code
    s = B.λ
    return if A isa GPUArraysCore.AbstractGPUArray
        A .= A + s * I
    else
        @inbounds for i in diagind(A)
            A[i] += s
        end
    end
end
function exponential!(A, method::ExpMethodHigham2005, _cache = alloc_mem(A, method))
    cache, _scale = _cache
    n = LinearAlgebra.checksquare(A)
    nA = opnorm(A, 1)

    # Maybe to balancing. `ilo`/`ihi`/`scale` are seeded with no-op defaults so they are
    # always defined before the symmetric undo block below; the two `do_balancing`
    # branches are not provably correlated to the compiler, so without these seeds the
    # undo block reads possibly-undefined locals (flagged by JET typo-mode).
    ilo = 1
    ihi = n
    scale = _scale
    prow = nothing  # row/col permutations from the GenericSchur (non-BLAS) balancing path
    pcol = nothing
    if method.do_balancing
        if A isa StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}
            ilo, ihi, scale = gebal_noalloc!('B', A, _scale)    # modifies A and _scale
        else
            A, bal = GenericSchur.balance!(A)
            ilo, ihi, scale = bal.ilo, bal.ihi, bal.D
            prow, pcol = bal.prow, bal.pcol
        end
    end

    # Make the call to the appropriate exp_gen! function
    X = Base.Cartesian.@nif 13 d -> begin
        nA < RHO_V[d]
    end d -> begin # if condition
        exp_gen!(cache, A, Val(d))
    end d -> begin # fallback (d == 13)
        exp_gen!(cache, A, Val(d))
    end

    # Undo the balancing
    if method.do_balancing
        for j in ilo:ihi
            scj = scale[j]
            for i in 1:n
                X[j, i] *= scj
            end
            for i in 1:n
                X[i, j] /= scj
            end
        end

        if A isa StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}
            if ilo > 1       # apply lower permutations in reverse order
                for j in (ilo - 1):-1:1
                    LinearAlgebra.rcswap!(j, Int(scale[j]), X)
                end
            end
            if ihi < n       # apply upper permutations in forward order
                for j in (ihi + 1):n
                    LinearAlgebra.rcswap!(j, Int(scale[j]), X)
                end
            end
        else
            if ilo > 1       # apply lower permutations in reverse order
                for j in (ilo - 1):-1:1
                    LinearAlgebra.rcswap!(j, prow[j], X)
                end
            end
            if ihi < n       # apply upper permutations in forward order
                for j in (ihi + 1):n
                    LinearAlgebra.rcswap!(j, pcol[j - ihi], X)
                end
            end
        end
    end

    return X
end
