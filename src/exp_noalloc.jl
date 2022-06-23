

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

function alloc_mem(A, ::ExpMethodHigham2005)
    T = eltype(A)
    scale = T <: LinearAlgebra.BlasFloat ? similar(A, real(T), size(A, 1)) : nothing
    return [similar(A) for i = 1:5], scale
end


# Import the generated code
for i = 1:13
    include("exp_generated/exp_$i.jl")
end

function getmem(cache, k) # Called from generated code
    return cache[k-1]
end
function ldiv_for_generated!(C, A, B) # C=A\B. Called from generated code
    F = lu!(A) # This allocation is unavoidable, due to the interface of LinearAlgebra
    ldiv!(F, B) # Result stored in B
    if (pointer_from_objref(C) != pointer_from_objref(B)) # Aliasing allowed
        copyto!(C, B)
    end
    return C
end

const RHO_V = (0.015, 0.25, 0.95, 2.1, 5.4, 10.8, 21.6, 43.2, 86.4, 172.8, 345.6, 691.2)

# From LinearAlgebra
const LIBLAPACK = VERSION >= v"1.7" ? BLAS.libblastrampoline : LAPACK.liblapack
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
                (BLAS.@blasfunc($gebal), LIBLAPACK),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ptr{BlasInt},
                    Ptr{$relty},
                    Ptr{BlasInt},
                    Clong,
                ),
                job,
                n,
                A,
                max(1, stride(A, 2)),
                ilo,
                ihi,
                scale,
                info,
                1,
            )
            LAPACK.chklapackerror(info[])
            ilo[], ihi[], scale
        end
    end
end

# Inplace add of a UniformScaling object (support julia 1.6.2)
@inline function inplace_add!(A, B::UniformScaling) # Called from generated code
    s = B.λ
    @inbounds for i in diagind(A)
        A[i] += s
    end
end
function exponential!(A, method::ExpMethodHigham2005, _cache = alloc_mem(A, method))
    cache, _scale = _cache
    n = LinearAlgebra.checksquare(A)
    nA = opnorm(A, 1)

    # Maybe to balancing
    if method.do_balancing
        if A isa StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}
            ilo, ihi, scale = gebal_noalloc!('B', A, _scale)    # modifies A and _scale
        else
            A, bal = GenericSchur.balance!(A)
            ilo, ihi, scale = bal.ilo, bal.ihi, bal.D
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
        for j = ilo:ihi
            scj = scale[j]
            for i = 1:n
                X[j, i] *= scj
            end
            for i = 1:n
                X[i, j] /= scj
            end
        end

        if A isa StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}
            if ilo > 1       # apply lower permutations in reverse order
                for j = (ilo-1):-1:1
                    LinearAlgebra.rcswap!(j, Int(scale[j]), X)
                end
            end
            if ihi < n       # apply upper permutations in forward order
                for j = (ihi+1):n
                    LinearAlgebra.rcswap!(j, Int(scale[j]), X)
                end
            end
        else
            if ilo > 1       # apply lower permutations in reverse order
                for j = (ilo-1):-1:1
                    LinearAlgebra.rcswap!(j, bal.prow[j], X)
                end
            end
            if ihi < n       # apply upper permutations in forward order
                for j = (ihi+1):n
                    LinearAlgebra.rcswap!(j, bal.pcol[j-ihi], X)
                end
            end
        end
    end

    return X
end
