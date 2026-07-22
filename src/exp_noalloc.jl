"""
    ExpMethodHigham2005(A::AbstractMatrix);
    ExpMethodHigham2005(b::Bool = true);

Matrix-exponential method using Higham's 2005 scaling-and-squaring Padé
algorithm and generated evaluation kernels.

# Arguments

  - `do_balancing`: whether to balance a suitable dense matrix before the Padé
    evaluation. The no-argument constructor defaults to `true`; the matrix
    constructor selects balancing only for strided matrices.

# Fields

  - `do_balancing::Bool`: whether to apply matrix balancing.
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
function _pade_linsolve(A::StridedMatrix{<:BlasFloat})
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
    scale = T <: BlasFloat ? similar(A, real(T), size(A, 1)) : nothing
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
    n = checksquare(A)
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
        A, bal = GenericSchur.balance!(A)
        ilo, ihi, scale = bal.ilo, bal.ihi, bal.D
        prow, pcol = bal.prow, bal.pcol
    end

    # Make the call to the appropriate exp_gen! function
    d = 13
    for d in 1:12
        if nA < RHO_V[d]
            break
        end
    end
    X = exp_gen!(cache, A, Val(d))

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

        if ilo > 1       # apply lower permutations in reverse order
            for j in (ilo - 1):-1:1
                rcswap!(j, prow[j], X)
            end
        end
        if ihi < n       # apply upper permutations in forward order
            for j in (ihi + 1):n
                rcswap!(j, pcol[j - ihi], X)
            end
        end
    end

    return X
end
