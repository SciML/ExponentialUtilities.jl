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

# Returns

An `ExpMethodHigham2005` algorithm object for use with [`exponential!`](@ref).

# Examples

```julia
A = [0.0 1.0; -1.0 0.0]
method = ExpMethodHigham2005(A)
exponential!(copy(A), method)
```
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

# Import the generated code: Padé approximants of degree 3, 5, 7, 9 and 13
include("exp_generated/exp_1.jl")
include("exp_generated/exp_2.jl")
include("exp_generated/exp_3.jl")
include("exp_generated/exp_4.jl")
include("exp_generated/exp_5.jl")

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

# Higham's θ_m for the Padé degrees m = 3, 5, 7, 9, 13 of kernels 1 to 5
const RHO_V = (0.015, 0.25, 0.95, 2.1, 5.4)

# The smallest s ≥ 0 with nA < RHO_V[5] * 2^s, read off the binary representation
function pade13_squarings(nA::Union{Float16, Float32, Float64, BigFloat})
    θ = RHO_V[5]
    if isfinite(nA) && nA >= θ
        return exponent(nA) - exponent(θ) + (significand(nA) >= significand(θ))
    else
        return 0
    end
end

# Fallback for other number types, e.g. ForwardDiff duals. The rounded log2 can be off
# by one at an interval edge, so an exact comparison with RHO_V[5] * 2^s settles it.
function pade13_squarings(nA)
    θ = RHO_V[5]
    if isfinite(nA) && nA >= θ
        s = ceil(Int, log2(nA / θ))
        t = ldexp(θ, s)
        if nA >= t
            s += 1
        elseif nA < t / 2
            s -= 1
        end
        return s
    else
        return 0
    end
end

# The degree-13 Padé approximant of 2^-s * A, squared s times, written to A
function exp_pade13!(cache, A, s)
    if s > 0
        coeff = ldexp(one(real(eltype(A))), -s)
        A .= coeff .* A
    end
    exp_gen!(cache, A, Val(5))
    B = getmem(cache, 2)
    npairs, rest = divrem(s, 2)
    for _ in 1:npairs
        mul!(B, A, A)
        mul!(A, B, B)
    end
    if rest == 1
        mul!(B, A, A)
        copyto!(A, B)
    end
    return A
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
    n = checksquare(A)

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
    nA = opnorm(A, 1)  # after balancing, since the kernel runs on the balanced matrix

    # RHO_V is tuned for double precision, so a type that resolves more digits keeps
    # degree 13 with at least 8 squarings, which has the smallest truncation error.
    # Otherwise use the first kernel d with nA < RHO_V[d], or degree 13 with as many
    # squarings as the norm needs.
    X = if precision(float(real(eltype(A)))) > precision(Float64)
        exp_pade13!(cache, A, max(8, pade13_squarings(nA)))
    else
        @nif(
            5,
            d -> nA < RHO_V[d],
            d -> exp_gen!(cache, A, Val(d)),
            d -> exp_pade13!(cache, A, pade13_squarings(nA)),
        )
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
