intlog2(x::T) where {T <: Integer} = T(8 * sizeof(T) - leading_zeros(x - one(T)))
# `ceil(Int, x)` rather than `ceil(UInt, x)` so the integer-conversion path only needs
# `trunc(::Type{Int}, ::AbstractFloat)`, which extended-precision types such as
# DoubleFloats.Double64 define, whereas `trunc(::Type{<:Unsigned}, ::AbstractFloat)` is
# unavailable on Julia 1.10. `x` here is a positive norm, so the signed conversion matches.
intlog2(x) = x > typemax(Int) ? ceil(Int, log2(x)) : intlog2(ceil(Int, x))

function naivemul!(
        C::StridedMatrix{T}, A::StridedMatrix{T},
        B::StridedMatrix{T}
    ) where {T <: BlasFloat}
    return mul!(C, A, B)
end
function naivemul!(
        C::AbstractSparseMatrix, A::AbstractSparseMatrix,
        B::AbstractSparseMatrix
    )
    return mul!(C, A, B)
end
function naivemul!(C, A, B)
    Maxis, Naxis = axes(C)
    # TODO: heuristically pick `Nunroll` and `Munroll` using `sizeof(T)`, and maybe based on size of register file as well.
    # Nunroll = 2
    # Munroll = 4
    # These values select the generic multiplication path's block shape.
    nstep = step(Naxis)
    mstep = step(Maxis)
    # I don't want to deal with axes having non-unit step
    return if nstep == mstep == 1
        if sizeof(eltype(C)) > 256
            naivemul!(C, A, B, Maxis, Naxis, Val(1), Val(1))
        elseif sizeof(eltype(C)) > 128
            naivemul!(C, A, B, Maxis, Naxis, Val(2), Val(1))
        elseif sizeof(eltype(C)) > 96
            naivemul!(C, A, B, Maxis, Naxis, Val(4), Val(1))
        elseif sizeof(eltype(C)) > 64
            naivemul!(C, A, B, Maxis, Naxis, Val(4), Val(2))
        else
            naivemul!(C, A, B, Maxis, Naxis, Val(4), Val(3))
        end
    else
        mul!(C, A, B)
    end
end
function naivemul!(C::AbstractMatrix{T}, A, B, Maxis, Naxis, ::Val, ::Val) where {T}
    @inbounds for n in Naxis
        for m in Maxis
            value = zero(T)
            for k in axes(B, 1)
                value = muladd(A[m, k], B[k, n], value)
            end
            C[m, n] = value
        end
    end
    return C
end

"""
    struct ExpMethodGeneric{T}
    ExpMethodGeneric()=ExpMethodGeneric{Val{13}}()
    ExpMethodGeneric(k::Integer)=ExpMethodGeneric{Val{k}}()
    ExpMethodGeneric(::Type{T}) where T = ExpMethodGeneric{Val{pade_order_for_type(T)}}()

Generic exponential implementation of the method `ExpMethodHigham2005`,
for any exp argument `x`  for which the functions
`LinearAlgebra.opnorm`, `+`, `*`, `^`, and `/` (including addition with
UniformScaling objects) are defined. The type `T` is used to adjust the
number of terms used in the Pade approximants at compile time.

For high-precision types like `BigFloat`, the Padé order is automatically
selected based on the precision to achieve machine-precision accuracy.
You can also manually specify the order: `ExpMethodGeneric(k)` uses a
`(k,k)` Padé approximant. To automatically select based on element type,
use `ExpMethodGeneric(T)` where `T` is the element type.

See "The Scaling and Squaring Method for the Matrix Exponential Revisited"
by Higham, Nicholas J. in 2005 for algorithm details.

# Arguments

  - `k`: optional Padé order. `ExpMethodGeneric()` uses order `13` for common
    floating-point inputs; `ExpMethodGeneric(T)` selects an order from `T`'s
    precision.

# Fields

  - `T`: a `Val{k}` type parameter storing the selected Padé order.
"""
struct ExpMethodGeneric{T} end
ExpMethodGeneric() = ExpMethodGeneric{Val(13)}()
ExpMethodGeneric(k::Integer) = ExpMethodGeneric{Val{k}()}()

"""
    pade_order_for_type(::Type{T}) where {T}

Compute the minimum Padé order k required for machine-precision accuracy
for a given floating-point type T. The (k,k) Padé approximant for exp(x)
has error bounded by (x/2)^(2k+1) / (2k+1)! for |x| ≤ 1.
"""
function pade_order_for_type(::Type{T}) where {T}
    # Get precision in bits
    p = _precision_bits(T)
    # For standard Float64, use the optimized k=13
    p <= 64 && return 13
    # For higher precision, compute required k
    # We need: (1/2)^(2k+1) / (2k+1)! < 2^(-p)
    # Adding a small buffer for safety
    target = big(1) // big(2)^(p + 10)
    for k in 13:500
        bound = (big(1) // 2)^(2k + 1) / factorial(big(2k + 1))
        if bound < target
            return k
        end
    end
    return 500  # fallback for extremely high precision
end

_precision_bits(::Type{Float16}) = 11
_precision_bits(::Type{Float32}) = 24
_precision_bits(::Type{Float64}) = 53
_precision_bits(::Type{BigFloat}) = precision(BigFloat)
_precision_bits(::Type{Complex{T}}) where {T} = _precision_bits(T)
_precision_bits(::Type{T}) where {T <: AbstractFloat} = precision(T)
# Fallback for other numeric types (integers, etc.)
_precision_bits(::Type{T}) where {T <: Number} = 53

ExpMethodGeneric(::Type{T}) where {T} = ExpMethodGeneric{Val{pade_order_for_type(T)}()}()

# Extract the element type from various input types
_eltype(x::Number) = typeof(x)
_eltype(x::AbstractArray{T}) where {T} = T

# Determine if the default k=13 is sufficient for the given element type
_needs_higher_order(::Type{T}) where {T} = _precision_bits(T) > 64

function exponential!(
        x, method::ExpMethodGeneric{Vk},
        cache = alloc_mem(x, method)
    ) where {Vk}
    # For high-precision types with default k=13, automatically use higher order
    T = _eltype(x)
    if Vk === Val{13}() && _needs_higher_order(T)
        k = pade_order_for_type(T)
        # Only recurse if we actually need a higher order than 13
        # Otherwise we'd get infinite recursion (fixes #206)
        if k > 13
            return exponential!(x, ExpMethodGeneric{Val{k}()}(), cache)
        end
    end

    nx = opnorm(x, 1)
    if isnan(nx) || nx > 4611686018427387904 # i.e. 2^62 since it would cause overflow in 2^s
        # This should (hopefully) ensure that the result is Inf or NaN depending on
        # which values are produced by a power series. I.e. the result shouldn't
        # be all Infs when there are both Infs and zero since Inf*0===NaN
        return x * sum(x * exp(nx))
    end
    # `nx > 1` (not `iszero(nx)`) so zero-valued Duals with nonzero partials
    # take the s = 0 branch: ForwardDiff >= 1 defines `iszero(::Dual)` to also
    # require zero partials, which sent Dual(0, 1) into `intlog2` where the
    # partials were dropped (and 2^intlog2(0) = 2^64 overflows to 0). For real
    # nx in (0, 1], intlog2(nx) was already 0, so this changes nothing else.
    s = nx > 1 ? intlog2(nx) : 0
    (Vk === Val{13}() && x isa AbstractMatrix && ismutable(x)) &&
        return exp_generic_mutable(x, s, Val{13}())
    if s >= 1
        return exponential!(x / (2^s), method, cache)^(2^s)
    else
        return exp_pade_p(x, Vk, Vk) / exp_pade_q(x, Vk, Vk)
    end
end

# Specialized (13,13) Padé numerator for the immutable-matrix path. The coefficient
# type is taken from `eltype(x)` at runtime (rather than hardcoded `Float64`) so that a
# Float32 input yields a Float32 result instead of being silently promoted to Float64.
# This is deliberately not `@generated`: deriving the type inside a generated body would
# require constructing the element type (e.g. a ForwardDiff `Dual`), which is world-age
# blocked, so ForwardDiff differentiation would fail.
function exp_pade_p(x, ::Val{13}, ::Val{13})
    T = float(eltype(x))
    return @evalpoly(
        x,
        UniformScaling(T(1 // 1)),
        UniformScaling(T(1 // 2)),
        UniformScaling(T(3 // 25)),
        UniformScaling(T(11 // 600)),
        UniformScaling(T(11 // 5520)),
        UniformScaling(T(3 // 18400)),
        UniformScaling(T(1 // 96600)),
        UniformScaling(T(1 // 1932000)),
        UniformScaling(T(1 // 48944000)),
        UniformScaling(T(1 // 1585785600)),
        UniformScaling(T(1 // 67395888000)),
        UniformScaling(T(1 // 3953892096000)),
        UniformScaling(T(1 // 355850288640000)),
        UniformScaling(T(1 // 64764752532480000))
    )
end

function exp_pade_p(x::Number, ::Val{13}, ::Val{13})
    T = float(typeof(x))
    return @evalpoly(
        x,
        T(1 // 1), T(1 // 2), T(3 // 25), T(11 // 600), T(11 // 5520),
        T(3 // 18400), T(1 // 96600), T(1 // 1932000), T(1 // 48944000),
        T(1 // 1585785600), T(1 // 67395888000), T(1 // 3953892096000),
        T(1 // 355850288640000), T(1 // 64764752532480000)
    )
end

function exp_generic_mutable(x::AbstractMatrix{T}, s, ::Val{13}) where {T}
    y1 = similar(x, promote_type(T, Float64))
    y2 = similar(y1)
    y3 = similar(y1)
    return exp_generic!(y1, y2, y3, x, s, Val{13}())
end
function exp_generic!(y1, y2, y3, x, s, ::Val{13})
    if s > 0
        _y3 = exp_generic_core!(y1, y2, y3, x .* (1 / (1 << s)), Val{13}())
        if typeof(_y3) === (y3)
            y3 = _y3
        else
            y3 .= _y3
        end
        for _ in 1:s
            naivemul!(y1, y3, y3)
            y3, y1 = y1, y3
        end
        return y3
    else
        return exp_generic_core!(y1, y2, y3, x, Val{13}())
    end
end
# `lu!` is only defined for `StridedMatrix`, and `lu(::StaticArray)` (note `MArray<:StaticArray`) returns `::StaticArrays.LU !== LinearAlgebra.LU`
_rdiv!(A, B::StridedMatrix) = rdiv!(A, lu!(B))
_rdiv!(A, B::SparseMatrixCSC) = A / Array(B)
_rdiv!(A, B) = A / B

function exp_generic_core!(y1, y2, y3, x, ::Val{13})
    @inbounds for i in eachindex(y3)
        y3[i] = -x[i]
    end
    exp_pade_p!(y1, y2, y3, Val{13}(), Val{13}())
    exp_pade_p!(y3, y2, x, Val{13}(), Val{13}())
    return _rdiv!(y3, y1)
end

function exp_pade_p!(
        y1::AbstractMatrix{T}, y2::AbstractMatrix{T}, x::AbstractMatrix,
        ::Val{13}, ::Val{13}
    ) where {T}
    N = size(x, 1) # check square is in `exp_generic`
    y1 .= x .* 1.5440497506703088e-17
    for c in (
            2.8101705462199623e-15, 2.529153491597966e-13, 1.48377004840414e-11,
            6.306022705717595e-10,
            2.0431513566525008e-8, 5.175983436853002e-7, 1.0351966873706003e-5,
            0.00016304347826086958,
            0.0019927536231884057, 0.018333333333333333, 0.12, 0.5,
        )
        @inbounds for n in 1:N
            y1[n, n] += c
        end
        naivemul!(y2, y1, x)
        y1, y2 = y2, y1
    end
    @inbounds for n in 1:N
        y1[n, n] += one(T)
    end
    return y1
end

@generated function exp_pade_p(x, ::Val{k}, ::Val{m}) where {k, m}
    factorial = Base.factorial ∘ big
    p = map(Tuple(0:k)) do j
        num = factorial(k + m - j) * factorial(k)
        den = factorial(k + m) * factorial(k - j) * factorial(j)
        (float ∘ eltype)(x)(num // den) * (x <: Number ? 1 : I)
    end
    return :(@evalpoly(x, $(p...)))
end

exp_pade_q(x, k, m) = exp_pade_p(-x, m, k)
