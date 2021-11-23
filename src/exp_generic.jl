
intlog2(x::T) where {T<:Integer} =  T(8*sizeof(T) - leading_zeros(x-one(T)))
intlog2(x) = x > typemax(UInt) ? ceil(Int, log2(x)) : intlog2(ceil(UInt,x)) % Int

naivemul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}) where {T<:LinearAlgebra.BlasFloat} = mul!(C,A,B)
function naivemul!(C, A, B)
    Maxis, Naxis = axes(C)
    # TODO: heuristically pick `Nunroll` and `Munroll` using `sizeof(T)`, and maybe based on size of register file as well.
    # Nunroll = 2
    # Munroll = 4
    # For now, the `4` and `2` are hardcoded to use `Base.Cartesian.@nexprs` without generated functions.
    # But a minor code reorgaanization, e.g. loading and storing tuples while moving the `Base.Cartesian.@nexprs`
    # into `tload` and `tstore!` functions could let us switch APIs.
    nstep = step(Naxis)
    mstep = step(Maxis)
    # I don't want to deal with axes having non-unit step
    if nstep == mstep == 1
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
        mul!(C,A,B)
    end
end
_const(A) = A
_const(A::Array) = Base.Experimental.Const(A)
# Separated to make it easier to test.
@generated function naivemul!(C::AbstractMatrix{T}, A, B, Maxis, Naxis, ::Val{MU}, ::Val{NU}) where {T,MU,NU}
    nrem_body = quote
        m = first(Maxis)-1
        while m < M - $(MU-1)
            Base.Cartesian.@nexprs $MU i -> Cmn_i = zero(T)
            for k ∈ Kaxis
                Base.Cartesian.@nexprs $MU i -> Cmn_i = muladd(_const(A)[m+i,k],_const(B)[k,nn],Cmn_i)
            end
            Base.Cartesian.@nexprs $MU i -> C[m+i,nn] = Cmn_i
            m += $MU
        end
        for mm ∈ 1+m:M
            Cmn = zero(T)
            for k ∈ Kaxis
                Cmn = muladd(_const(A)[mm,k], _const(B)[k,nn], Cmn)
            end
            C[mm,nn] = Cmn
        end
    end
    nrem_quote = if NU > 2
        :(for nn ∈ 1+n:N; $nrem_body; end)
    else
        :(let nn = N; $nrem_body; end)
    end
    quote
        N = last(Naxis)
        M = last(Maxis)
        Kaxis = axes(B,1)
        Base.Experimental.@aliasscope begin
            n = first(Naxis)-1
            @inbounds begin
                while n < N - $(NU-1)
                    m = first(Maxis)-1
                    while m < M - $(MU-1)
                        Base.Cartesian.@nexprs $NU j -> Base.Cartesian.@nexprs $MU i -> Cmn_i_j = zero(T)
                        for k ∈ Kaxis
                            Base.Cartesian.@nexprs $MU i -> Ak_i = _const(A)[m+i,k]
                            Base.Cartesian.@nexprs $NU j -> begin
                                Bk_j = _const(B)[k,n+j]
                                Base.Cartesian.@nexprs $MU i -> Cmn_i_j = muladd(Ak_i, Bk_j, Cmn_i_j)
                            end
                        end
                        Base.Cartesian.@nexprs $NU j -> Base.Cartesian.@nexprs $MU i -> C[m+i,n+j] = Cmn_i_j
                        m += $MU
                    end
                    for mm ∈ 1+m:M
                        Base.Cartesian.@nexprs $NU j -> Cmn_j = zero(T)
                        for k ∈ Kaxis
                            Base.Cartesian.@nexprs $NU j -> Cmn_j = muladd(_const(A)[mm,k],_const(B)[k,n+j],Cmn_j)
                        end
                        Base.Cartesian.@nexprs $NU j -> C[mm,n+j] = Cmn_j
                    end
                    n += $NU
                end
                $(NU > 1 ? nrem_quote : nothing)
            end
        end
        C
    end
end

"""
    struct ExpMethodGeneric{T}
    ExpMethodGeneric()=ExpMethodGeneric{Val{13}}();

Generic exponential implementation of the method `ExpMethodHigham2005`,
for any exp argument `x`  for which the functions
`LinearAlgebra.opnorm`, `+`, `*`, `^`, and `/` (including addition with
UniformScaling objects) are defined. The type `T` is used to adjust the
number of terms used in the Pade approximants at compile time.


See "The Scaling and Squaring Method for the Matrix Exponential Revisited"
by Higham, Nicholas J. in 2005 for algorithm details.

"""
struct ExpMethodGeneric{T}
end
ExpMethodGeneric()=ExpMethodGeneric{Val(13)}();

function exponential!(x,method::ExpMethodGeneric{Vk},cache=alloc_mem(x,method)) where Vk
    nx = LinearAlgebra.opnorm1(x)
    if !isfinite(nx)
        # This should (hopefully) ensure that the result is Inf or NaN depending on
        # which values are produced by a power series. I.e. the result shouldn't
        # be all Infs when there are both Infs and zero since Inf*0===NaN
        return x*sum(x*nx)
    end
    s = iszero(nx) ? 0 : intlog2(nx)
    (Vk === Val{13}() && x isa AbstractMatrix && ismutable(x)) && return exp_generic_mutable(x, s, Val{13}())
    if s >= 1
        return exponential!(x/(2^s), method)^(2^s)
    else
        return exp_pade_p(x, Vk, Vk) / exp_pade_q(x, Vk, Vk)
    end
end

function exp_pade_p(x, ::Val{13}, ::Val{13})
    @evalpoly(x,LinearAlgebra.UniformScaling{Float64}(1.0),
                LinearAlgebra.UniformScaling{Float64}(0.5),
                LinearAlgebra.UniformScaling{Float64}(0.12),
                LinearAlgebra.UniformScaling{Float64}(0.018333333333333333),
                LinearAlgebra.UniformScaling{Float64}(0.0019927536231884057),
                LinearAlgebra.UniformScaling{Float64}(0.00016304347826086958),
                LinearAlgebra.UniformScaling{Float64}(1.0351966873706003e-5),
                LinearAlgebra.UniformScaling{Float64}(5.175983436853002e-7),
                LinearAlgebra.UniformScaling{Float64}(2.0431513566525008e-8),
                LinearAlgebra.UniformScaling{Float64}(6.306022705717595e-10),
                LinearAlgebra.UniformScaling{Float64}(1.48377004840414e-11),
                LinearAlgebra.UniformScaling{Float64}(2.529153491597966e-13),
                LinearAlgebra.UniformScaling{Float64}(2.8101705462199623e-15),
                LinearAlgebra.UniformScaling{Float64}(1.5440497506703088e-17))
end

function exp_generic_mutable(x::AbstractMatrix{T}, s, ::Val{13}) where {T}
    y1 = similar(x, promote_type(T,Float64))
    y2 = similar(y1)
    y3 = similar(y1)
    exp_generic!(y1, y2, y3, x, s, Val{13}())
end
function exp_generic!(y1, y2, y3, x, s, ::Val{13})
    if s > 0
        _y3 = exp_generic_core!(y1, y2, y3, x .* (1/(1 << s)), Val{13}())
        if typeof(_y3) === (y3)
            y3 = _y3
        else
            y3 .= _y3
        end
        for _ ∈ 1:s
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
_rdiv!(A, B) = A / B

function exp_generic_core!(y1, y2, y3, x, ::Val{13})
    @inbounds for i ∈ eachindex(y3)
        y3[i] = -x[i]
    end
    exp_pade_p!(y1, y2, y3, Val{13}(), Val{13}())
    exp_pade_p!(y3, y2, x, Val{13}(), Val{13}())
    return _rdiv!(y3, y1)
end

function exp_pade_p!(y1::AbstractMatrix{T}, y2::AbstractMatrix{T}, x::AbstractMatrix, ::Val{13}, ::Val{13}) where {T}
    N = size(x,1) # check square is in `exp_generic`
    y1 .= x .* 1.5440497506703088e-17
    for c ∈ (
        2.8101705462199623e-15, 2.529153491597966e-13, 1.48377004840414e-11, 6.306022705717595e-10,
        2.0431513566525008e-8, 5.175983436853002e-7, 1.0351966873706003e-5, 0.00016304347826086958,
        0.0019927536231884057, 0.018333333333333333, 0.12, 0.5)

        @inbounds for n ∈ 1:N
            y1[n,n] += c
        end
        naivemul!(y2, y1, x)
        y1, y2 = y2, y1
    end
    @inbounds for n ∈ 1:N
        y1[n,n] += one(T)
    end
    return y1
end

function exp_pade_p(x::Number, ::Val{13}, ::Val{13})
    @evalpoly(x,1.0,0.5,0.12,0.018333333333333333,0.0019927536231884057,
              0.00016304347826086958,1.0351966873706003e-5,5.175983436853002e-7,
              2.0431513566525008e-8,6.306022705717595e-10,1.48377004840414e-11,
              2.529153491597966e-13,2.8101705462199623e-15,1.5440497506703088e-17)
end

@generated function exp_pade_p(x, ::Val{k}, ::Val{m}) where {k, m}
    factorial = Base.factorial ∘ big
    p = map(Tuple(0:k)) do j
        num = factorial(k + m - j) * factorial(k)
        den = factorial(k + m) * factorial(k - j)*factorial(j)
        (float ∘ eltype)(x)(num // den) * (x <: Number ? 1 : I)
    end
    :(@evalpoly(x, $(p...)))
end

exp_pade_q(x, k, m) = exp_pade_p(-x, m, k)
