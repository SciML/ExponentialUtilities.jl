## Destructive matrix exponential using algorithm from Higham, 2008,
## "Functions of Matrices: Theory and Computation", SIAM
##
## Non-allocating version of `LinearAlgebra.exp!`. Modifies `A` to
## become (approximately) `exp(A)`.
function _exp!(A::StridedMatrix{T}; caches=nothing) where T <: LinearAlgebra.BlasFloat
    X = A
    n = LinearAlgebra.checksquare(A)
    # if ishermitian(A)
        # return copytri!(parent(exp(Hermitian(A))), 'U', true)
    # end

    if caches == nothing
        A2   = Matrix{T}(undef, n, n)
        P    = Matrix{T}(undef, n, n)
        U    = Matrix{T}(undef, n, n)
        V    = Matrix{T}(undef, n, n)
        temp = Matrix{T}(undef, n, n)
    else
        A2, P, U, V, temp = caches
    end
    fill!(P, zero(T)); fill!(@diagview(P), one(T)) # P = Inn

    ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    nA = opnorm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            C = T[17643225600.,8821612800.,2075673600.,302702400.,
                     30270240.,   2162160.,    110880.,     3960.,
                           90.,         1.]
        elseif nA > 0.25
            C = T[17297280.,8648640.,1995840.,277200.,
                     25200.,   1512.,     56.,     1.]
        elseif nA > 0.015
            C = T[30240.,15120.,3360.,
                    420.,   30.,   1.]
        else
            C = T[120.,60.,12.,1.]
        end
        mul!(A2, A, A)
        @. U = C[2] * P
        @. V = C[1] * P
        for k in 1:(div(size(C, 1), 2) - 1)
            k2 = 2 * k
            mul!(temp, P, A2); P, temp = temp, P # equivalent to P *= A2
            @. U += C[k2 + 2] * P
            @. V += C[k2 + 1] * P
        end
        mul!(temp, A, U); U, temp = temp, U # equivalent to U = A * U
        @. X = V + U
        @. temp = V - U
        LAPACK.gesv!(temp, X)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A ./= convert(T,2^si)
        end
        C  = T[64764752532480000.,32382376266240000.,7771770303897600.,
                1187353796428800.,  129060195264000.,  10559470521600.,
                    670442572800.,      33522128640.,      1323241920.,
                        40840800.,           960960.,           16380.,
                             182.,                1.]
        mul!(A2, A, A)
        @. U = C[2] * P
        @. V = C[1] * P
        for k in 1:6
            k2 = 2 * k
            mul!(temp, P, A2); P, temp = temp, P # equivalent to P *= A2
            @. U += C[k2 + 2] * P
            @. V += C[k2 + 1] * P
        end
        mul!(temp, A, U); U, temp = temp, U # equivalent to U = A * U
        @. X = V + U
        @. temp = V - U
        LAPACK.gesv!(temp, X)

        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                mul!(temp, X, X)
                X .= temp
            end
        end
    end

    # Undo the balancing
    for j = ilo:ihi
        scj = scale[j]
        for i = 1:n
            X[j,i] *= scj
        end
        for i = 1:n
            X[i,j] /= scj
        end
    end

    if ilo > 1       # apply lower permutations in reverse order
        for j in (ilo-1):-1:1; LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
    end
    if ihi < n       # apply upper permutations in forward order
        for j in (ihi+1):n;    LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
    end
    return X
end

intlog2(x::T) where {T<:Integer} =  T(8*sizeof(T) - leading_zeros(x-one(T)))
intlog2(x) = intlog2(ceil(Int,x))

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
        naivemul!(C, A, B, Maxis, Naxis)
    else
        mul!(C,A,B)
    end
end
# Separated to make it easier to test.
function naivemul!(C::AbstractMatrix{T}, A, B, Maxis, Naxis) where {T}
    N = last(Naxis)
    M = last(Maxis)
    Kaxis = axes(B,1)
    Base.Experimental.@aliasscope begin
        n = first(Naxis)-1
        @inbounds begin
            while n < N - 1
                m = first(Maxis)-1
                while m < M - 3
                    Base.Cartesian.@nexprs 2 j -> Base.Cartesian.@nexprs 4 i -> Cmn_i_j = zero(T)
                    for k ∈ Kaxis
                        Base.Cartesian.@nexprs 2 j -> Base.Cartesian.@nexprs 4 i -> Cmn_i_j = muladd(Base.Experimental.Const(A)[m+i,k],Base.Experimental.Const(B)[k,n+j],Cmn_i_j)
                    end
                    Base.Cartesian.@nexprs 2 j -> Base.Cartesian.@nexprs 4 i -> C[m+i,n+j] = Cmn_i_j
                    m += 4
                end
                for mm ∈ 1+m:M
                    Base.Cartesian.@nexprs 2 j -> Cmn_j = zero(T)
                    for k ∈ Kaxis
                        Base.Cartesian.@nexprs 2 j -> Cmn_j = muladd(Base.Experimental.Const(A)[mm,k],Base.Experimental.Const(B)[k,n+j],Cmn_j)
                    end
                    Base.Cartesian.@nexprs 2 j -> C[mm,n+j] = Cmn_j
                end
                n += 2
            end
            m = first(Maxis)-1
            while m < M - 3
                Base.Cartesian.@nexprs 4 i -> Cmn_i = zero(T)
                for k ∈ Kaxis
                    Base.Cartesian.@nexprs 4 i -> Cmn_i = muladd(Base.Experimental.Const(A)[m+i,k],Base.Experimental.Const(B)[k,N],Cmn_i)
                end
                Base.Cartesian.@nexprs 4 i -> C[m+i,N] = Cmn_i
                m += 4
            end
            for mm ∈ 1+m:M
                Cmn = zero(T)
                for k ∈ Kaxis
                    Cmn = muladd(Base.Experimental.Const(A)[mm,k], Base.Experimental.Const(B)[k,N], Cmn)
                end
                C[mm,N] = Cmn
            end
        end
    end
    C
end

"""
    exp(x, vk=Val{13}())
Generic exponential function, working on any `x` for which the functions
`LinearAlgebra.opnorm`, `+`, `*`, `^`, and `/` (including addition with
UniformScaling objects) are defined. Use the argument `vk` to adjust the
number of terms used in the Pade approximants at compile time.

See "The Scaling and Squaring Method for the Matrix Exponential Revisited"
by Higham, Nicholas J. in 2005 for algorithm details.
"""
function exp_generic(x, vk=Val{13}())
    nx = opnorm(x, 1)
    if !isfinite(nx)
        # This should (hopefully) ensure that the result is Inf or NaN depending on
        # which values are produced by a power series. I.e. the result shouldn't
        # be all Infs when there are both Infs and zero since Inf*0===NaN
        return x*sum(x*nx)
    end
    s = iszero(nx) ? 0 : intlog2(nx)
    (vk === Val{13}() && x isa AbstractMatrix && ismutable(x)) && return exp_generic_mutable(x, s, Val{13}())
    if s >= 1
        exp_generic(x/(2^s), vk)^(2^s)
    else
        exp_pade_p(x, vk, vk) / exp_pade_q(x, vk, vk)
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
        exp_generic_core!(y1, y2, y3, x .* (1/(1 << s)), Val{13}())
        for _ ∈ 1:s
            naivemul!(y1, y3, y3)
            y3, y1 = y1, y3
        end
        return y3
    else
        return exp_generic_core!(y1, y2, y3, x, Val{13}())
    end
end
function exp_generic_core!(y1, y2, y3, x, ::Val{13})
    @inbounds for i ∈ eachindex(y3)
        y3[i] = -x[i]
    end
    exp_pade_p!(y1, y2, y3, Val{13}(), Val{13}())
    exp_pade_p!(y3, y2, x, Val{13}(), Val{13}())
    rdiv!(y3, lu!(y1))
    return y3
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
