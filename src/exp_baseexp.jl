"""
    ExpMethodHigham2005Base()

The same as `ExpMethodHigham2005` but follows `Base.exp` closer.

"""
struct ExpMethodHigham2005Base end
function alloc_mem(A::StridedMatrix{T},
                   method::ExpMethodHigham2005Base) where {T <: LinearAlgebra.BlasFloat}
    n = LinearAlgebra.checksquare(A)
    A2 = Matrix{T}(undef, n, n)
    P = Matrix{T}(undef, n, n)
    U = Matrix{T}(undef, n, n)
    V = Matrix{T}(undef, n, n)
    temp = Matrix{T}(undef, n, n)
    return (A2, P, U, V, temp)
end

## Destructive matrix exponential using algorithm from Higham, 2008,
## "Functions of Matrices: Theory and Computation", SIAM
##
## Non-allocating version of `LinearAlgebra.exp!`. Modifies `A` to
## become (approximately) `exp(A)`.
function exponential!(A::StridedMatrix{T}, method::ExpMethodHigham2005Base,
                      cache = alloc_mem(A, method)) where {T <: LinearAlgebra.BlasFloat}
    X = A
    n = LinearAlgebra.checksquare(A)
    # if ishermitian(A)
    # return copytri!(parent(exp(Hermitian(A))), 'U', true)
    # end

    A2, P, U, V, temp = cache

    fill!(P, zero(T))
    fill!(@diagview(P), one(T)) # P = Inn

    if A isa StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}
        ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    else
        A, bal = GenericSchur.balance!(A)
        ilo, ihi, scale = bal.ilo, bal.ihi, bal.D
    end

    nA = opnorm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            C = T[17643225600.0, 8821612800.0, 2075673600.0, 302702400.0,
                  30270240.0, 2162160.0, 110880.0, 3960.0,
                  90.0, 1.0]
        elseif nA > 0.25
            C = T[17297280.0, 8648640.0, 1995840.0, 277200.0,
                  25200.0, 1512.0, 56.0, 1.0]
        elseif nA > 0.015
            C = T[30240.0, 15120.0, 3360.0,
                  420.0, 30.0, 1.0]
        else
            C = T[120.0, 60.0, 12.0, 1.0]
        end
        mul!(A2, A, A)
        @. U = C[2] * P
        @. V = C[1] * P
        for k in 1:(div(size(C, 1), 2) - 1)
            k2 = 2 * k
            mul!(temp, P, A2)
            P, temp = temp, P # equivalent to P *= A2
            @. U += C[k2 + 2] * P
            @. V += C[k2 + 1] * P
        end
        mul!(temp, A, U)
        U, temp = temp, U # equivalent to U = A * U
        @. X = V + U
        @. temp = V - U
        LAPACK.gesv!(temp, X)
    else
        s = log2(nA / 5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int, s)
            A ./= convert(T, 2^si)
        end
        C = T[64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
              1187353796428800.0, 129060195264000.0, 10559470521600.0,
              670442572800.0, 33522128640.0, 1323241920.0,
              40840800.0, 960960.0, 16380.0,
              182.0, 1.0]
        mul!(A2, A, A)
        @. U = C[2] * P
        @. V = C[1] * P
        for k in 1:6
            k2 = 2 * k
            mul!(temp, P, A2)
            P, temp = temp, P # equivalent to P *= A2
            @. U += C[k2 + 2] * P
            @. V += C[k2 + 1] * P
        end
        mul!(temp, A, U)
        U, temp = temp, U # equivalent to U = A * U
        @. X = V + U
        @. temp = V - U
        LAPACK.gesv!(temp, X)

        if s > 0            # squaring to reverse dividing by power of 2
            for t in 1:si
                mul!(temp, X, X)
                X .= temp
            end
        end
    end

    # Undo the balancing
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
                LinearAlgebra.rcswap!(j, bal.prow[j], X)
            end
        end
        if ihi < n       # apply upper permutations in forward order
            for j in (ihi + 1):n
                LinearAlgebra.rcswap!(j, bal.pcol[j - ihi], X)
            end
        end
    end

    return X
end
