using LinearSolve: LinearSolve
using SciMLBase: LinearProblem, successful_retcode

# Allocation-free translation of the balancing transform used by LAPACK GEBAL.
# The caller-owned `scale` array stores both permutations and diagonal scales.
function gebal_noalloc!(A::StridedMatrix{T}, scale::AbstractVector{R}) where {T, R <: Real}
    n = checksquare(A)
    length(scale) == n || throw(DimensionMismatch("scale must have length $n"))
    stride(A, 1) == 1 || throw(ArgumentError("matrix does not have unit stride in its first dimension"))
    for a in A
        isfinite(a) || throw(ArgumentError("matrix contains Infs or NaNs"))
    end

    fill!(scale, one(R))
    ilo = 1
    ihi = n

    # Isolate rows with no off-diagonal entries at the lower-right corner.
    ihi = n + 1
    @inbounds while ihi > 1
        ihi -= 1
        exchange = false
        source = 0
        for j in ihi:-1:1
            exchange = true
            source = j
            for i in 1:ihi
                if i != j && !iszero(A[j, i])
                    exchange = false
                    break
                end
            end
            exchange && break
        end
        exchange || break
        scale[ihi] = source
        if source != ihi
            for i in 1:ihi
                A[i, source], A[i, ihi] = A[i, ihi], A[i, source]
            end
            for j in ilo:n
                A[source, j], A[ihi, j] = A[ihi, j], A[source, j]
            end
        end
    end

    # Isolate columns with no off-diagonal entries at the upper-left corner.
    if ihi > 1
        ilo = 0
        @inbounds while ilo < n
            ilo += 1
            exchange = false
            source = 0
            for j in ilo:ihi
                exchange = true
                source = j
                for i in ilo:ihi
                    if i != j && !iszero(A[i, j])
                        exchange = false
                        break
                    end
                end
                exchange && break
            end
            exchange || break
            scale[ilo] = source
            if source != ilo
                for i in 1:ihi
                    A[i, source], A[i, ilo] = A[i, ilo], A[i, source]
                end
                for j in ilo:n
                    A[source, j], A[ilo, j] = A[ilo, j], A[source, j]
                end
            end
        end
    end

    radix = R(2)
    factor = R(0.95)
    sfmin1 = floatmin(R) / eps(R)
    sfmin2 = sfmin1 * radix
    sfmax2 = inv(sfmin2)
    converged = false
    while !converged
        converged = true
        @inbounds for i in ilo:ihi
            column_norm = zero(R)
            column_max = zero(R)
            row_norm = zero(R)
            row_max = zero(R)
            for j in ilo:ihi
                column_entry = abs(A[j, i])
                row_entry = abs(A[i, j])
                column_norm = hypot(column_norm, column_entry)
                row_norm = hypot(row_norm, row_entry)
                column_max = max(column_max, column_entry)
                row_max = max(row_max, row_entry)
            end
            (iszero(column_norm) || iszero(row_norm)) && continue

            g = row_norm / radix
            original_sum = column_norm + row_norm
            f = one(R)
            while column_norm < row_norm / radix
                if column_norm >= g || max(f, column_norm, column_max) >= sfmax2 ||
                        min(row_norm, g, row_max) <= sfmin2
                    break
                end
                f *= radix
                column_norm *= radix
                column_max *= radix
                row_norm /= radix
                g /= radix
                row_max /= radix
            end
            g = column_norm / radix
            while row_norm <= column_norm / radix
                if g < row_norm || max(row_norm, row_max) >= sfmax2 ||
                        min(f, column_norm, g, column_max) <= sfmin2
                    break
                end
                f /= radix
                column_norm /= radix
                g /= radix
                column_max /= radix
                row_norm *= radix
                row_max *= radix
            end
            column_norm + row_norm >= factor * original_sum && continue

            converged = false
            scale[i] *= f
            invf = inv(f)
            for j in ilo:n
                A[i, j] *= invf
            end
            for j in 1:ihi
                A[j, i] *= f
            end
        end
    end
    return ilo, ihi, scale
end

"""
    ExpMethodHigham2005Base()

Matrix-exponential method using the Base-compatible Higham scaling-and-squaring
implementation.

This is primarily the default reduced-matrix method used by the Krylov APIs.
It has no fields or constructor arguments.

# Returns

An `ExpMethodHigham2005Base` algorithm object for use with
[`exponential!`](@ref).

# Examples

```julia
A = [0.0 1.0; -1.0 0.0]
exponential!(copy(A), ExpMethodHigham2005Base())
```
"""
struct ExpMethodHigham2005Base end
function alloc_mem(
        A::StridedMatrix{T},
        method::ExpMethodHigham2005Base
    ) where {T <: BlasFloat}
    n = checksquare(A)
    A2 = Matrix{T}(undef, n, n)
    P = Matrix{T}(undef, n, n)
    U = Matrix{T}(undef, n, n)
    V = Matrix{T}(undef, n, n)
    temp = Matrix{T}(undef, n, n)
    # Cached LinearSolve workspace for the Padé denominator solves, using the
    # default algorithm choice: it size-selects the fastest available LU
    # (RecursiveFactorization when loaded — markedly faster than the generic
    # small/medium n). The buffers are workspace-owned, so aliasing lets the
    # factorization overwrite Abuf in place on every `solve!` (no n×n copy),
    # keeping each call allocation-free.
    Abuf = Matrix{T}(undef, n, n)
    Bbuf = Matrix{T}(undef, n, n)
    linsolve = LinearSolve.init(
        LinearProblem(Abuf, Bbuf);
        alias = LinearSolve.LinearAliasSpecifier(alias_A = true, alias_b = true)
    )
    scale = Vector{real(T)}(undef, n)  # balancing scale, filled by gebal_noalloc!
    return (A2, P, U, V, temp, linsolve, scale)
end

# X .= temp \ X through the cached LinearSolve workspace: refill the
# workspace-owned buffers, mark A as replaced, and solve in place.
function _pade_linsolve!(
        X::StridedMatrix{T}, temp::StridedMatrix{T}, linsolve
    ) where {T}
    Abuf = linsolve.A
    copyto!(Abuf, temp)
    linsolve.A = Abuf # flag refactorization; alias_A lets it overwrite Abuf
    copyto!(linsolve.b, X)
    sol = LinearSolve.solve!(linsolve)
    # exp! historically threw on a singular denominator;
    # keep that contract rather than propagating a failed factorization.
    if !successful_retcode(sol.retcode)
        throw(SingularException(0))
    end
    copyto!(X, sol.u)
    return X
end

# Padé numerator/denominator coefficients for the Higham (2005) scaling-and-
# squaring orders used below. Stored as `const` tuples so no coefficient vector
# is allocated per `exponential!` call; the values are converted to the working
# element type inside `_pade_evaluate!`.
const _PADE_C3 = (120.0, 60.0, 12.0, 1.0)
const _PADE_C5 = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
const _PADE_C7 = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
const _PADE_C9 = (
    17643225600.0, 8821612800.0, 2075673600.0, 302702400.0,
    30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0,
)
const _PADE_C13 = (
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0,
    40840800.0, 960960.0, 16380.0, 182.0, 1.0,
)

# Evaluate the Padé numerator `U` and denominator `V` for the coefficient set
# `C`, then solve `(V - U) X = (V + U)`. `C` is passed as a concrete `NTuple`
# so this specializes on its length (the caller dispatches each norm range to
# the matching tuple), keeping the loop type-stable and allocation-free. The
# `P`/`temp`/`U` swaps are local buffer rebindings; only `X` carries the result.
@inline function _pade_evaluate!(
        X, U, V, P, A2, temp, A, C::NTuple{N, Float64}, linsolve, ::Type{T}
    ) where {N, T}
    mul!(A2, A, A)
    @. U = convert(T, C[2]) * P
    @. V = convert(T, C[1]) * P
    @inbounds for k in 1:(N ÷ 2 - 1)
        k2 = 2 * k
        mul!(temp, P, A2)
        P, temp = temp, P # equivalent to P *= A2
        cu = convert(T, C[k2 + 2])
        cv = convert(T, C[k2 + 1])
        @. U += cu * P
        @. V += cv * P
    end
    mul!(temp, A, U)
    U, temp = temp, U # equivalent to U = A * U
    @. X = V + U
    @. temp = V - U
    _pade_linsolve!(X, temp, linsolve)
    return X
end

## Destructive matrix exponential using algorithm from Higham, 2008,
## "Functions of Matrices: Theory and Computation", SIAM
##
## Non-allocating version of `LinearAlgebra.exp!`. Modifies `A` to
## become (approximately) `exp(A)`.
function exponential!(
        A::StridedMatrix{T}, method::ExpMethodHigham2005Base,
        cache = alloc_mem(A, method)
    ) where {T <: BlasFloat}
    X = A
    n = checksquare(A)
    # if ishermitian(A)
    # return copytri!(parent(exp(Hermitian(A))), 'U', true)
    # end

    A2, P, U, V, temp, linsolve, scale = cache

    fill!(P, zero(T))
    fill!(@_diagview(P), one(T)) # P = Inn

    # `A` is a strided BlasFloat matrix here (method signature), so LAPACK
    # balancing applies; the non-allocating wrapper writes into the cache's
    # `scale` buffer instead of allocating one (as GenericSchur.balance! would).
    ilo, ihi, scale = gebal_noalloc!(A, scale)    # modifies A and scale

    nA = opnorm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            _pade_evaluate!(X, U, V, P, A2, temp, A, _PADE_C9, linsolve, T)
        elseif nA > 0.25
            _pade_evaluate!(X, U, V, P, A2, temp, A, _PADE_C7, linsolve, T)
        elseif nA > 0.015
            _pade_evaluate!(X, U, V, P, A2, temp, A, _PADE_C5, linsolve, T)
        else
            _pade_evaluate!(X, U, V, P, A2, temp, A, _PADE_C3, linsolve, T)
        end
    else
        s = log2(nA / 5.4)               # power of 2 later reversed by squaring
        si = 0                           # always defined so the s > 0 squaring loop is type-stable
        if s > 0
            si = ceil(Int, s)
            A ./= convert(T, 2^si)
        end
        _pade_evaluate!(X, U, V, P, A2, temp, A, _PADE_C13, linsolve, T)

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

    # LAPACK gebal encodes the row/column permutations in the `scale` array
    # itself (unlike GenericSchur, which returns separate prow/pcol).
    if ilo > 1       # apply lower permutations in reverse order
        for j in (ilo - 1):-1:1
            rcswap!(j, Int(scale[j]), X)
        end
    end
    if ihi < n       # apply upper permutations in forward order
        for j in (ihi + 1):n
            rcswap!(j, Int(scale[j]), X)
        end
    end

    return X
end
