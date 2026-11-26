# Arnoldi/Lanczos iteration algorithms (with custom IOP)

#######################################
# Output type/cache
"""
    KrylovSubspace{T}(n, [maxiter = 30]) -> Ks
    KrylovSubspace{T, U, VType}(n, [maxiter = 30]) -> Ks

Construct an uninitialized Krylov subspace to be filled by [`arnoldi!`](@ref)
or [`lanczos!`](@ref).

# Arguments

  - `T`: element type of the Krylov basis.
  - `U`: element type of the Hessenberg matrix. Use `real(T)` for a Hermitian
    problem.
  - `VType`: matrix storage type for the basis. It must support
    `VType(undef, rows, columns)`; choose a GPU-backed matrix type for GPU
    vectors.
  - `n`: dimension of the operator domain.
  - `maxiter`: maximum Krylov dimension, defaulting to `30`.
  - `augmented`: number of rows reserved for an augmented operator, defaulting
    to `0`.

# Fields

  - `m`: current Krylov dimension. It can be smaller than `maxiter` after a
    happy breakdown.
  - `maxiter`: allocated maximum Krylov dimension.
  - `augmented`: number of augmented rows.
  - `beta`: norm of the starting vector from the last factorization.
  - `wasbreakdown`: whether the last factorization ended in a happy breakdown.
  - `V`: orthonormal Krylov basis storage.
  - `H`: Hessenberg or tridiagonal recurrence-coefficient storage.

# Returns

An uninitialized `KrylovSubspace`. Call `arnoldi!` or `lanczos!` before using
its basis or coefficients.

# Examples

```julia
A = [-2.0 1.0; 1.0 -2.0]
b = [1.0, 0.0]
Ks = KrylovSubspace{Float64}(length(b), 2)
arnoldi!(Ks, A, b)
```
"""
mutable struct KrylovSubspace{
        T, U, B, VType <: AbstractMatrix{T},
        HType <: AbstractMatrix{U},
    }
    m::Int             # subspace dimension
    maxiter::Int       # maximum allowed subspace size
    augmented::Int     # length of the augmented part
    beta::B            # norm(b,2)
    wasbreakdown::Bool # Did a happy breakdown occur?
    V::VType  # orthonormal bases
    H::HType  # Gram-Schmidt coefficients (real for Hermitian matrices)
end

function KrylovSubspace{T, U, VType}(
        n::Integer, maxiter::Integer = 30,
        augmented::Integer = false
    ) where {T, U, VType <: AbstractMatrix{T}}
    V = VType(undef, n + augmented, maxiter + 1)
    H = fill(zero(U), maxiter + 1, maxiter + !iszero(augmented))
    return KrylovSubspace{T, U, real(T), VType, Matrix{U}}(
        maxiter, maxiter, augmented,
        zero(real(T)), false, V, H
    )
end

KrylovSubspace{T}(args...) where {T} = KrylovSubspace{T, T}(args...)
KrylovSubspace{T, U}(args...) where {T, U} = KrylovSubspace{T, U, Matrix{T}}(args...)

# Build a subspace whose basis matches the storage of `prototype`, so a device-backed
# input keeps its array type instead of silently falling back to a host `Matrix`.
function _krylov_subspace(
        prototype::AbstractArray, ::Type{T}, ::Type{U},
        n::Integer, maxiter::Integer, augmented::Integer = false
    ) where {T, U}
    V = similar(prototype, T, (n + augmented, maxiter + 1))
    H = fill(zero(U), maxiter + 1, maxiter + !iszero(augmented))
    return KrylovSubspace{T, U, real(T), typeof(V), Matrix{U}}(
        maxiter, maxiter, augmented, zero(real(T)), false, V, H
    )
end

getV(Ks::KrylovSubspace) = @view(Ks.V[:, 1:(Ks.m + 1)])
getH(Ks::KrylovSubspace) = @view(Ks.H[1:(Ks.m + 1), 1:(Ks.m + !iszero(Ks.augmented))])
function Base.resize!(Ks::KrylovSubspace{T, U}, maxiter::Integer) where {T, U}
    isaugmented = !iszero(Ks.augmented)
    V = similar(Ks.V, T, (size(Ks.V, 1), maxiter + 1))
    H = similar(Ks.H, U, (maxiter + 1, maxiter + isaugmented))
    fill!(H, zero(U))
    if isaugmented
        copyto!(@view(V[axes(Ks.V)...]), Ks.V)
        copyto!(@view(H[axes(Ks.H)...]), Ks.H)
    end
    Ks.V = V
    Ks.H = H
    Ks.m = Ks.maxiter = maxiter
    return Ks
end
function Base.show(io::IO, ::MIME"text/plain", Ks::KrylovSubspace)
    io′ = IOContext(io, :limit => true, :displaysize => 3 .* displaysize(io) .÷ 7)
    println(io, "$(Ks.m)-dimensional Krylov subspace with fields")
    println(io, "beta: $(Ks.beta)")
    println(io, "V:")
    show(io′, "text/plain", getV(Ks))
    println(io)
    println(io, "H:")
    return show(io′, "text/plain", getH(Ks))
end

#######################################
# Arnoldi/Lanczos with custom IOP
## High-level interface
"""
    arnoldi(A,b[;m,tol,opnorm,iop]) -> Ks

Perform Arnoldi iterations to obtain the Krylov subspace ``K_m(A, b)``.

# Arguments

  - `A`: a square matrix or a matrix-free operator satisfying the
    Matrix-Free Operator Interface page in the manual.
  - `b`: starting vector.

# Keywords

  - `m`: requested Krylov dimension, default `min(30, size(A, 1))`.
  - `ishermitian`: select Lanczos for a Hermitian operator; defaults to
    `LinearAlgebra.ishermitian(A)`.
  - Additional keywords are forwarded to [`arnoldi!`](@ref), including `tol`,
    `iop`, and `opnorm`.

# Returns

A populated [`KrylovSubspace`](@ref). Its `m` can be smaller than requested
after a happy breakdown.

# Examples

```julia
A = [-2.0 1.0; 0.0 -1.0]
b = [1.0, 0.0]
Ks = arnoldi(A, b; m = 2)
```

The first `m + 1` columns of `Ks.V` and the leading `(m + 1)` by `m` block of
`Ks.H` are related by the recurrence formula

```
v_1=b,\\quad Av_j = \\sum_{i=1}^{j+1}h_{ij}v_i\\quad(j = 1,2,\\ldots,m)
```

`iop` determines the length of the incomplete orthogonalization procedure [^1].
The default value of 0 indicates full Arnoldi. For symmetric/Hermitian `A`,
`iop` will be ignored and the Lanczos algorithm will be used instead.

Refer to `KrylovSubspace` for more information regarding the output.

Happy-breakdown occurs whenever `norm(v_j) < tol * opnorm`, in this case,
the dimension of `Ks` is smaller than `m`.

[^1]: Koskela, A. (2015). Approximating the matrix exponential of an
    advection-diffusion operator using the incomplete orthogonalization method. In
    Numerical Mathematics and Advanced Applications-ENUMATH 2013 (pp. 345-353).
    Springer, Cham.
"""
function arnoldi(
        A, b; m = min(30, size(A, 1)), ishermitian = LinearAlgebra.ishermitian(A),
        kwargs...
    )
    TA, Tb = eltype(A), eltype(b)
    T = promote_type(TA, Tb)
    n = length(b)
    U = ishermitian ? real(T) : T

    # V stores the Krylov vectors
    V = similar(b, T, (n, m + 1))
    # H is a small dense array in tridiagonal form
    H = zeros(U, (m + 1, m))

    Ks = KrylovSubspace{T, U, real(T), typeof(V), typeof(H)}(
        m, m, false, zero(real(T)), false, V, H
    )

    return arnoldi!(Ks, A, b; m = m, ishermitian = ishermitian, kwargs...)
end

## Low-level interface
@inline function applyA!(y, A, x, V, j, n, p)
    # We cannot add `@inbounds` to `mul!`, because it is provided by the user.
    mul!(y, A, x)
    return
end

# augmented
# split augmented V? so this runs on GPU
@inline function applyA!(y, A::Tuple, x, V, j, n, p)
    A, B = A
    @inbounds begin
        # V[1:n, j + 1] = A * @view(V[1:n, j]) + B * @view(V[n+1:n+p, j])
        mul!(@view(V[1:n, j + 1]), A, @view(V[1:n, j]))

        mul!(@view(V[1:n, j + 1]), B, @view(V[(n + 1):(n + p), j]), true, true)
        copyto!(@view(V[(n + 1):(n + p - 1), j + 1]), @view(V[(n + 2):(n + p), j]))
        V[end, j + 1] = 0
    end
    return
end

function checkdims(A, b, V)
    isaugmented = b isa Tuple # isaugmented
    if isaugmented
        b′, b_aug = b
        n, p = length(b′), length(b_aug)
        _A = first(A)
    else
        n, p = size(V, 1), 0
        _A, b′, b_aug = A, b, nothing
    end
    length(b′) == size(_A, 1) == size(_A, 2) == size(V, 1) - p ||
        throw(DimensionMismatch("length(b′) [$(length(b′))] == size(_A,1) [$(size(_A, 1))] == size(_A,2) [$(size(_A, 2))] == size(V, 1)-p [$(size(V, 1) - p)] doesn't hold"))
    return b′, b_aug, n, p
end

##############################
# Utilities
##############################
"""
    firststep!(Ks, V, H, b) -> nothing

Initialize an Arnoldi or Lanczos iteration: zero `H`, set `Ks.beta` to
`norm(b)`, and store the normalized starting vector `b / norm(b)` in the first
column of `V`. A zero `b` leaves the first column untouched, with
`Ks.beta == 0` signalling the caller to stop.

# Arguments

  - `Ks`: Krylov storage whose `beta` field is set.
  - `V`: basis storage; its first column is overwritten.
  - `H`: recurrence-coefficient storage, filled with zeros.
  - `b`: starting vector.
"""
function firststep!(Ks::KrylovSubspace, V, H, b)
    return @inbounds begin
        fill!(H, zero(eltype(H)))
        Ks.beta = norm(b)
        if !iszero(Ks.beta)
            invbeta = inv(Ks.beta)
            if fast_scalar_indexing(V)
                # Explicit loop rather than `@. V[:, 1] = b / Ks.beta`: the column
                # `view(V, :, 1)` the broadcast would materialize does not elide,
                # allocating a SubArray on every arnoldi! call. Scalar indexing is
                # only valid where it is fast (CPU arrays).
                for i in eachindex(b)
                    V[i, 1] = b[i] * invbeta
                end
            else
                # GPU / non-scalar-indexing arrays: broadcast into the column.
                @views V[:, 1] .= b .* invbeta
            end
        end
    end
end

"""
    firststep!(Ks, V, H, b, b_aug, t, mu, l) -> nothing

Initialize an Arnoldi or Lanczos iteration for an augmented system: fill
`b_aug` with the scaled powers `t^i / i! * mu`, zero `H`, and store the
normalized concatenation of column `l` of `b` with `b_aug` in the first column
of `V`. `Ks.beta` is set to the norm of that concatenated vector.

# Arguments

  - `Ks`: Krylov storage whose `beta` field is set.
  - `V`: basis storage; its first column is overwritten.
  - `H`: recurrence-coefficient storage, filled with zeros.
  - `b`: matrix whose column `l` is the physical starting vector.
  - `b_aug`: augmented-part storage, overwritten in place.
  - `t`, `mu`: time and scale defining the augmented entries.
  - `l`: index of the column of `b` to use.
"""
function firststep!(Ks::KrylovSubspace, V, H, b, b_aug, t, mu, l)
    return @inbounds begin
        n, p = length(b), length(b_aug)
        map!(b_aug, 1:p) do k
            k == p && return mu
            i = p - k
            return t^i / factorial(i) * mu
        end

        # Initialize the matrices V and H
        fill!(H, 0)

        # Normalize initial vector (this norm is nonzero)
        bl = @view b[:, l]
        Ks.beta = beta = sqrt(bl'bl + b_aug'b_aug)

        if !iszero(Ks.beta)
            # The first Krylov basis vector
            @. V[1:n, 1] = bl / beta
            @. V[(n + 1):(n + p), 1] = b_aug / beta
        end
    end
end

##############################
# Arnoldi
##############################
"""
    arnoldi_step!(j, iop, A, V, H, n = -1, p = -1) -> β

Take step `j` of the Arnoldi iteration: apply `A` to column `j` of `V`,
orthogonalize the result against the previous `iop` basis vectors (all of them
when `iop >= j`), store the coefficients in column `j` of `H`, and normalize
the remainder into column `j + 1` of `V`.

# Arguments

  - `j`: 1-based index of the basis vector being extended.
  - `iop`: incomplete-orthogonalization length.
  - `A`: matrix, matrix-free operator, or augmented-operator tuple.
  - `V`: basis storage; column `j + 1` is overwritten.
  - `H`: Hessenberg storage; column `j` is overwritten.
  - `n`, `p`: physical dimension and augmented size, or `-1` when not
    augmented.

# Returns

`β = H[j + 1, j]`, the norm of the unnormalized new basis vector. A value
below the caller's tolerance signals a happy breakdown.
"""
function arnoldi_step!(
        j::Integer, iop::Integer, A::AT,
        V::AbstractMatrix{T}, H::AbstractMatrix{U},
        n::Int = -1, p::Int = -1
    ) where {AT, T, U}
    x, y = @view(V[:, j]), @view(V[:, j + 1])
    applyA!(y, A, x, V, j, n, p)

    # NOTE: H should always be Array
    # on CUDA, we prefer to perform dot
    # using CUBLAS and store the result in Array
    # since the size of H is rather small
    @inbounds for i in max(1, j - iop + 1):j
        α = H[i, j] = coeff(U, dot(@view(V[:, i]), y))
        axpy!(-α, @view(V[:, i]), y)
    end
    β = H[j + 1, j] = norm(y)
    @. y /= β
    return β
end

"""
    arnoldi!(Ks,A,b[;tol,m,opnorm,iop,init]) -> Ks

Populate an existing [`KrylovSubspace`](@ref) without allocating its basis.

# Arguments

  - `Ks`: reusable Krylov storage whose element and storage types must support
    `A` and `b`.
  - `A`: square matrix or matrix-free operator.
  - `b`: starting vector.

# Keywords

  - `tol`: happy-breakdown threshold, default `1e-7`.
  - `m`: requested Krylov dimension, bounded by `Ks.maxiter` unless the cache
    is resized.
  - `ishermitian`: select [`lanczos!`](@ref) for Hermitian operators.
  - `iop`: incomplete-orthogonalization length; `0` performs full Arnoldi.
  - `init`, `t`, `mu`, `l`: advanced controls for continuing or augmented
    Krylov constructions.

# Returns

The mutated `Ks`.

# Examples

```julia
A = [-2.0 1.0; 0.0 -1.0]
b = [1.0, 0.0]
Ks = KrylovSubspace{Float64}(length(b), 2)
arnoldi!(Ks, A, b; m = 2)
```
"""
function arnoldi!(
        Ks::KrylovSubspace{T1, U}, A::AT, b;
        tol::Real = 1.0e-7, m::Int = min(Ks.maxiter, size(A, 1)),
        ishermitian::Bool = LinearAlgebra.ishermitian(A isa Tuple ? first(A) : A),
        opnorm = nothing, iop::Int = 0,
        init::Int = 0, t::Number = NaN, mu::Number = NaN,
        l::Int = -1
    ) where {T1 <: Number, U <: Number, AT}
    Ks.wasbreakdown = false

    ishermitian &&
        return lanczos!(Ks, A, b; tol = tol, m = m, init = init, t = t, mu = mu, l = l)
    m > Ks.maxiter ? resize!(Ks, m) : Ks.m = m # might change if happy-breakdown occurs
    @inbounds V, H = getV(Ks), getH(Ks)
    b′, b_aug, n, p = checkdims(A, b, V)
    if iszero(init)
        isaugmented = AT <: Tuple
        isaugmented ? firststep!(Ks::KrylovSubspace, V, H, b′, b_aug, t, mu, l) :
            firststep!(Ks::KrylovSubspace, V, H, b)
        init = 1
    end
    iszero(Ks.beta) && return Ks
    iszero(iop) && (iop = m)
    for j in init:m
        beta = arnoldi_step!(j, iop, A, V, H, n, p)
        if beta < tol # happy-breakdown
            Ks.m = j
            Ks.wasbreakdown = true
            break
        end
    end
    return Ks
end

##############################
# Lanczos
##############################

"""
    lanczos_step!(j, A, V, u, v, n = -1, p = -1) -> β

Take step `j` of the Lanczos iteration: apply the Hermitian operator `A` to
column `j` of `V`, orthogonalize against columns `j` and `j - 1`, and
normalize the remainder into column `j + 1` of `V`.

# Arguments

  - `j`: 1-based index of the basis vector being extended.
  - `A`: Hermitian matrix, matrix-free operator, or augmented-operator tuple.
  - `V`: basis storage; column `j + 1` is overwritten.
  - `u`: diagonal recurrence coefficients; `u[j]` is overwritten.
  - `v`: off-diagonal recurrence coefficients; `v[j]` is overwritten.
  - `n`, `p`: physical dimension and augmented size, or `-1` when not
    augmented.

# Returns

`β = v[j]`, the norm of the unnormalized new basis vector. A value below the
caller's tolerance signals a happy breakdown.
"""
function lanczos_step!(
        j::Integer, A,
        V::AbstractMatrix{T},
        u::AbstractVector{U},
        v::AbstractVector{B},
        n::Int = -1, p::Int = -1
    ) where {B, T, U}
    x, y = @view(V[:, j]), @view(V[:, j + 1])
    applyA!(y, A, x, V, j, n, p)
    α = u[j] = coeff(U, dot(x, y))
    axpy!(-α, x, y)
    j > 1 && axpy!(-v[j - 1], @view(V[:, j - 1]), y)
    β = v[j] = norm(y)
    @. y /= β
    return β
end

"""
    coeff(::Type{U}, α) -> U-compatible coefficient

Return the recurrence coefficient `α` in the storage type `U`: `α` itself when
`α` already has type `U`, and `real(α)` when `U <: Real` and `α` is complex,
as required for the real recurrence coefficients of a Hermitian problem.
"""
coeff(::Type{T}, α::T) where {T} = α
coeff(::Type{U}, α::T) where {U <: Real, T <: Complex} = real(α)

"""
    realview(::Type{R}, V) -> AbstractVector{R}

Return a view of the real parts of `V` when `V` is a complex vector with real
part type `R`, and `V` itself when `V` is already a vector of `R`. Writing
through the complex case sets the real parts and leaves the imaginary parts
untouched; callers use it on recurrence-coefficient storage that is real for a
Hermitian problem but may be held in complex storage.
"""
function realview(::Type{R}, V::AbstractVector{C}) where {R, C <: Complex}
    return @view(reinterpret(R, V)[1:2:end])
end
realview(::Type{R}, V::AbstractVector{R}) where {R} = V

"""
    lanczos!(Ks,A,b[;tol,m,opnorm]) -> Ks

Populate an existing [`KrylovSubspace`](@ref) with the Lanczos recurrence for
a Hermitian operator.

# Arguments

  - `Ks`: reusable Krylov storage with real-valued Hessenberg coefficients.
  - `A`: Hermitian matrix or matrix-free operator.
  - `b`: starting vector.

# Keywords

`tol`, `m`, `opnorm`, `init`, `t`, `mu`, and `l` have the same role as for
[`arnoldi!`](@ref). Call this only when `A` is Hermitian for the scalar product
implemented by `mul!`.

# Returns

The mutated `Ks`.

# Examples

```julia
using LinearAlgebra

A = Hermitian([-2.0 1.0; 1.0 -2.0])
b = [1.0, 0.0]
Ks = KrylovSubspace{Float64, Float64}(length(b), 2)
lanczos!(Ks, A, b; m = 2)
```
"""
function lanczos!(
        Ks::KrylovSubspace{T1, U, B}, A::AT, b;
        tol = 1.0e-7, m = min(Ks.maxiter, size(A, 1)),
        opnorm = nothing,
        init::Int = 0, t::Number = NaN, mu::Number = NaN,
        l::Int = -1
    ) where {T1 <: Number, U <: Number, B, AT}
    Ks.wasbreakdown = false

    m > Ks.maxiter ? resize!(Ks, m) : Ks.m = m # might change if happy-breakdown occurs
    @inbounds V, H = getV(Ks), getH(Ks)
    b′, b_aug, n, p = checkdims(A, b, V)
    if iszero(init)
        isaugmented = AT <: Tuple
        isaugmented ? firststep!(Ks::KrylovSubspace, V, H, b′, b_aug, t, mu, l) :
            firststep!(Ks::KrylovSubspace, V, H, b)
        init = 1
    end
    iszero(Ks.beta) && return Ks
    @inbounds begin
        u = @diagview(H)
        # `v` is always real, even though `u` may (in general) be complex.
        v = realview(B, @diagview(H, -1))
    end
    for j in 1:m
        if tol > lanczos_step!(j, A, V, u, v, n, p)
            # happy-breakdown
            Ks.m = j
            Ks.wasbreakdown = true
            break
        end
    end
    @inbounds copyto!(@diagview(H, 1), @view(v[1:(end - 1)]))
    return Ks
end
