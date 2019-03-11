# Arnoldi/Lanczos iteration algorithms (with custom IOP)

#######################################
# Output type/cache
"""
    KrylovSubspace{T}(n,[maxiter=30]) -> Ks

Constructs an uninitialized Krylov subspace, which can be filled by `arnoldi!`.

The dimension of the subspace, `Ks.m`, can be dynamically altered but should
be smaller than `maxiter`, the maximum allowed arnoldi iterations.

    getV(Ks) -> V
    getH(Ks) -> H

Access methods for the (extended) orthonormal basis `V` and the (extended)
Gram-Schmidt coefficients `H`. Both methods return a view into the storage
arrays and has the correct dimensions as indicated by `Ks.m`.

    resize!(Ks, maxiter) -> Ks

Resize `Ks` to a different `maxiter`, destroying its contents.

This is an expensive operation and should be used scarcely.
"""
mutable struct KrylovSubspace{B, T, U}
    m::Int        # subspace dimension
    maxiter::Int  # maximum allowed subspace size
    augmented::Int# length of the augmented part
    beta::B       # norm(b,2)
    V::Matrix{T}  # orthonormal bases
    H::Matrix{U}  # Gram-Schmidt coefficients (real for Hermitian matrices)
    KrylovSubspace{T,U}(n::Integer, maxiter::Integer=30, augmented::Integer=false) where {T,U} =
        new{real(T), T, U}(maxiter, maxiter, augmented, zero(real(T)), Matrix{T}(undef, n + augmented, maxiter + 1),
                           fill(zero(U), maxiter + 1, maxiter + !iszero(augmented)))
    KrylovSubspace{T}(args...) where {T} = KrylovSubspace{T,T}(args...)
end
getV(Ks::KrylovSubspace) = @view(Ks.V[:, 1:Ks.m + 1])
getH(Ks::KrylovSubspace) = @view(Ks.H[1:Ks.m + 1, 1:Ks.m+!iszero(Ks.augmented)])
function Base.resize!(Ks::KrylovSubspace{B,T,U}, maxiter::Integer) where {B,T,U}
    isaugmented = !iszero(Ks.augmented)
    V = Matrix{T}(undef, size(Ks.V, 1), maxiter + 1)
    H = fill(zero(U), maxiter + 1, maxiter + isaugmented)
    if isaugmented
      copyto!(@view(V[axes(Ks.V)...]), Ks.V)
      copyto!(@view(H[axes(Ks.H)...]), Ks.H)
    end
    Ks.V = V; Ks.H = H
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
    show(io′, "text/plain", getH(Ks))
end

#######################################
# Arnoldi/Lanczos with custom IOP
## High-level interface
"""
    arnoldi(A,b[;m,tol,opnorm,iop]) -> Ks

Performs `m` anoldi iterations to obtain the Krylov subspace K_m(A,b).

The n x (m + 1) basis vectors `getV(Ks)` and the (m + 1) x m upper Hessenberg
matrix `getH(Ks)` are related by the recurrence formula

```
v_1=b,\\quad Av_j = \\sum_{i=1}^{j+1}h_{ij}v_i\\quad(j = 1,2,\\ldots,m)
```

`iop` determines the length of the incomplete orthogonalization procedure [^1].
The default value of 0 indicates full Arnoldi. For symmetric/Hermitian `A`,
`iop` will be ignored and the Lanczos algorithm will be used instead.

Refer to `KrylovSubspace` for more information regarding the output.

Happy-breakdown occurs whenver `norm(v_j) < tol * opnorm`, in this case
the dimension of `Ks` is smaller than `m`.

[^1]: Koskela, A. (2015). Approximating the matrix exponential of an
advection-diffusion operator using the incomplete orthogonalization method. In
Numerical Mathematics and Advanced Applications-ENUMATH 2013 (pp. 345-353).
Springer, Cham.
"""
function arnoldi(A, b; m=min(30, size(A, 1)), ishermitian=LinearAlgebra.ishermitian(A), kwargs...)
    TA, Tb = eltype(A), eltype(b)
    T = promote_type(TA, Tb)
    Ks = KrylovSubspace{T, ishermitian ? real(T) : T}(length(b), m)
    arnoldi!(Ks, A, b; m=m, ishermitian=ishermitian, kwargs...)
end

## Low-level interface

@inline function applyA!(y, A::AT, x, V, j, n, p) where AT
    # We cannot add `@inbounds` to `mul!`, because it is provided by the user.
    AT <: Tuple #= augmented =# || (mul!(y, A, x); return)
    A, B = A
    @inbounds begin
        # V[1:n, j + 1] = A * @view(V[1:n, j]) + B * @view(V[n+1:n+p, j])
        mul!(@view(V[1:n, j + 1]), A, @view(V[1:n, j]))
        BLAS.gemm!('N', 'N', 1.0, B, @view(V[n+1:n+p, j]), 1.0, @view(V[1:n, j + 1]))
        copyto!(@view(V[n+1:n+p-1, j + 1]), @view(V[n+2:n+p, j]))
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
    length(b′) == size(_A,1) == size(_A,2) == size(V, 1)-p || throw(DimensionMismatch("length(b′) [$(length(b′))] == size(_A,1) [$(size(_A,1))] == size(_A,2) [$(size(_A,2))] == size(V, 1)-p [$(size(V, 1)-p)] doesn't hold"))
    return b′, b_aug, n, p
end

##############################
# Utilities
##############################
"""
    firststep!(Ks, V, H, b) -> nothing

Compute the first step of Arnoldi or Lanczos iteration.
"""
function firststep!(Ks::KrylovSubspace, V, H, b)
    @inbounds begin
        fill!(H, zero(eltype(H)))
        Ks.beta = norm(b)
        @. V[:, 1] = b / Ks.beta
    end
    return
end

"""
    firststep!(Ks, V, H, b, b_aug, t, mu, l) -> nothing

Compute the first step of Arnoldi or Lanczos iteration of augmented system.
"""
function firststep!(Ks::KrylovSubspace, V, H, b, b_aug, t, mu, l)
    @inbounds begin
        n, p = length(b), length(b_aug)
        for k=1:p-1
            i = p - k
            b_aug[k] = t^i/factorial(i) * mu
        end
        b_aug[p] = mu

        # Initialize the matrices V and H
        fill!(H, 0)

        # Normalize initial vector (this norm is nonzero)
        bl = @view b[:, l]
        Ks.beta = beta = sqrt(bl'bl + b_aug'b_aug)

        # The first Krylov basis vector
        @. V[1:n, 1]     = bl / beta
        @. V[n+1:n+p, 1] = b_aug / beta
    end
    return
end

##############################
# Arnoldi
##############################
"""
    arnoldi_step!(j, iop, n, A, V, H)

Take the `j`:th step of the Lanczos iteration.
"""
function arnoldi_step!(j::Integer, iop::Integer, A::AT,
                       V::AbstractMatrix{T}, H::AbstractMatrix{U},
                       n::Int=-1, p::Int=-1) where {AT,T,U}
    x, y = @view(V[:, j]), @view(V[:, j+1])
    applyA!(y, A, x, V, j, n, p)
    @inbounds for i = max(1, j - iop + 1):j
        α = H[i, j] = coeff(U, dot(@view(V[:, i]), y))
        axpy!(-α, @view(V[:, i]), y)
    end
    β = H[j+1, j] = norm(y)
    @. y /= β
    return β
end

"""
    arnoldi!(Ks,A,b[;tol,m,opnorm,iop,init]) -> Ks

Non-allocating version of `arnoldi`.
"""
function arnoldi!(Ks::KrylovSubspace{B, T1, U}, A::AT, b;
                  tol::Real=1e-7, m::Int=min(Ks.maxiter, size(A, 1)),
                  ishermitian::Bool=LinearAlgebra.ishermitian(A isa Tuple ? first(A) : A),
                  opnorm=LinearAlgebra.opnorm(A isa Tuple ? first(A) : A,Inf), iop::Int=0,
                  init::Int=0, t::Number=NaN, mu::Number=NaN, l::Int=-1) where {B, T1 <: Number, U <: Number, AT}
    ishermitian && return lanczos!(Ks, A, b; tol=tol, m=m, opnorm=opnorm, init=init, t=t, mu=mu, l=l)
    m > Ks.maxiter ? resize!(Ks, m) : Ks.m = m # might change if happy-breakdown occurs
    @inbounds V, H = getV(Ks), getH(Ks)
    b′, b_aug, n, p = checkdims(A, b, V)
    if iszero(init)
        isaugmented = AT <: Tuple
        isaugmented ? firststep!(Ks::KrylovSubspace, V, H, b′, b_aug, t, mu, l) : firststep!(Ks::KrylovSubspace, V, H, b)
        init = 1
    end
    # vtol = tol * opnorm
    vtol = tol * (opnorm isa Number ? opnorm : opnorm(A,Inf)) # backward compatibility
    iszero(iop) && (iop = m)
    for j = init:m
        beta = arnoldi_step!(j, iop, A, V, H, n, p)
        if beta < vtol # happy-breakdown
            Ks.m = j
            break
        end
    end
    return Ks
end

##############################
# Lanczos
##############################

"""
    lanczos_step!(j, m, n, A, V, H)

Take the `j`:th step of the Lanczos iteration.
"""
function lanczos_step!(j::Integer, A,
                       V::AbstractMatrix{T},
                       u::AbstractVector{U},
                       v::AbstractVector{B},
                       n::Int=-1, p::Int=-1) where {B,T,U}
    x, y = @view(V[:, j]),@view(V[:, j+1])
    applyA!(y, A, x, V, j, n, p)
    α = u[j] = coeff(U, dot(x, y))
    axpy!(-α, x, y)
    j > 1 && axpy!(-v[j-1], @view(V[:, j-1]), y)
    β = v[j] = norm(y)
    @. y /= β
    return β
end

"""
    coeff(::Type,α)

Helper functions that returns the real part if that is what is
required (for Hermitian matrices), otherwise returns the value
as-is.
"""
coeff(::Type{T},α::T) where {T} = α
coeff(::Type{U},α::T) where {U<:Real,T<:Complex} = real(α)


"""
    realview(::Type, V) -> real view of `V`
"""
realview(::Type{R}, V::AbstractVector{C}) where {R,C<:Complex} =
    @view(reinterpret(R, V)[1:2:end])
realview(::Type{R}, V::AbstractVector{R}) where {R} = V

"""
    lanczos!(Ks,A,b[;tol,m,opnorm]) -> Ks

A variation of `arnoldi!` that uses the Lanczos algorithm for
Hermitian matrices.
"""
function lanczos!(Ks::KrylovSubspace{B, T1, U}, A::AT, b;
                  tol=1e-7, m=min(Ks.maxiter, size(A, 1)),
                  opnorm=LinearAlgebra.opnorm(A,Inf),
                  init::Int=0, t::Number=NaN, mu::Number=NaN, l::Int=-1) where {B, T1 <: Number, U <: Number, AT}
    m > Ks.maxiter ? resize!(Ks, m) : Ks.m = m # might change if happy-breakdown occurs
    @inbounds V, H = getV(Ks), getH(Ks)
    b′, b_aug, n, p = checkdims(A, b, V)
    if iszero(init)
        isaugmented = AT <: Tuple
        isaugmented ? firststep!(Ks::KrylovSubspace, V, H, b′, b_aug, t, mu, l) : firststep!(Ks::KrylovSubspace, V, H, b)
        init = 1
    end
    @inbounds begin
        u = @diagview(H)
        # `v` is always real, even though `u` may (in general) be complex.
        v = realview(B, @diagview(H,-1))
    end
    # vtol = tol * opnorm
    vtol = tol * (opnorm isa Number ? opnorm : opnorm(A,Inf)) # backward compatibility
    for j = 1:m
        if vtol > lanczos_step!(j, A, V, u, v, n, p)
            # happy-breakdown
            Ks.m = j
            break
        end
    end
    @inbounds copyto!(@diagview(H,1), v[1:end-1])
    return Ks
end
