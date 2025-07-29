module ExponentialUtilitiesStaticArraysExt

export default_tolerance, theta, THETA32, THETA64

using StaticArrays
import Base: @propagate_inbounds
import LinearAlgebra: tr, I, opnorm, norm
import ExponentialUtilities

# Look-Up Table Generation
default_tolerance(::Type{T}) where {T <: AbstractFloat} = eps(T) / 2
@inline function trexp(M::Integer, x::T) where {T}
    y = T <: BigInt ? one(BigFloat) : T <: Integer ? one(Float64) : one(T)
    for m in M:-1:1
        y = 1 + x / m * y
    end
    return y
end
h(M::Integer, x::Number) = log(exp(-x) * trexp(M, x))
h̃(M::Integer, x::Number) = ifelse(isodd(M), -1, 1) * h(M, -x)
function θf((M, ϵ)::Tuple{<:Integer, <:Number}, x::Number)
    return h̃(M + 1, x) / x - ϵ
end
θf(M::Integer, ϵ::Number) = Base.Fix1(θf, (M, ϵ))
function θfp(M::Integer, x::Number)
    Tk = trexp(M + 1, -x)
    Tkm1 = trexp(M, -x)
    return ifelse(isodd(M), -1, 1) / x^2 * (log(Tk) + x * Tkm1 / Tk)
end
θfp(M::Integer) = Base.Fix1(θfp, M)

function newton_find_zero(f::Function, dfdx::Function, x0::Real;
        xrtol::Real = eps(typeof(x0)) / 2, maxiter::Integer = 100)
    0 ≤ xrtol ≤ 1 || throw(DomainError(xrtol, "relative tolerance in x must be in [0,1]"))
    maxiter > 0 || throw(DomainError(maxiter, "maxiter should be a positive integer"))
    x, xp = x0, typemax(x0)
    for _ in 1:maxiter
        xp = x
        x -= f(x) / dfdx(x)
        if abs(x - xp) ≤ xrtol * max(x, xp) || !isfinite(x)
            break
        end
    end
    return x
end
function calc_thetas(
        m_max::Integer, ::Type{T}; tol::T = default_tolerance(T)) where {T <: AbstractFloat}
    m_max > 0 || throw(DomainError(m_max, "argument m_max must be positive"))
    ϵ = BigFloat(tol)
    θ = Vector{T}(undef, m_max + 1)
    @inbounds θ[1] = eps(T)
    @inbounds for m in 1:m_max
        θ[m + 1] = newton_find_zero(θf(m, ϵ), θfp(m), big(θ[m]), xrtol = ϵ)
    end
    return θ
end

const P_MAX = 8
const M_MAX = 55
const THETA32 = Tuple(calc_thetas(M_MAX, Float32))
const THETA64 = Tuple(calc_thetas(M_MAX, Float64))

@propagate_inbounds theta(::Type{Float64}, m::Integer) = THETA64[m]
@propagate_inbounds theta(::Type{Float32}, m::Integer) = THETA32[m]
@propagate_inbounds theta(::Type{Complex{T}}, m::Integer) where {T} = theta(T, m)
@propagate_inbounds theta(::Type{T}, ::Integer) where {T} = throw(DomainError(
    T, "type must be either Float32 or Float64"))
@propagate_inbounds theta(x::Number, m::Integer) = theta(typeof(x), m)

# runtime parameter search
@propagate_inbounds @inline function calculate_s(
        α::T, m::I)::I where {T <: Number, I <: Integer}
    return ceil(I, α / theta(T, m))
end
@propagate_inbounds @inline function parameter_search(
        nA::Number, m::I)::I where {I <: Integer}
    return m * calculate_s(nA, m)
end
@propagate_inbounds @inline function parameters(A::SMatrix{
        N, N, T})::Tuple{Int, Int} where {N, T}
    1 ≤ N ≤ 50 || throw(DomainError(N,
        "leading dimension of A must be ≤ 50; larger matrices require Higham's 1-norm estimation algorithm"))
    nA = opnorm(A, 1)
    iszero(nA) && return (0, 1)
    @inbounds if nA ≤ 4theta(T, M_MAX) * P_MAX * (P_MAX + 3) / (M_MAX * 1)
        mo = argmin(Base.Fix1(parameter_search, nA), 1:M_MAX)
        s = calculate_s(nA, mo)
        return (mo, s)
    else
        Aᵐ = A * A
        pη = √(opnorm(Aᵐ, 1))
        (Cmo::Int, mo::Int) = (typemax(Int), 1)
        for p in 2:P_MAX
            Aᵐ *= A
            η = opnorm(Aᵐ, 1)^inv(p + 1)
            α = max(pη, η)
            pη = η
            (Cmp::Int, mp::Int) = findmin(
                Base.Fix1(parameter_search, α), (p * (p - 1) - 1):M_MAX)
            (Cmo, mo) = min((Cmp, mp), (Cmo, mo))
        end
        s = max(Cmo ÷ mo, 1)
        return (mo, s)
    end
end

# exponential matrix-vector product for SArray types
"""
    expv(t::Number,A::SMatrix{N,N},v::SVector{N};kwarg...) → exp(t*A)*v

    Computes the matrix-vector product exp(t*A)*v without forming exp(t*A) explicitly.
    This implementation is based on the algorithm presented in Al-Mohy & Higham (2011).
    Presently, the relative tolerance is fixed to eps(T)/2 where T is the type of the
    output.
"""
@propagate_inbounds function ExponentialUtilities.expv(
        t::Number, A::SMatrix{N, N, T}, v::SVector{N}; kwarg...) where {N, T}
    Ti = promote_type(StaticArrays.arithmetic_closure(T), eltype(v))
    N ≤ 4 && return exp(t * A) * v
    Ai::SMatrix{N, N, Ti} = A

    μ = tr(Ai) / N
    Ai -= μ * I
    Ai *= t
    mo, s = parameters(Ai)
    F = v
    Ai /= s
    η = exp(μ * t / s)
    ϵ = default_tolerance(T)
    for _ in 1:s
        c₁ = norm(v, Inf)
        for j in 1:mo
            v = (Ai * v) / j
            F += v
            c₂ = norm(v, Inf)
            c₁ + c₂ ≤ ϵ * norm(F, Inf) && break
            c₁ = c₂
        end
        F *= η
        v = F
        all(isfinite, v) || break
    end

    return F
end

end
