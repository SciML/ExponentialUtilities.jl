# Compute ϕm(A)b using the Krylov subspace Km{A,b}

############################
# Cache for expv
mutable struct ExpvCache{T}
    mem::Vector{T}
    ExpvCache{T}(maxiter::Int) where {T} = new{T}(Vector{T}(undef, maxiter^2))
end
function Base.resize!(C::ExpvCache{T}, maxiter::Int) where {T}
    C.mem = Vector{T}(undef, maxiter^2 * 2)
    return C
end
function get_cache(C::ExpvCache, m::Int)
    m^2 > length(C.mem) && resize!(C, m) # resize the cache if needed
    reshape(@view(C.mem[1:m^2]), m, m)
end

############################
# Expv
"""
    expv(t,A,b; kwargs) -> exp(tA)b

Compute the matrix-exponential-vector product using Krylov.

A Krylov subspace is constructed using `arnoldi` and `exp!` is called
on the Hessenberg matrix. Consult `arnoldi` for the values of the
keyword arguments. An alternative algorithm, where an error estimate
generated on-the-fly is used to terminate the Krylov iteration, can be
employed by setting the kwarg `mode=:error_estimate`.

    expv(t,Ks; cache) -> exp(tA)b

Compute the expv product using a pre-constructed Krylov subspace.
"""
function expv(t, A, b; mode=:happy_breakdown, kwargs...)
    # Master dispatch function
    if mode == :happy_breakdown
        _expv_hb(t, A, b; kwargs...)
    elseif mode == :error_estimate
        _expv_ee(t, A, b; kwargs...)
    else
        throw(ArgumentError("Unknown Krylov iteration termination mode, $(mode)"))
    end
end
function _expv_hb(t, A, b; cache=nothing, kwargs_arnoldi...)
    # Happy-breakdown mode: first construct Krylov subspace then expv!
    Ks = arnoldi(A, b; kwargs_arnoldi...)
    w = similar(b)
    expv!(w, t, Ks; cache=cache)
end
function _expv_ee(t, A, b; m=min(30, size(A, 1)), tol=1e-7, rtol=√(tol))
    # Error-estimate mode: construction of Krylov subspace and expv! at the same time
    n = size(A,1)
    T = promote_type(typeof(t), eltype(A), eltype(b))
    U = ishermitian(A) ? real(T) : T
    Ks = KrylovSubspace{T,U}(n, m)
    w = similar(b)
    expv!(w, t, A, b, Ks, get_subspace_cache(Ks); atol=tol, rtol=rtol)
end
function expv(t, Ks::KrylovSubspace{B, T, U}; kwargs...) where {B, T, U}
    n = size(getV(Ks), 1)
    w = Vector{T}(undef, n)
    expv!(w, t, Ks; kwargs...)
end
"""
    expv!(w,t,Ks[;cache]) -> w

Non-allocating version of `expv` that uses precomputed Krylov subspace `Ks`.
"""
function expv!(w::AbstractVector{T}, t::Number, Ks::KrylovSubspace{B, T, U};
               cache=nothing) where {B, T <: Number, U <: Number}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert length(w) == size(V, 1) "Dimension mismatch"
    if cache == nothing
        cache = Matrix{U}(undef, m, m)
    elseif isa(cache, ExpvCache)
        cache = get_cache(cache, m)
    else
        throw(ArgumentError("Cache must be an ExpvCache"))
    end
    copyto!(cache, @view(H[1:m, :]))
    if ishermitian(cache)
        # Optimize the case for symtridiagonal H
        F = eigen!(SymTridiagonal(cache))
        expHe = F.vectors * (exp.(lmul!(t,F.values)) .* @view(F.vectors[1, :]))
    else
        expH = exp!(lmul!(t,cache))
        expHe = @view(expH[:, 1])
    end
    lmul!(beta, mul!(w, @view(V[:, 1:m]), expHe)) # exp(A) ≈ norm(b) * V * exp(H)e
end

############################
# Cache for phiv
mutable struct PhivCache{T}
    mem::Vector{T}
    function PhivCache{T}(maxiter::Int, p::Int) where {T}
        numelems = maxiter + maxiter^2 + (maxiter + p)^2 + maxiter*(p + 1)
        new{T}(Vector{T}(undef, numelems))
    end
end
function Base.resize!(C::PhivCache{T}, maxiter::Int, p::Int) where {T}
    numelems = maxiter + maxiter^2 + (maxiter + p)^2 + maxiter*(p + 1)
    C.mem = Vector{T}(undef, numelems * 2)
    return C
end
function get_caches(C::PhivCache, m::Int, p::Int)
    numelems = m + m^2 + (m + p)^2 + m*(p + 1)
    numelems^2 > length(C.mem) && resize!(C, m, p) # resize the cache if needed
    e = @view(C.mem[1:m]); offset = m
    Hcopy = reshape(@view(C.mem[offset + 1:offset + m^2]), m, m); offset += m^2
    C1 = reshape(@view(C.mem[offset + 1:offset + (m+p)^2]), m+p, m+p); offset += (m+p)^2
    C2 = reshape(@view(C.mem[offset + 1:offset + m*(p+1)]), m, p+1)
    return e, Hcopy, C1, C2
end

############################
# Phiv
"""
    phiv(t,A,b,k;correct,kwargs) -> [phi_0(tA)b phi_1(tA)b ... phi_k(tA)b][, errest]

Compute the matrix-phi-vector products using Krylov. `k` >= 1.

The phi functions are defined as

```math
\\varphi_0(z) = \\exp(z),\\quad \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - 1}{z}
```

A Krylov subspace is constructed using `arnoldi` and `phiv_dense` is called
on the Hessenberg matrix. If `correct=true`, then phi_0 through phi_k-1 are
updated using the last Arnoldi vector v_m+1 [^1]. If `errest=true` then an
additional error estimate for the second-to-last phi is also returned. For
the additional keyword arguments, consult `arnoldi`.

  phiv(t,Ks,k;correct,kwargs) -> [phi_0(tA)b phi_1(tA)b ... phi_k(tA)b][, errest]

Compute the matrix-phi-vector products using a pre-constructed Krylov subspace.

[^1]: Niesen, J., & Wright, W. (2009). A Krylov subspace algorithm for evaluating
the φ-functions in exponential integrators. arXiv preprint arXiv:0907.4631.
Formula (10).
"""
function phiv(t, A, b, k; cache=nothing, correct=false, errest=false, kwargs_arnoldi...)
    Ks = arnoldi(A, b; kwargs_arnoldi...)
    w = Matrix{eltype(b)}(undef, length(b), k+1)
    phiv!(w, t, Ks, k; cache=cache, correct=correct, errest=errest)
end
function phiv(t, Ks::KrylovSubspace{B, T, U}, k; kwargs...) where {B, T, U}
    n = size(getV(Ks), 1)
    w = Matrix{T}(undef, n, k+1)
    phiv!(w, t, Ks, k; kwargs...)
end
"""
    phiv!(w,t,Ks,k[;cache,correct,errest]) -> w[,errest]

Non-allocating version of 'phiv' that uses precomputed Krylov subspace `Ks`.
"""
function phiv!(w::AbstractMatrix{T}, t::Number, Ks::KrylovSubspace{B, T, U}, k::Integer;
               cache=nothing, correct=false, errest=false) where {B, T <: Number, U <: Number}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert size(w, 1) == size(V, 1) "Dimension mismatch"
    @assert size(w, 2) == k + 1 "Dimension mismatch"
    if cache == nothing
        cache = PhivCache{T}(m, k)
    elseif !isa(cache, PhivCache)
        throw(ArgumentError("Cache must be a PhivCache"))
    end
    e, Hcopy, C1, C2 = get_caches(cache, m, k)
    lmul!(t, copyto!(Hcopy, @view(H[1:m, :])))
    fill!(e, zero(T)); e[1] = one(T) # e is the [1,0,...,0] basis vector
    phiv_dense!(C2, Hcopy, e, k; cache=C1) # C2 = [ϕ0(H)e ϕ1(H)e ... ϕk(H)e]
    lmul!(beta, mul!(w, @view(V[:, 1:m]), C2)) # f(A) ≈ norm(b) * V * f(H)e
    if correct
        # Use the last Arnoldi vector for correction with little additional cost
        # correct_p = beta * h_{m+1,m} * (em^T phi_p+1(H) e1) * v_m+1
        betah = beta * H[end,end] * t
        vlast = @view(V[:,end])
        @inbounds for i = 1:k
            axpy!(betah * C2[end, i+1], vlast, @view(w[:, i]))
        end
    end
    if errest
        err = abs(beta * H[end, end] * t * C2[end, end])
        return w, err
    else
        return w
    end
end
