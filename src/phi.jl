# Dense algorithms relating to the evaluation of scalar/matrix phi functions
# that are used by the exponential integrators.

"""
    phi(z,k[;cache]) -> [phi_0(z),phi_1(z),...,phi_k(z)]

Compute the scalar phi functions for all orders up to k.

The phi functions are defined as

```math
\\varphi_0(z) = \\exp(z),\\quad \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - 1}{z}
```

Instead of using the recurrence relation, which is numerically unstable, a
formula given by Sidje is used (Sidje, R. B. (1998). Expokit: a software
package for computing matrix exponentials. ACM Transactions on Mathematical
Software (TOMS), 24(1), 130-156. Theorem 1).
"""
function phi(z::T, k::Integer; cache=nothing) where {T <: Number}
    # Construct the matrix
    if cache == nothing
        cache = fill(zero(T), k+1, k+1)
    else
        fill!(cache, zero(T))
    end
    cache[1,1] = z
    for i = 1:k
        cache[i,i+1] = one(T)
    end
    P = exp!(cache)
    return P[1,:]
end

"""
    phiv_dense(A,v,k[;cache]) -> [phi_0(A)v phi_1(A)v ... phi_k(A)v]

Compute the matrix-phi-vector products for small, dense `A`. `k`` >= 1.

The phi functions are defined as

```math
\\varphi_0(z) = \\exp(z),\\quad \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - 1}{z}
```

Instead of using the recurrence relation, which is numerically unstable, a
formula given by Sidje is used (Sidje, R. B. (1998). Expokit: a software
package for computing matrix exponentials. ACM Transactions on Mathematical
Software (TOMS), 24(1), 130-156. Theorem 1).
"""
function phiv_dense(A, v, k; kwargs...)
    w = Matrix{eltype(A)}(undef, length(v), k+1)
    phiv_dense!(w, A, v, k; kwargs...)
end
"""
    phiv_dense!(w,A,v,k[;cache]) -> w

Non-allocating version of `phiv_dense`.
"""
function phiv_dense!(w::AbstractMatrix{T}, A::AbstractMatrix{T},
                     v::AbstractVector{T}, k::Integer; cache=nothing) where {T <: Number}
    @assert size(w, 1) == size(A, 1) == size(A, 2) == length(v) "Dimension mismatch"
    @assert size(w, 2) == k+1 "Dimension mismatch"
    m = length(v)
    # Construct the extended matrix
    if cache == nothing
        cache = fill(zero(T), m+k, m+k)
    else
        @assert size(cache) == (m+k, m+k) "Dimension mismatch"
        fill!(cache, zero(T))
    end
    cache[1:m, 1:m] = A
    cache[1:m, m+1] = v
    for i = m+1:m+k-1
        cache[i, i+1] = one(T)
    end
    P = exp!(cache)
    # Extract results
    @views mul!(w[:, 1], P[1:m, 1:m], v)
    @inbounds for i = 1:k
        @inbounds for j = 1:m
            w[j, i+1] = P[j, m+i]
        end
    end
    return w
end

"""
    phi(A,k[;cache]) -> [phi_0(A),phi_1(A),...,phi_k(A)]

Compute the matrix phi functions for all orders up to k. `k` >= 1.

The phi functions are defined as

```math
\\varphi_0(z) = \\exp(z),\\quad \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - 1}{z}
```

Calls `phiv_dense` on each of the basis vectors to obtain the answer. If A
is `Diagonal`, instead calls the scalar `phi` on each diagonal element and the
return values are also `Diagonal`s
"""
function phi(A::AbstractMatrix{T}, k; kwargs...) where {T <: Number}
    m = size(A, 1)
    if A isa Diagonal
        out = [similar(A) for i = 1:k+1]
    else
        out = [Matrix{T}(undef, m, m) for i = 1:k+1]
    end
    phi!(out, A, k; kwargs...)
end
"""
    phi!(out,A,k[;caches]) -> out

Non-allocating version of `phi` for non-diagonal matrix inputs.
"""
function phi!(out::Vector{Matrix{T}}, A::AbstractMatrix{T}, k::Integer; caches=nothing) where {T <: Number}
    m = size(A, 1)
    @assert length(out) == k + 1 && all(P -> size(P) == (m,m), out) "Dimension mismatch"
    if caches == nothing
        e = Vector{T}(undef, m)
        W = Matrix{T}(undef, m, k+1)
        C = Matrix{T}(undef, m+k, m+k)
    else
        e, W, C = caches
        @assert size(e) == (m,) && size(W) == (m, k+1) && size(C) == (m+k, m+k) "Dimension mismatch"
    end
    @inbounds for i = 1:m
        fill!(e, zero(T)); e[i] = one(T) # e is the ith basis vector
        phiv_dense!(W, A, e, k; cache=C) # W = [phi_0(A)*e phi_1(A)*e ... phi_k(A)*e]
        @inbounds for j = 1:k+1
            @inbounds for s = 1:m
                out[j][s, i] = W[s, j]
            end
        end
    end
    return out
end
function phi!(out::Vector{Diagonal{T,V}}, A::Diagonal{T,V}, k::Integer;
              caches=nothing) where {T <: Number, V <: AbstractVector{T}}
    for i = 1:size(A, 1)
        phiz = phi(A[i, i], k; cache=caches)
        for j = 1:k+1
            out[j][i, i] = phiz[j]
        end
    end
    return out
end
