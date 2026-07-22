# Compute ϕm(A)b using the Krylov subspace Km{A,b}

############################
# Cache for expv
"""
    ExpvCache{T}(maxiter::Int)

Reusable scratch workspace for the in-place [`expv!`](@ref) Krylov
matrix-exponential-vector product with element type `T`. It holds a single flat
buffer sized for an `maxiter`×`maxiter` Hessenberg matrix; pass the cache as the
`cache` keyword to `expv!` to avoid reallocating that buffer on repeated calls.
The buffer is grown automatically (via `resize!`) if a later call requests a
larger subspace than the one it was allocated for.

# Arguments

  - `T`: element type of the Krylov Hessenberg workspace.
  - `maxiter`: largest Krylov dimension expected for repeated calls.

# Example

```julia
cache = ExpvCache{Float64}(30)
expv!(similar(b), 0.1, arnoldi(A, b); cache)
```

# Fields

  - `mem::Vector{T}`: flat storage of length `maxiter^2` reshaped on demand into
    the `m`×`m` working copy of the Hessenberg matrix.
"""
mutable struct ExpvCache{T, W}
    mem::Vector{T}
    # exponential! workspaces (see `alloc_mem`) keyed by extended-matrix size.
    # `W` is a single concrete type: the workspace type is fixed by the element
    # type `T` and the default `ExpMethodHigham2005Base`, and is independent of
    # the matrix size (LinearSolve's DefaultLinearSolver is a size-independent
    # wrapper), so a cache spanning several subspace sizes stores one concrete
    # type -- only the instances (one per size) differ. `W === Nothing` when `T`
    # has no `alloc_mem` preallocation (e.g. BigFloat), and exponential! is then
    # called without a workspace.
    expcache::Vector{Tuple{Int, W}}
    # First column of exp(H), copied out so the final `mul!` consumes a
    # contiguous vector rather than a column view of the reshaped `mem` buffer
    # (that view does not elide and would allocate a SubArray each call).
    expcol::Vector{T}
end
function ExpvCache{T}(maxiter::Int) where {T}
    W = Base.promote_op(alloc_mem, Matrix{T}, typeof(ExpMethodHigham2005Base()))
    return ExpvCache{T, W}(
        Vector{T}(undef, maxiter^2), Tuple{Int, W}[], Vector{T}(undef, maxiter)
    )
end
function Base.resize!(C::ExpvCache{T}, maxiter::Int) where {T}
    C.mem = Vector{T}(undef, maxiter^2 * 2)
    length(C.expcol) < maxiter && resize!(C.expcol, maxiter)
    return C
end
function get_cache(C::ExpvCache, m::Int)
    m^2 > length(C.mem) && resize!(C, m) # resize the cache if needed
    return reshape(@view(C.mem[1:(m^2)]), m, m)
end

############################
# Expv
"""
    expv(t, A, b; kwargs) -> exp(tA)b

Compute the matrix-exponential-vector product with a Krylov approximation.

# Arguments

  - `t`: scalar time or scale factor.
  - `A`: square matrix or matrix-free operator satisfying the
    Matrix-Free Operator Interface page in the manual.
  - `b`: input vector.
  - `Ks`: a precomputed [`KrylovSubspace`](@ref) for the `(A, b)` pair.

# Keywords

  - `mode`: `:happy_breakdown` (default) builds a regular Arnoldi basis;
    `:error_estimate` uses the Hermitian error-estimate method.
  - `cache`: optional [`ExpvCache`](@ref) for the reduced exponential.
  - `expmethod`: matrix-exponential implementation for the reduced problem.
  - Remaining keywords for the `(t, A, b)` form are forwarded to
    [`arnoldi`](@ref).

# Returns

The vector approximating ``\\exp(t A)b``.

A Krylov subspace is constructed using `arnoldi` and `exp!` is called
on the Hessenberg matrix. Consult `arnoldi` for the values of the
keyword arguments. An alternative algorithm, where an error estimate
generated on-the-fly is used to terminate the Krylov iteration, can be
employed by setting the kwarg `mode=:error_estimate`.

    expv(t, Ks; cache) -> exp(tA)b

Compute the expv product using a pre-constructed Krylov subspace.
"""
function expv(t, A, b; mode = :happy_breakdown, kwargs...)
    # Master dispatch function
    return if mode == :happy_breakdown
        _expv_hb(t, A, b; kwargs...)
    elseif mode == :error_estimate
        _expv_ee(t, A, b; kwargs...)
    else
        throw(ArgumentError("Unknown Krylov iteration termination mode, $(mode)"))
    end
end
function _expv_hb(
        t::Tt, A, b;
        expmethod = ExpMethodHigham2005Base(),
        cache = nothing, kwargs_arnoldi...
    ) where {Tt}
    # Happy-breakdown mode: first construct Krylov subspace then expv!
    Ks = arnoldi(A, b; kwargs_arnoldi...)
    w = similar(b, promote_type(Tt, eltype(A), eltype(b)))
    return expv!(w, t, Ks; cache = cache, expmethod = expmethod)
end
function _expv_ee(
        t::Tt, A, b; m = min(30, size(A, 1)), tol = 1.0e-7, rtol = √(tol),
        ishermitian::Bool = LinearAlgebra.ishermitian(A),
        expmethod = ExpMethodHigham2005Base()
    ) where {Tt}
    # Error-estimate mode: construction of Krylov subspace and expv! at the same time
    n = size(A, 1)
    T = promote_type(typeof(t), eltype(A), eltype(b))
    U = ishermitian ? real(T) : T
    Ks = KrylovSubspace{T, U}(n, m)
    w = similar(b, promote_type(Tt, eltype(A), eltype(b)))
    return expv!(
        w, t, A, b, Ks, get_subspace_cache(Ks); atol = tol, rtol = rtol,
        ishermitian = ishermitian, expmethod = expmethod
    )
end
function expv(
        t::Tt, Ks::KrylovSubspace{T, U}; expmethod = ExpMethodHigham2005(),
        kwargs...
    ) where {Tt, T, U}
    n = size(getV(Ks), 1)
    w = Vector{promote_type(Tt, T)}(undef, n)
    return expv!(w, t, Ks; kwargs...)
end
"""
    expv!(w,t,Ks[;cache]) -> w

Compute ``\\exp(t A)b`` from a precomputed Krylov basis without allocating the
output vector.

# Arguments

  - `w`: output vector, overwritten in place.
  - `t`: scalar time or scale factor.
  - `Ks`: populated [`KrylovSubspace`](@ref).

# Keywords

  - `cache`: `nothing` or a reusable [`ExpvCache`](@ref).
  - `expmethod`: reduced matrix-exponential implementation.

# Returns

The mutated `w`.
"""
function expv!(
        w::AbstractVector{Tw}, t::Real, Ks::KrylovSubspace{T, U};
        cache = nothing, expmethod = ExpMethodHigham2005Base()
    ) where {Tw, T, U}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert length(w) == size(V, 1) "Dimension mismatch"
    if iszero(beta)
        # Zero input: the Krylov basis V was never initialized (firststep! skips
        # the fill when beta == 0), so `beta * V * expHe` would be `0 * garbage`,
        # which is NaN whenever V holds uninitialized memory. The result is exactly
        # zero, matching the complex `expv!` method's guard below.
        w .= false
        return w
    end
    if isnothing(cache)
        Hcopy = Matrix{U}(undef, m, m)
        expwork = nothing
    elseif isa(cache, ExpvCache)
        Hcopy = get_cache(cache, m)
        expwork = get_expcache!(cache, Hcopy, expmethod)
    else
        throw(ArgumentError("Cache must be an ExpvCache"))
    end
    copyto!(Hcopy, @view(H[1:m, :]))
    Vm = @view(V[:, 1:m])
    if ishermitian(Hcopy)
        # Optimize the case for symtridiagonal H
        F = eigen!(SymTridiagonal(Hcopy))
        expHe = F.vectors * (exp.(lmul!(t, F.values)) .* @view(F.vectors[1, :]))
        return lmul!(beta, mul!(w, Vm, expHe)) # exp(A) ≈ norm(b) * V * exp(H)e
    else
        lmul!(t, Hcopy)
        _exponential!(Hcopy, expmethod, expwork)
        # Consume the first column of exp(H) as a contiguous vector. A column
        # view of `Hcopy` (a reshape of the cache buffer) does not elide; copying
        # into the cache's `expcol` keeps the mul! input allocation-free.
        if cache isa ExpvCache
            expcol = cache.expcol
            length(expcol) < m && resize!(expcol, m)
            @inbounds for i in 1:m
                expcol[i] = Hcopy[i, 1]
            end
            return lmul!(beta, mul!(w, Vm, @view(expcol[1:m])))
        else
            return lmul!(beta, mul!(w, Vm, @view(Hcopy[:, 1])))
        end
    end
end

# NOTE: Tw can be Float64, while t is ComplexF64 and T is Float32
#       or Tw can be Float64, while t is ComplexF32 and T is Float64
#       thus they can not share the same TypeVar.
function expv!(
        w::AbstractVector{Complex{Tw}}, t::Complex{Tt}, Ks::KrylovSubspace{T, U};
        cache = nothing, expmethod = ExpMethodHigham2005Base()
    ) where {Tw, Tt, T, U}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert length(w) == size(V, 1) "Dimension mismatch"
    if isnothing(cache)
        cache = Matrix{U}(undef, m, m)
    elseif isa(cache, ExpvCache)
        cache = get_cache(cache, m)
    else
        throw(ArgumentError("Cache must be an ExpvCache"))
    end
    if iszero(Ks.beta)
        w .= false
        return w
    end
    copyto!(cache, @view(H[1:m, :]))
    if ishermitian(cache)
        # Optimize the case for symtridiagonal H
        F = eigen!(SymTridiagonal(real(cache)))
        expHe = F.vectors * (exp.(t * F.values) .* @view(F.vectors[1, :]))
    else
        expH = exponential!(t * cache, expmethod)
        expHe = @view(expH[:, 1])
    end
    # `ArrayInterfaceCore.restructure` will convert the `expHe` to the target matrix type that can interact with `V`.
    return lmul!(beta, mul!(w, @view(V[:, 1:m]), compatible_multiplicative_operand(V, expHe))) # exp(A) ≈ norm(b) * V * exp(H)e
end

# GPU expv! for Real t (non-allocating in hermitian branch)
function ExponentialUtilities.expv!(
        w::GPUArraysCore.AbstractGPUVector{Tw},
        t::Real, Ks::KrylovSubspace{T, U};
        cache = nothing,
        expmethod = ExpMethodHigham2005Base()
    ) where {Tw, T, U}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert length(w) == size(V, 1) "Dimension mismatch"
    if isnothing(cache)
        cache = Matrix{U}(undef, m, m)
    elseif isa(cache, ExpvCache)
        cache = get_cache(cache, m)
    else
        throw(ArgumentError("Cache must be an ExpvCache"))
    end
    if iszero(Ks.beta)
        w .= false
        return w
    end
    copyto!(cache, @view(H[1:m, :]))
    if ishermitian(cache)
        # Optimize the case for symtridiagonal H
        F = eigen!(SymTridiagonal(cache))
        # Use lmul! to avoid allocation (modifies F.values in place)
        expHe = F.vectors * (exp.(lmul!(t, F.values)) .* @view(F.vectors[1, :]))
    else
        lmul!(t, cache)
        expH = exponential!(cache, expmethod)
        expHe = @view(expH[:, 1])
    end

    return lmul!(beta, mul!(w, @view(V[:, 1:m]), Adapt.adapt(typeof(w), expHe))) # exp(A) ≈ norm(b) * V * exp(H)e
end

# GPU expv! for Complex t (allocates in hermitian branch due to Real->Complex conversion)
function ExponentialUtilities.expv!(
        w::GPUArraysCore.AbstractGPUVector{Complex{Tw}},
        t::Complex{Tt}, Ks::KrylovSubspace{T, U};
        cache = nothing,
        expmethod = ExpMethodHigham2005Base()
    ) where {Tw, Tt, T, U}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert length(w) == size(V, 1) "Dimension mismatch"
    if isnothing(cache)
        cache = Matrix{U}(undef, m, m)
    elseif isa(cache, ExpvCache)
        cache = get_cache(cache, m)
    else
        throw(ArgumentError("Cache must be an ExpvCache"))
    end
    if iszero(Ks.beta)
        w .= false
        return w
    end
    copyto!(cache, @view(H[1:m, :]))
    if ishermitian(cache)
        # Optimize the case for symtridiagonal H
        F = eigen!(SymTridiagonal(cache))
        # Must allocate here: F.values is Real, t is Complex
        expHe = F.vectors * (exp.(t * F.values) .* @view(F.vectors[1, :]))
    else
        expH = exponential!(t * cache, expmethod)
        expHe = @view(expH[:, 1])
    end

    return lmul!(beta, mul!(w, @view(V[:, 1:m]), Adapt.adapt(typeof(w), expHe))) # exp(A) ≈ norm(b) * V * exp(H)e
end

compatible_multiplicative_operand(::AbstractArray, source::AbstractArray) = source

############################
# Cache for phiv
"""
    PhivCache(w, maxiter::Int, p::Int)

Reusable scratch workspace for the in-place [`phiv!`](@ref) Krylov
matrix-phi-vector product. It packs all the intermediate buffers needed to
evaluate `ϕ_0(tA)b … ϕ_p(tA)b` over a Krylov subspace of dimension up to
`maxiter` into a single flat vector, whose element type matches `eltype(w)`
(the output array `w`). Pass the cache as the `cache` keyword to `phiv!` to
avoid reallocating these buffers on repeated calls; the buffer is grown
automatically (via `resize!`) if a later call requests a larger subspace or
higher order than the one it was allocated for.

The `useview` type parameter records whether views into the flat buffer are
usable (`true` for ordinary `Array`s, `false` for GPU arrays, which get freshly
allocated reshaped copies instead).

# Arguments

  - `w`: representative output array whose element type and device determine
    the workspace layout.
  - `maxiter`: largest Krylov dimension expected for repeated calls.
  - `p`: highest phi-function order expected for repeated calls.

# Example

```julia
cache = PhivCache(similar(b, length(b), 3), 30, 2)
phiv!(similar(b, length(b), 3), 0.1, arnoldi(A, b), 2; cache)
```

# Fields

  - `mem::Vector{T}`: flat storage that is carved (by `get_caches`) into the
    subspace vector, a Hessenberg working copy, and the two augmented matrices
    used by the phi-function recurrence.
  - `expcache::Vector{Tuple{Int, W}}`: `exponential!` workspaces (see
    `alloc_mem`) keyed by extended-matrix size, reused across calls. `W` is a
    single concrete workspace type; see [`ExpvCache`](@ref) for why one type
    covers all sizes.
"""
mutable struct PhivCache{useview, T, W}
    mem::Vector{T}
    expcache::Vector{Tuple{Int, W}}
    # Reusable t^l/l! coefficient scratch for phiv_timestep! (which threads this
    # cache in). Kept here so phiv_timestep! need not allocate it per call; plain
    # phiv! does not touch it. `coeffs[1]` stays one(T); the rest are overwritten.
    coeffs::Vector{T}
    # Length-1 `ts` scratch for the scalar-time phiv_timestep!/expv_timestep!
    # wrappers, so they need not allocate a `[t]` per call.
    ts1::Vector{Float64}
end

function PhivCache(w, maxiter::Int, p::Int)
    numelems = maxiter + maxiter^2 + (maxiter + p)^2 + maxiter * (p + 1)
    T = eltype(w)
    mem = Vector{T}(undef, numelems)
    W = Base.promote_op(alloc_mem, Matrix{T}, typeof(ExpMethodHigham2005Base()))
    useview = !(w isa GPUArraysCore.AbstractGPUArray)
    return PhivCache{useview, T, W}(
        mem, Tuple{Int, W}[], ones(T, max(p, 1)), Vector{Float64}(undef, 1)
    )
end

"""
    get_expcache!(C, A, expmethod) -> workspace or nothing

Return a reusable `exponential!` workspace (as produced by `alloc_mem(A,
expmethod)`) stored in the cache `C`, allocating one the first time each
distinct extended-matrix size is requested. A small list of workspaces is kept
because a single integrator step may evaluate phi functions of several
different orders (and hence sizes) against the same cache; they all share the
one concrete workspace type `W` the cache was built for.

The typed store only holds workspaces for the default `ExpMethodHigham2005Base`
(the method the cache's `W` was derived from); every other method dispatches to
the fallback returning `nothing`, so the caller uses the two-argument
`exponential!`. `nothing` is likewise returned for element types with no
`alloc_mem` preallocation (`W === Nothing`, e.g. `BigFloat`). Dispatching on the
method type keeps the return concretely typed (`W` or `Nothing`, never a union).
"""
function get_expcache!(
        C::Union{ExpvCache{T, W}, PhivCache{<:Any, T, W}}, A,
        expmethod::ExpMethodHigham2005Base
    ) where {T, W}
    W === Nothing && return nothing  # nothing::Nothing === W, so return stays concrete
    n = size(A, 1)
    entries = C.expcache
    for (nc, work) in entries
        nc == n && return work
    end
    work = alloc_mem(A, expmethod)::W
    # Bound the store: adaptive Krylov can wander over many subspace sizes, and
    # each workspace holds several n×n matrices. Resetting is cheap and rare.
    length(entries) >= 16 && empty!(entries)
    push!(entries, (n, work))
    return work
end
# Non-default methods are not cached; the caller allocates via two-arg exponential!.
get_expcache!(::Union{ExpvCache, PhivCache}, A, expmethod) = nothing

# Call exponential! with the reusable workspace when one is available; the
# two-argument form allocates its own.
_exponential!(A, method, ::Nothing) = exponential!(A, method)
_exponential!(A, method, work) = exponential!(A, method, work)
function Base.resize!(C::PhivCache, maxiter::Int, p::Int)
    numelems = maxiter + maxiter^2 + (maxiter + p)^2 + maxiter * (p + 1)
    C.mem = similar(C.mem, numelems * 2)
    return C
end
function get_caches(C::PhivCache{useview, T}, m::Int, p::Int) where {useview, T}
    numelems = m + m^2 + (m + p)^2 + m * (p + 1)
    numelems > length(C.mem) && resize!(C, m, p) # resize the cache if needed
    e = @view(C.mem[1:m])
    offset = m

    if useview
        Hcopy = reshape(@view(C.mem[(offset + 1):(offset + m^2)]), m, m)
        offset += m^2
        C1 = reshape(@view(C.mem[(offset + 1):(offset + (m + p)^2)]), m + p, m + p)
        offset += (m + p)^2
        C2 = reshape(@view(C.mem[(offset + 1):(offset + m * (p + 1))]), m, p + 1)
    else
        Hcopy = reshape(C.mem[(offset + 1):(offset + m^2)], m, m)
        offset += m^2
        C1 = reshape(C.mem[(offset + 1):(offset + (m + p)^2)], m + p, m + p)
        offset += (m + p)^2
        C2 = reshape(C.mem[(offset + 1):(offset + m * (p + 1))], m, p + 1)
    end
    return e, Hcopy, C1, C2
end

############################
# Phiv
"""
    phiv(t,A,b,k;correct,kwargs) -> [phi_0(tA)b phi_1(tA)b ... phi_k(tA)b][, errest]

Compute matrix-phi-vector products with a Krylov approximation. `k >= 1`.

# Arguments

  - `t`: scalar time or scale factor.
  - `A`: square matrix or matrix-free operator satisfying the
    Matrix-Free Operator Interface page in the manual.
  - `b`: input vector.
  - `k`: highest phi-function order.
  - `Ks`: precomputed [`KrylovSubspace`](@ref) for the `(A, b)` pair.

# Keywords

  - `cache`: optional [`PhivCache`](@ref) for reduced phi-function work.
  - `correct`: apply the last-Arnoldi-vector correction to orders `0:k-1`.
  - `errest`: return `(w, estimate)` instead of just `w`.
  - Remaining keywords for the `(t, A, b, k)` form are forwarded to
    [`arnoldi`](@ref).

# Returns

An `n` by `k + 1` matrix whose columns are ``\\varphi_j(tA)b`` for
`j = 0:k`, or that matrix paired with an error estimate when `errest=true`.

The phi functions are defined as

```math
\\varphi_0(z) = \\exp(z),\\quad \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - \\varphi_k(0)}{z}
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
function phiv(
        t, A, b, k; cache = nothing, correct = false, errest = false,
        kwargs_arnoldi...
    )
    Ks = arnoldi(A, b; kwargs_arnoldi...)
    w = Matrix{eltype(b)}(undef, length(b), k + 1)
    return phiv!(w, t, Ks, k; cache = cache, correct = correct, errest = errest)
end
function phiv(t, Ks::KrylovSubspace{T, U}, k; kwargs...) where {T, U}
    n = size(getV(Ks), 1)
    w = Matrix{T}(undef, n, k + 1)
    return phiv!(w, t, Ks, k; kwargs...)
end
"""
    phiv!(w,t,Ks,k[;cache,correct,errest]) -> w[,errest]

Compute phi-vector products from a precomputed Krylov basis without allocating
the output matrix.

# Arguments

  - `w`: output matrix with `size(w, 2) == k + 1`, overwritten in place.
  - `t`: scalar time or scale factor.
  - `Ks`: populated [`KrylovSubspace`](@ref).
  - `k`: highest phi-function order.

# Keywords

`cache`, `correct`, and `errest` have the same meaning as for [`phiv`](@ref).

# Returns

The mutated `w`, or `(w, estimate)` when `errest=true`.
"""
function phiv!(
        w::AbstractMatrix, t::Number, Ks::KrylovSubspace, k::Integer;
        cache = nothing, correct = false,
        errest = false, expmethod = ExpMethodHigham2005Base()
    )
    w, err = _phiv!(w, t, Ks, k, cache, correct, expmethod)
    # Split the error estimate out into an internal helper that always returns
    # `(w, err)`: returning either `w` or `(w, err)` from `phiv!` directly makes
    # the return type a union, which boxes (an allocation) at internal callers
    # such as `phiv_timestep!` that always request the estimate.
    return errest ? (w, err) : w
end

function _phiv!(
        w::AbstractMatrix, t::Number, Ks::KrylovSubspace{T, U}, k::Integer,
        cache, correct, expmethod
    ) where {T <: Number, U <: Number}
    m, beta, V, H = Ks.m, Ks.beta, getV(Ks), getH(Ks)
    @assert size(w, 1) == size(V, 1) "Dimension mismatch"
    @assert size(w, 2) == k + 1 "Dimension mismatch"
    if isnothing(cache)
        cache = PhivCache(w, m, k)
    elseif !isa(cache, PhivCache)
        throw(ArgumentError("Cache must be a PhivCache"))
    end
    e, Hcopy, C1, C2 = get_caches(cache, m, k)
    expwork = get_expcache!(cache, C1, expmethod)
    lmul!(t, copyto!(Hcopy, @view(H[1:m, :])))
    fill!(e, zero(T))
    allowed_setindex!(e, one(T), 1) # e is the [1,0,...,0] basis vector
    phiv_dense!(C2, Hcopy, e, k; cache = C1, expmethod = expmethod, expcache = expwork) # C2 = [ϕ0(H)e ϕ1(H)e ... ϕk(H)e]
    # C2 is a strided reshape of the flat cache buffer: BLAS can consume it
    # directly, so only adapt (which copies) when w needs another storage type.
    aC2 = w isa Array ? C2 : Adapt.adapt(typeof(w), C2)
    lmul!(beta, mul!(w, @view(V[:, 1:m]), aC2)) # f(A) ≈ norm(b) * V * f(H)e
    if correct
        # Use the last Arnoldi vector for correction with little additional cost
        # correct_p = beta * h_{m+1,m} * (em^T phi_p+1(H) e1) * v_m+1
        betah = beta * H[end, end] * t
        vlast = @view(V[:, end])
        @inbounds for i in 1:k
            axpy!(betah * C2[end, i + 1], vlast, @view(w[:, i]))
        end
    end
    err = abs(beta * H[end, end] * t * C2[end, end])
    return w, err
end
