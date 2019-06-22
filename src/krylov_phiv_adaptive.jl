# Krylov phiv with internal time-stepping
"""
    exp_timestep(ts,A,b[;adaptive,tol,kwargs...]) -> U

Evaluates the matrix exponentiation-vector product using time stepping

```math
u = \\exp(tA)b
```

`ts` is an array of time snapshots for u, with `U[:,j] ≈ u(ts[j])`. `ts` can
also be just one value, in which case only the end result is returned and `U`
is a vector.

The time stepping formula of Niesen & Wright is used [^1]. If the time step
`tau` is not specified, it is chosen according to (17) of Neisen & Wright. If
`adaptive==true`, the time step and Krylov subsapce size adaptation scheme of
Niesen & Wright is used, the relative tolerance of which can be set using the
keyword parameter `tol`. The delta and gamma parameter of the adaptation
scheme can also be adjusted.

Set `verbose=true` to print out the internal steps (for debugging). For the
other keyword arguments, consult `arnoldi` and `phiv`, which are used
internally.

Note that this function is just a special case of `phiv_timestep` with a more
intuitive interface (vector `b` instead of a n-by-1 matrix `B`).

[^1]: Niesen, J., & Wright, W. (2009). A Krylov subspace algorithm for
evaluating the φ-functions in exponential integrators. arXiv preprint
arXiv:0907.4631.
"""
function expv_timestep(ts::Vector{tType}, A, b; kwargs...) where {tType <: Real}
    U = Matrix{eltype(b)}(undef, size(A, 1), length(ts))
    expv_timestep!(U, ts, A, b; kwargs...)
end
function expv_timestep(t::tType, A, b; kwargs...) where {tType <: Real}
    u = Vector{eltype(b)}(undef, size(A, 1))
    expv_timestep!(u, t, A, b; kwargs...)
end
"""
    expv_timestep!(u,t,A,b[;kwargs]) -> u

Non-allocating version of `expv_timestep`.
"""
function expv_timestep!(u::AbstractVector{T}, t::tType, A, b::AbstractVector{T};
                        kwargs...) where {T <: Number, tType <: Real}
    expv_timestep!(reshape(u, length(u), 1), [t], A, b; kwargs...)
    return u
end
function expv_timestep!(U::AbstractMatrix{T}, ts::Vector{tType}, A, b::AbstractVector{T};
                        kwargs...) where {T <: Number, tType <: Real}
    B = reshape(b, length(b), 1)
    phiv_timestep!(U, ts, A, B; kwargs...)
end
"""
    phiv_timestep(ts,A,B[;adaptive,tol,kwargs...]) -> U

Evaluates the linear combination of phi-vector products using time stepping

```math
u = \\varphi_0(tA)b_0 + t\\varphi_1(tA)b_1 + \\cdots + t^p\\varphi_p(tA)b_p
```

`ts` is an array of time snapshots for u, with `U[:,j] ≈ u(ts[j])`. `ts` can
also be just one value, in which case only the end result is returned and `U`
is a vector.

The time stepping formula of Niesen & Wright is used [^1]. If the time step
`tau` is not specified, it is chosen according to (17) of Neisen & Wright. If
`adaptive==true`, the time step and Krylov subsapce size adaptation scheme of
Niesen & Wright is used, the relative tolerance of which can be set using the
keyword parameter `tol`. The delta and gamma parameter of the adaptation
scheme can also be adjusted.

Set `verbose=true` to print out the internal steps (for debugging). For the
other keyword arguments, consult `arnoldi` and `phiv`, which are used
internally.

[^1]: Niesen, J., & Wright, W. (2009). A Krylov subspace algorithm for
evaluating the φ-functions in exponential integrators. arXiv preprint
arXiv:0907.4631.
"""
function phiv_timestep(ts::Vector{tType}, A, B; kwargs...) where {tType <: Real}
    U = Matrix{eltype(B)}(undef, size(A, 1), length(ts))
    phiv_timestep!(U, ts, A, B; kwargs...)
end
function phiv_timestep(t::tType, A, B; kwargs...) where {tType <: Real}
    u = Vector{eltype(B)}(undef, size(A, 1))
    phiv_timestep!(u, t, A, B; kwargs...)
end
"""
    phiv_timestep!(U,ts,A,B[;kwargs]) -> U

Non-allocating version of `phiv_timestep`.
"""
function phiv_timestep!(u::AbstractVector{T}, t::tType, A, B::AbstractMatrix{T};
                        kwargs...) where {T <: Number, tType <: Real}
    phiv_timestep!(reshape(u, length(u), 1), [t], A, B; kwargs...)
    return u
end
function phiv_timestep!(U::AbstractMatrix{T}, ts::Vector{tType}, A, B::AbstractMatrix{T}; tau::Real=0.0,
                        m::Int=min(10, size(A, 1)), tol::Real=1e-7, opnorm=LinearAlgebra.opnorm(A,Inf), iop::Int=0,
                        correct::Bool=false, caches=nothing, adaptive=false, delta::Real=1.2,
                        ishermitian::Bool=LinearAlgebra.ishermitian(A),
                        gamma::Real=0.8, NA::Int=0, verbose=false) where {T <: Number, tType <: Real}
    # Choose initial timestep
    opnorm = opnorm isa Number ? opnorm : opnorm(A,Inf) # backward compatibility
    abstol = tol * opnorm
    verbose && println("Absolute tolerance: $abstol")
    if iszero(tau)
        b0norm = norm(@view(B[:, 1]), Inf)
        tau = 10/opnorm * (abstol * ((m+1)/ℯ)^(m+1) * sqrt(2*pi*(m+1)) /
                          (4*opnorm*b0norm))^(1/m)
        verbose && println("Initial time step unspecified, chosen to be $tau")
    end
    # Initialization
    n = size(U, 1)
    sort!(ts); tend = ts[end]
    p = size(B, 2) - 1
    @assert length(ts) == size(U, 2) "Dimension mismatch"
    @assert n == size(A, 1) == size(A, 2) == size(B, 1) "Dimension mismatch"
    if caches == nothing
        u = Vector{T}(undef, n)              # stores the current state
        W = Matrix{T}(undef, n, p+1)         # stores the w vectors
        P = Matrix{T}(undef, n, p+2)         # stores output from phiv!
        Ks = KrylovSubspace{T}(n, m)  # stores output from arnoldi!
        phiv_cache = nothing         # cache used by phiv!
    else
        u, W, P, Ks, phiv_cache = caches
        @assert length(u) == n && size(W, 1) == n && size(P, 1) == n "Dimension mismatch"
        # W and P may be bigger than actually needed
        W = @view(W[:, 1:p+1])
        P = @view(P[:, 1:p+2])
    end
    copyto!(u, @view(B[:, 1])) # u(0) = b0
    coeffs = ones(tType, p);
    if adaptive # initialization step for the adaptive scheme
        if ishermitian
            iop = 2 # does not have an effect on arnoldi!, just for flops estimation
        end
        if iszero(NA)
            _A = convert(AbstractMatrix, A)
            if isa(_A, SparseMatrixCSC)
                NA = nnz(_A)
            else
                NA = count(!iszero, _A) # not constant operation, should be best avoided
            end
        end
    end

    t = 0.0       # current time
    snapshot = 1  # which snapshot to compute next
    while t < tend # time stepping loop
        if t + tau > tend # last step
            tau = tend - t
        end
        # Part 1: compute w0...wp using the recurrence relation (16)
        copyto!(@view(W[:, 1]), u) # w0 = u(t)
        @inbounds for l = 1:p-1 # compute cl = t^l/l!
            coeffs[l+1] = coeffs[l] * t / l
        end
        @views @inbounds for j = 1:p
            mul!(W[:, j+1], A, W[:, j])
            for l = 0:p-j
                axpy!(coeffs[l+1], B[:, j+l+1], W[:, j+1])
            end
        end
        # Part 2: compute ϕp(tau*A)wp using Krylov, possibly with adaptation
        arnoldi!(Ks, A, @view(W[:, end]); tol=tol, m=m, opnorm=opnorm, iop=iop)
        _, epsilon = phiv!(P, tau, Ks, p + 1; cache=phiv_cache, correct=correct, errest=true)
        verbose && println("t = $t, m = $m, tau = $tau, error estimate = $epsilon")
        if adaptive
            omega = (tend / tau) * (epsilon / abstol)
            epsilon_old = epsilon; m_old = m; tau_old = tau
            q = m/4; kappa = 2.0; maxtau = tend - t
            while omega > delta # inner loop of Algorithm 3
                m_new, tau_new, q, kappa = _phiv_timestep_adapt(
                    m, tau, epsilon, m_old, tau_old, epsilon_old, q, kappa,
                    gamma, omega, maxtau, n, p, NA, iop, LinearAlgebra.opnorm(getH(Ks), 1), verbose)
                m, m_old = m_new, m
                tau, tau_old = tau_new, tau
                # Compute ϕp(tau*A)wp using the new parameters
                arnoldi!(Ks, A, @view(W[:, end]); tol=tol, m=m, opnorm=opnorm, iop=iop)
                _, epsilon_new = phiv!(P, tau, Ks, p + 1; cache=phiv_cache, correct=correct, errest=true)
                epsilon, epsilon_old = epsilon_new, epsilon
                omega = (tend / tau) * (epsilon / abstol)
                verbose && println("  * m = $m, tau = $tau, error estimate = $epsilon")
            end
        end
        # Part 3: update u using (15)
        lmul!(tau^p, copyto!(u, @view(P[:, end - 1])))
        @inbounds for l = 1:p-1 # compute cl = tau^l/l!
            coeffs[l+1] = coeffs[l] * tau / l
        end
        @views @inbounds for j = 0:p-1
            axpy!(coeffs[j+1], W[:, j+1], u)
        end
        # Fill out all snapshots in between the current step
        while snapshot <= length(ts) && t + tau >= ts[snapshot]
            tau_snapshot = ts[snapshot] - t
            u_snapshot = @view(U[:, snapshot])
            phiv!(P, tau_snapshot, Ks, p + 1; cache=phiv_cache, correct=correct)
            lmul!(tau_snapshot^p, copyto!(u_snapshot, @view(P[:, end - 1])))
            @inbounds for l = 1:p-1 # compute cl = tau^l/l!
                coeffs[l+1] = coeffs[l] * tau_snapshot / l
            end
            @views @inbounds for j = 0:p-1
                axpy!(coeffs[j+1], W[:, j+1], u_snapshot)
            end
            snapshot += 1
        end

        t += tau
    end

return U
end
# Helper functions for phiv_timestep!
function _phiv_timestep_adapt(m, tau, epsilon, m_old, tau_old, epsilon_old, q, kappa,
                              gamma, omega, maxtau, n, p, NA, iop, Hnorm, verbose)
    # Compute new m and tau (Algorithm 4)
    if tau_old > tau
        q = log(tau/tau_old) / log(epsilon/epsilon_old) - 1
    end # else keep q the same
    tau_new = tau * (gamma / omega)^(1/(q + 1))
    tau_new = min(max(tau_new, tau/5), 2*tau, maxtau)
    if m_old < m
        kappa = (epsilon/epsilon_old)^(1/(m_old - m))
    end # else keep kappa the same
    m_new = m + ceil(Int, log(omega / gamma) / log(kappa))
    m_new = min(max(m_new, div(3*m, 4), 1), Int(ceil(4*m / 3)))
    verbose && println("  - Proposed new m: $m_new, new tau: $tau_new")
    # Compare costs of using new m vs new tau (23)
    cost_tau = _phiv_timestep_estimate_flops(m, tau_new, n, p, NA, iop, Hnorm, maxtau)
    cost_m = _phiv_timestep_estimate_flops(m_new, tau, n, p, NA, iop, Hnorm, maxtau)
    verbose && println("  - Cost to use new m: $cost_m flops, new tau: $cost_tau flops")
    if cost_tau < cost_m
        m_new = m
    else
        tau_new = tau
    end
    return m_new, tau_new, q, kappa
end
function _phiv_timestep_estimate_flops(m, tau, n, p, NA, iop, Hnorm, maxtau)
    # Estimate flops for the update of W and u
    flops_W = 2 * (p - 1) * (NA + n)
    flops_u = (2 * p + 1) * n
    # Estimate flops for arnoldi!
    if iop == 0
        iop = m
    end
    flops_matvec = 2 * m * NA
    flops_vecvec = 0
    for i = 1:m
        flops_vecvec += 3 * min(i, iop)
    end
    # Estimate flops for phiv! (7)
    MH = 44/3 + 2 * ceil(max(0.0, log2(Hnorm / 5.37)))
    flops_phiv = round(Int, MH * (m + p)^3)

    flops_onestep = flops_W + flops_u + flops_matvec + flops_vecvec + flops_phiv
    return flops_onestep * Int(ceil(maxtau / tau))
end
function _phiv_timestep_caches(u_prototype, maxiter::Int, p::Int)
    n = length(u_prototype); T = eltype(u_prototype)
    u = similar(u_prototype)                      # stores the current state
    W = Matrix{T}(undef, n, p+1)                  # stores the w vectors
    P = Matrix{T}(undef, n, p+2)                  # stores output from phiv!
    Ks = KrylovSubspace{T}(n, maxiter)            # stores output from arnoldi!
    phiv_cache = PhivCache{T}(maxiter, p+1)       # cache used by phiv! (need +1 for error estimation)
    return u, W, P, Ks, phiv_cache
end
