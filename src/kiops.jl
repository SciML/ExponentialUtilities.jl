using LinearAlgebra

"""
    kiops(tstops, A, u; kwargs...) -> (w, stats)

Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
vectors from ``u``, that is

```math
  w(i) = φ_0(t[i] A) u[:, 1] + φ_1(t[i] A) u[:, 2] + φ_2(t[i] A) u[:, 3] + ...
```

The size of the Krylov subspace is changed dynamically during the integration.
The Krylov subspace is computed using the incomplete orthogonalization method.

Arguments:
  - `tstops`     - Array of `tstop`
  - `A`          - the matrix argument of the ``φ`` functions
  - `u`          - the matrix with columns representing the vectors to be multiplied by the ``φ`` functions

Keyword arguments:
  - `tol`        - the convergence tolerance required (default: 1e-7)
  - `mmin`, `mmax` - let the Krylov size vary between mmin and mmax (default: 10, 128)
  - `m`          - an estimate of the appropriate Krylov size (default: mmin)
  - `iop`        - length of incomplete orthogonalization procedure (default: 2)
  - `ishermitian` -  whether ``A`` is Hermitian (default: ishermitian(A))
  - `opnorm`     -  the operator norm of ``A`` (default: opnorm(A, Inf))
  - `task1`      - if true, divide the result by 1/T^p

Returns:
  - `w`        - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
  - `stats[1]` - number of substeps
  - `stats[2]` - number of rejected steps
  - `stats[3]` - number of Krylov steps
  - `stats[4]` - number of matrix exponentials
  - `stats[5]` - the Krylov size of the last substep

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions

References:
* Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
* Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
* Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
"""
function kiops(tau_out, A, u; mmin::Int = 10, mmax::Int = 128, m::Int=min(mmin, mmax),
               tol::Real=1e-7, opnorm=LinearAlgebra.opnorm(A, Inf), iop::Int=2,
               ishermitian::Bool=LinearAlgebra.ishermitian(A), task1::Bool = false)
    n, ppo = size(u, 1), size(u, 2)
    p = ppo - 1

    if p == 0
        p = 1
        # Add extra column of zeros
        u = [u zero(u)]
    end

    # Preallocate matrix
    TA, Tb = eltype(A), eltype(u)
    T = promote_type(TA, Tb)
    Ks = KrylovSubspace{T, ishermitian ? real(T) : T}(n, m, p)

    step    = 0
    krystep = 0
    ireject = 0
    reject  = 0
    exps    = 0
    sgn     = sign(tau_out[end])
    tau_now = 0
    tau_end = abs(tau_out[end])
    j       = 0

    numSteps = size(tau_out, 2)

    # Initial condition
    w     = zeros(n, numSteps)
    w_aug = zeros(p)
    copyto!(@view(w[:, 1]), @view(u[:, 1]))

    # Normalization factors
    normU = norm(@view(u[:, 2:end]), 1)
    if ppo > 1 && normU > 0
        ex = ceil(log2(normU))
        nu = exp2(-ex)
        mu = exp2(ex)
    else
        nu = 1
        mu = 1
    end

    # Flip the rest of the u matrix
    u_flip = reverse(@view(u[:, 2:end]), dims = 2)
    rmul!(u_flip, nu)

    # Compute and initial starting approximation for the step size
    tau = tau_end

    # Setting the safety factors and tolerance requirements
    if tau_end > 1
        gamma = 0.2
        gamma_mmax = 0.1
    else
        gamma = 0.9
        gamma_mmax = 0.6
    end
    delta = 1.4

    # Used in the adaptive selection
    oldm = -1; oldtau = NaN; omega = NaN
    orderold = true; kestold = true

    l=1

    local beta, kest
    while tau_now < tau_end
        oldj = Ks.m
        arnoldi!(Ks, (A, u_flip), (w, w_aug); opnorm=opnorm, ishermitian=ishermitian, iop=iop, init=j, t=tau_now, mu=mu, l=l, m=m)
        V = getfield(Ks, :V)
        H = getfield(Ks, :H)
        j = Ks.m
        happy = j < oldj
        beta = Ks.beta

        # To obtain the phi_1 function which is needed for error estimate
        H[1, j + 1] = 1

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j + 1, j]
        H[j + 1, j] = 0

        # Compute the exponential of the augmented matrix
        F = exp(sgn * tau * H[1:j + 1, 1:j + 1])
        exps = exps + 1

        # Restore the value of H_{m+1,m}
        H[j + 1, j] = nrm

        if happy
            # Happy breakdown wrap up
            omega   = 0
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new   = m
            happy   = false
        else
            # Local truncation error estimation
            err = abs(beta * nrm * F[j, j + 1])

            # Error for this step
            oldomega = omega
            omega = tau_end * err / (tau * tol)

            # Estimate order
            if m == oldm && tau != oldtau && ireject >= 1
                order = max(1, log(omega/oldomega) / log(tau/oldtau))
                orderold = false
            elseif orderold || ireject == 0
                orderold = true
                order = j/4
            else
                orderold = true
            end
            # Estimate k
            if m != oldm && tau == oldtau && ireject >= 1
                kest = max(1.1, (omega/oldomega)^(1/(oldm-m)))
                kestold = false
            elseif kestold || ireject == 0
                kestold = true
                kest = 2
            else
                kestold = true
            end

            if omega > delta
                remaining_time = tau_end - tau_now
            else
                remaining_time = tau_end - (tau_now + tau)
            end

            # Krylov adaptivity

            same_tau = min(remaining_time, tau)
            tau_opt  = tau * (gamma / omega)^(1 / order)
            tau_opt  = min(remaining_time, max(tau/5, min(5*tau, tau_opt)))

            m_opt = ceil(Int, j + log(omega / gamma) / log(kest))
            m_opt = max(mmin, min(mmax, max(3÷4*m, min(m_opt, cld(4, 3)*m))))

            if j == mmax
                if omega > delta
                    m_new = j
                    tau_new = tau * (gamma_mmax / omega)^(1 / order)
                    tau_new = min(tau_end - tau_now, max(tau/5, tau_new))
                else
                    tau_new = tau_opt
                    m_new = m
                end
            else
                m_new = m_opt
                tau_new = same_tau
            end
        end

        # Check error against target
        if omega <= delta
            tau_now, l, j, reject, ireject, step = kiops_update_solution!(tau_now, tau, tau_out, w, l, V, F, H, beta, j, n, step, numSteps, reject, ireject)
        else
            # Nope, try again
            ireject = ireject + 1

            # Restore the original matrix
            H[1, j + 1] = 0
        end

        oldtau = tau
        tau    = tau_new

        oldm = m
        m    = m_new
    end

    # FIXME: phiHan doesn't seem right
    if tau_out[1]!=1 && task1
        wl = @view w[:, l]
        if length(tau_out)==1
            @. wl = wl*(1/tau_out[l])^p
        else
            tmp = maximum(abs, u, dims=2)
            phiHan = map(first, findall(!iszero, tmp))

            if isempty(phiHan)
                phiHan = length(u)
            else
                phiHan = phiHan .- 1
            end

            for l = 1:numSteps
                @. wl = wl*(1/tau_out[l]^phiHan)
            end
        end
    end

    m_ret=m

    stats = (step, reject, krystep, exps, m_ret)

    return w, stats
end

Base.@propagate_inbounds function kiops_update_solution!(tau_now, tau, tau_out, w, l, V, F, H, beta, j, n, step, numSteps, reject, ireject)
    # Yep, got the required tolerance update
    reject = reject + ireject
    step = step + 1

    # Udate for tau_out in the interval (tau_now, tau_now + tau)
    blownTs = 0
    nextT = tau_now + tau
    for k = l:numSteps
        if abs(tau_out[k]) < abs(nextT)
            blownTs = blownTs + 1
        end
    end

    if blownTs != 0
        # Copy current w to w we continue with.
        copyto!(@view(w[:, l+blownTs]), @view(w[:, l]))

        for k = 0:blownTs - 1
            tauPhantom = tau_out[l+k] - tau_now
            F2 = exp(sign(tau_out[end]) * tauPhantom * @view(H[1:j, 1:j]))
            mul!(@view(w[:, l+k]), @view(V[1:n, 1:j]), @view(F2[1:j, 1]))
            rmul!(@view(w[:, l+k]), beta)
        end

        # Advance l.
        l = l + blownTs
    end

    # Using the standard scheme
    mul!(@view(w[:, l]), @view(V[1:n, 1:j]), @view(F[1:j, 1]))
    rmul!(@view(w[:, l]), beta)

    # Update tau_out
    tau_now = tau_now + tau

    j = 0
    ireject = 0
    return tau_now, l, j, reject, ireject, step
end
