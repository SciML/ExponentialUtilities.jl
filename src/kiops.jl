using LinearAlgebra

"""
    kiops(tstops, A, u; kwargs...) -> (w, m, stats)

Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
vectors from ``u``, that is

```math
  w(i) = φ_0(t[i] A) u[:, 1] + φ_1(t[i] A) u[:, 2] + φ_2(t[i] A) u[:, 3] + ...
```

The size of the Krylov subspace is changed dynamically during the integration.
The Krylov subspace is computed using the incomplete orthogonalization method.

Arguments:
  - `tstops`     - Array of `tstop`
  - `A`          - the matrix argument of the ``φ`` functions.
  - `u`          - the matrix with columns representing the vectors to be multiplied by the ``φ`` functions.

Keyword arguments:
  - `tol`        - the convergence tolerance required.
  - `m_init`     - an estimate of the appropriate Krylov size.
  - `mmin`, `mmax` - let the Krylov size vary between mmin and mmax
  - `task1`      - If true, divide the result by 1/T^p

Returns:
  - `w`        - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``.
  - `m`        - the Krylov size of the last substep.
  - `stats[1]` - number of substeps
  - `stats[2]` - number of rejected steps
  - `stats[3]` - number of Krylov steps
  - `stats[4]` - number of matrix exponentials

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions

References:
* Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
* Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
* Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
"""
function kiops(tau_out, A, u; tol = 1.0e-7, mmin::Int = 10, mmax::Int = 128, m_init::Int = mmin, task1::Bool = false)

    n, ppo = size(u, 1), size(u, 2)
    p = ppo - 1

    if p == 0
        p = 1
        # Add extra column of zeros
        u = [u zero(u)]
    end

    # Krylov parameters
    orth_len = 2

    # We only allow m to vary between mmin and mmax
    m = max(mmin, min(m_init, mmax))

    # Preallocate matrix
    V = zeros(n + p, mmax + 1)
    H = zeros(mmax + 1, mmax + 1)

    step    = 0
    krystep = 0
    ireject = 0
    reject  = 0
    exps    = 0
    sgn     = sign(tau_out[end])
    tau_now = 0
    tau_end = abs(tau_out[end])
    happy   = false
    j       = 0

    numSteps = size(tau_out, 2)

    # Initial condition
    w     = zeros(n, numSteps)
    w_aug = zeros(p, 1)
    @views copyto!(w[:, 1], u[:, 1])

    # Normalization factors
    normU = norm(u[:, 2:end], 1)
    if ppo > 1 && normU > 0
        ex = ceil(log2(normU))
        nu = 2^(-ex)
        mu = 2^(ex)
    else
        nu = 1
        mu = 1
    end

    # Flip the rest of the u matrix
    u_flip = nu*reverse(u[:, 2:end], dims = 2)

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
    oldm = NaN; oldtau = NaN; omega = NaN
    orderold = true; kestold = true

    l=1

    local beta, kest
    while tau_now < tau_end

        # Compute necessary starting information
        if j == 0

            # Update the last part of w
            for k=1:p-1
                i = p - k
                w_aug[k] = (tau_now^i)/factorial(i) * mu
            end
            w_aug[p] = mu

            # Initialize the matrices V and H
            fill!(H, 0)

            # Normalize initial vector (this norm is nonzero)
            beta = sqrt( w[:,l]'w[:,l] .+ dot(w_aug, w_aug) )

            # The first Krylov basis vector
            V[1:n, 1]     = (1/beta) * w[:,l]
            V[n+1:n+p, 1] = (1/beta) * w_aug
        end

        # Incomplete orthogonalization process
        while j < m
            j += 1

            # Augmented matrix - vector product
            V[1:n      , j + 1] = A * V[1:n, j] + u_flip * V[n+1:n+p, j]
            V[n+1:n+p-1, j + 1] = V[n+2:n+p, j]
            V[end      , j + 1] = 0

            # Modified Gram-Schmidt
            for i = max(1,j-orth_len+1):j
                H[i, j] = V[:, i]' * V[:, j + 1]
                V[:, j + 1] = V[:, j + 1] - H[i, j] * V[:, i]
            end

            nrm = norm(V[:, j + 1])

            # Happy breakdown
            if nrm < tol
                happy = true
                break
            end

            H[j + 1, j] = nrm
            V[:, j + 1] = (1/nrm) * V[:, j + 1]

            krystep = krystep + 1
        end

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
            happy   = false
            m_new   = m
            tau_new = min(tau_end - (tau_now + tau), tau)

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

            m_opt = ceil(j + log(omega / gamma) / log(kest))
            m_opt = max(mmin, min(mmax, max(floor(3/4*m), min(m_opt, ceil(4/3*m)))))

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
                w[:,l + blownTs] = w[:,l]

                for k = 0:blownTs - 1
                    tauPhantom = tau_out[l+k] - tau_now
                    F2 = exp(sgn * tauPhantom * H[1:j, 1:j])
                    w[:, l+k] = beta * V[1:n, 1:j] * F2[1:j, 1]
                end

                # Advance l.
                l = l + blownTs
            end

            # Using the standard scheme
            w[:, l] = beta * V[1:n, 1:j] * F[1:j, 1]

            # Update tau_out
            tau_now = tau_now + tau

            j = 0
            ireject = 0

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

    if tau_out[1]!=1 && task1
        if length(tau_out)==1
            w[:,l] = w[:,l]*(1/tau_out[l])^p
        else
            tmp = maximum(abs, u, dims=2)
            phiHan = map(x->x[1], findall(!iszero, tmp))

            if isempty(phiHan)
                phiHan = length(u)
            else
                phiHan = phiHan .- 1
            end

            for l = 1:numSteps
                @. w[:,l] = w[:,l]*(1/tau_out[l]^phiHan)
            end
        end
    end

    m_ret=m

    stats = [step, reject, krystep, exps]

    return w, m_ret, stats
end
