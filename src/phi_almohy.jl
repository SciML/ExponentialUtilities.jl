# Scaling-and-recovering algorithm for the matrix phi-functions, computing
# phi_0(A), ..., phi_p(A) simultaneously in O(p*n^3) work.
#
# This is a port of Algorithm 5.1 of
#
#   Awad H. Al-Mohy and Xiaobo Liu, "A scaling and recovering algorithm for the
#   matrix phi-functions", arXiv:2506.01193 (2025),
#
# and follows the authors' reference MATLAB implementation `phi_funm`.
#
# The idea: scale A down by 2^s, evaluate the [m/m] diagonal Pade approximant to
# phi_p of the scaled matrix, recover the lower-index approximants via the
# recurrence R^{(j)}_m(z) = z R^{(j+1)}_m(z) + 1/j!, then undo the scaling with
# the double-argument formula
#
#   phi_j(2A) = 2^{-j} ( phi_0(A) phi_j(A) + sum_{k=1}^{j} phi_k(A)/(j-k)! ).

# theta[m, p] = theta_{m,p} from Table 3.1 of the paper (largest 1-norm of the
# scaled matrix for which the [m/m] Pade approximant is backward stable to
# double-precision unit roundoff). Rows are m = 1:20, columns p = 1:10.
const _PHI_THETA_MP = [
    1.999463452408407e-5 3.7631213142601604e-5 7.366006416045163e-5 0.00014973317297025854 0.0003152443333771182 0.0006855209983764435 0.0015357294906993542 0.0035357368946407606 0.008345789028062234 0.02013882808928226;
    0.0038062018282832713 0.006090206286125726 0.009869682746779615 0.016211831146383013 0.026984240563843326 0.0454757968381803 0.07749273259855331 0.13324895779231027 0.2304625604362521 0.3988991104549146;
    0.03971636005661334 0.05806968886880692 0.08534220076759817 0.12612151517169362 0.18736524835661617 0.27955116495245524 0.41822735418681667 0.6258702800351991 0.9335970443986562 1.1616793320890249;
    0.15442675548312682 0.21278117034577634 0.29371996708854947 0.40617647304246707 0.5623843002320996 0.7787883505754265 1.0464245027100287 1.2572921799132364 1.480350861451984 1.713871003185325;
    0.37980898016147974 0.5014624976587007 0.6621033824456818 0.8739858777744354 1.110828442901451 1.3375805277521873 1.5770955744229305 1.8273848586096524 2.08673114576671 2.3536689190936406;
    0.726177195703321 0.9281910159646274 1.1591052927815408 1.4012982671152012 1.6570386251184455 1.924026602416338 2.20029216725632 2.4841706972729956 2.7742685942850143 3.069426294589578;
    1.1898666361923196 1.4469917284172122 1.7187282676344413 2.0024009817133357 2.295729523138923 2.5968019781751672 2.9040336511997444 3.216121483847106 3.531999855119243 3.8508005672569903;
    1.7605812331512907 2.060907194742016 2.371480030315257 2.690078917436836 3.014877598333601 3.3443898815994957 3.677415494646134 4.01299056190377 4.350344448873932 4.688863279363638;
    2.425818958547233 2.7623053687512256 3.105174730511504 3.4527057639100476 3.8035234182863555 4.156538184296475 4.510892686686127 4.865916398054041 5.2210881567935115 5.57600564043454;
    3.173113456793749 3.5393251225873685 3.9086764034913664 4.279911937822335 4.652058062541427 5.024366624424222 5.396268333440519 5.767334816025469 6.1372481669904 6.505776744744432;
    3.991025201329815 4.3813599805888455 4.772204432206606 5.162703837030561 5.552220615863406 5.940286915538899 6.326566605567779 6.710825187972159 7.092906173082131 7.4727126397309815;
    4.869485489784578 5.279199870248916 5.687344501175957 6.09339280009268 6.496977936247796 6.8978550856394785 7.295871912058635 7.690945666690415 8.083045525424248 8.472179024890941;
    5.799827904547885 6.224971321619904 6.64693480650714 7.065451733211646 7.48036740469753 7.891610269499859 8.29916985381997 8.703079946679036 9.103405852177376 9.500234768451538;
    6.774681996458441 7.211997240187581 7.644910128682907 8.073354767800579 8.497338949398504 8.916922998846443 9.332203860678709 9.74330319935391 10.150358556572542 10.5535168239286;
    7.787817051572435 8.234634975085694 8.676142196766529 9.112424389503664 9.543611219824884 9.969861373150518 10.391351525913848 10.808268293369911 11.220802409044003 11.629144570605774;
    8.833978214833847 9.28811956766922 9.736292792866143 10.178695474892383 10.615546677548995 11.04707680323746 11.473520338548882 11.895110742359616 12.312076916038997 12.724640835984562;
    9.908733823133844 10.368423129623023 10.821685340709736 11.26879859255289 11.710045790427412 12.145708155762449 12.576060822320501 13.001369927621694 13.421890788016032 13.837866852904227;
    11.008341035221061 11.472133539821815 11.929195701618417 12.379861767254168 12.824458619109002 13.263302077220441 13.696694607678852 14.124924034990169 14.548262963142152 14.966968689718191;
    12.129631147167878 12.596352058386916 13.056160721729096 13.509428665638305 13.956511465304834 14.397747062985934 14.83345501195776 15.263936358537684 15.689473955602649 16.110333058994453;
    13.269913405576162 13.738607964511967 14.20030229038679 14.655390835644507 15.104246216977073 15.547218984875734 15.984637960610213 16.41681094336813 16.84402564730148 17.266550769620974;
]

# theta_{m,p} with the p>7 rule of the paper: for p>7 the p=7 column is used, and
# indices are guarded so m outside 1:20 does not error.
@inline function _phi_theta(m::Integer, p::Integer)
    return _PHI_THETA_MP[m, p > 7 ? 7 : p]
end

# k! as a Float64 (k small; avoids Int overflow for the recovery coefficients).
@inline function _factf(k::Integer)
    r = 1.0
    for i in 2:k
        r *= i
    end
    return r
end

# Exact 1-norm of B^k for an entrywise-nonnegative real B, via ‖B^k‖_1 =
# ‖(B^T)^k 𝟙‖_∞ (k matrix-vector products, no explicit power).
function _normpow_nonneg(B::AbstractMatrix{<:Real}, k::Integer)
    n = size(B, 1)
    Bt = transpose(B)
    v = ones(eltype(B), n)
    tmp = similar(v)
    for _ in 1:k
        mul!(tmp, Bt, v)
        v, tmp = tmp, v
    end
    return maximum(v)
end

# Scaling parameter `t` derived from the first term of the backward-error series
# (Eq. (3.12)); mirrors the `ell` subfunction of the reference implementation.
function _phi_ell(A::AbstractMatrix, m::Integer, p::Integer, phat::Integer)
    normT = opnorm(A, 1)
    normT == 0 && return 0
    t0 = normT > 1 ? log2(normT) : 0.0
    scalefac = exp2(t0)
    normTs = normT / scalefac
    delta = (p - 1) * (p - phat) / p + 1
    coeff = (_factf(m + p) / _factf(2m + p)) * (_factf(m) / _factf(2m + p + 1))
    K = 2m + p + 1
    c = (coeff / normTs^delta)^(1 / K)
    scaledT = c .* abs.(A ./ scalefac)
    alpha = _normpow_nonneg(scaledT, K)
    t = log2(2alpha / eps(Float64)) / (K - delta) + t0
    return max(ceil(Int, t), 0)
end

# Select the Pade degree m, scaling parameter s, and Paterson-Stockmeyer block
# size tau that minimize the equivalent number of matrix products (Section 4).
function _select_parameters_phi(A::AbstractMatrix, p::Integer)
    m_max = 12
    i_max = ceil(Int, sqrt(8 * (m_max + 1)) - 3) - 1
    m_max = (i_max + 3)^2 ÷ 8
    phat = _phi_theta(m_max, p) < 1 ? 0 : p
    r_max = floor(Int, (1 + sqrt(1 + 4 * (2m_max + phat + 1))) / 2)

    # eta[j] = ‖A^{j+1}‖_1^{1/(j+1)} for j = 1:r_max, forming the powers exactly.
    # (The powers are low, j+1 ≤ r_max+1 ≲ 8; the cost is negligible next to the
    # rest of the algorithm and never underestimates, keeping s conservative.)
    eta = zeros(Float64, r_max)
    P = A * A
    eta[1] = opnorm(P, 1)^(1 / 2)
    for j in 2:r_max
        P = P * A
        eta[j] = opnorm(P, 1)^(1 / (j + 1))
    end
    alpha = [max(eta[j], eta[j + 1]) for j in 1:(r_max - 1)]

    Cost = zeros(Float64, i_max + 1, r_max - 1)
    for i in 0:i_max
        m_i = (i + 3)^2 ÷ 8
        θ = _phi_theta(m_i, p)
        phat_i = θ < 1 ? 0 : p
        t = _phi_ell(A, m_i, p, phat_i)
        for r in 2:r_max
            if 2m_i + phat_i + 1 >= r * (r - 1)
                a = alpha[r - 1]
                s0 = (a > 0 && isfinite(a)) ? max(ceil(Int, log2(a / θ)), t) : t
                Cost[i + 1, r - 1] = i + p + s0 * (p + 1)
            end
        end
    end

    minval = Inf
    for v in Cost
        if v > 0 && v < minval
            minval = v
        end
    end
    # Match the reference's column-major "last" tie-break for reproducibility.
    idx = 0
    for (j, v) in enumerate(Cost)
        v == minval && (idx = j)
    end
    i_star = (idx - 1) % (i_max + 1)
    m = (i_star + 3)^2 ÷ 8
    s = round(Int, (minval - i_star - p) / (p + 1))

    tau = floor(Int, sqrt(2m))
    if tau - 1 + 2 * (m ÷ tau) - 2 * (m % tau == 0) != i_star
        tau = ceil(Int, sqrt(2m))
    end
    return m, s, tau
end

# Renormalized [m/m] Pade coefficients (numerator and denominator, low order
# first) of the approximant to phi_p. Follows the recurrences of Berland,
# Skaflestad, and Wright used in the reference `pade_coef`.
function _phi_pade_coef(m::Integer, p::Integer)
    n1 = prod(Float64.((m + 1):(2m + 1)))   # (2m+1)!/m!
    d1 = n1
    Ncoef = Vector{Float64}(undef, m + 1)
    Dcoef = Vector{Float64}(undef, m + 1)
    Amat = Matrix{Float64}(undef, m + 1, m + 1)
    for k in 1:p
        if k == p
            # First column: Amat[r,1] = n1 * prod_{l=1}^{r-1} 1/(k+l).
            cp = 1.0
            Amat[1, 1] = n1
            for r in 2:(m + 1)
                cp /= (k + (r - 1))
                Amat[r, 1] = n1 * cp
            end
            for c in 2:(m + 1), r in 2:(m + 1)
                I = r - 1
                J = c - 1
                Amat[r, c] = -(m + 1 - J) * (k + 1 + I - J) / ((2m + k + 1 - J) * J)
            end
            for r in 1:(m + 1)
                s = 0.0
                pr = 1.0
                for c in 1:r
                    pr *= Amat[r, c]
                    s += pr
                end
                Ncoef[r] = s
            end
            Dcoef[1] = d1
            cpd = 1.0
            for r in 2:(m + 1)
                i = r - 1
                cpd *= -(m + 1 - i) / (i * (2m + k + 1 - i))
                Dcoef[r] = d1 * cpd
            end
        end
        n1 = n1 * (2m + k + 1) / (k + 1)
        d1 = d1 * (2m + k + 1)
    end
    return Ncoef, Dcoef
end

# Simultaneously evaluate the numerator N(A) and denominator D(A) of the Pade
# approximant via the Paterson-Stockmeyer scheme with block size tau.
function _paterson_stockmeyer(A::AbstractMatrix{T}, Nc, Dc, tau::Integer) where {T}
    m = length(Nc) - 1
    n = size(A, 1)
    Apow = Vector{Matrix{T}}(undef, tau + 1)
    Apow[1] = Matrix{T}(I, n, n)
    Apow[2] = Matrix(A)
    for i in 3:(tau + 1)
        Apow[i] = Apow[i - 1] * A
    end
    Atau = copy(Apow[tau + 1])
    N = zeros(T, n, n)
    D = zeros(T, n, n)
    nu = m ÷ tau
    for i in 0:nu
        start = i * tau + 1
        stop = min((i + 1) * tau, m + 1)
        blkN = zeros(T, n, n)
        blkD = zeros(T, n, n)
        for l in 1:(stop - start + 1)
            blkN .+= Nc[start + l - 1] .* Apow[l]
            blkD .+= Dc[start + l - 1] .* Apow[l]
        end
        if i == 0
            N .+= blkN
            D .+= blkD
        else
            N .+= blkN * Atau
            D .+= blkD * Atau
            Atau = Atau * Apow[tau + 1]
        end
    end
    return N, D
end

"""
    _phi_almohy!(out, A, p) -> out

Simultaneously compute `phi_0(A), phi_1(A), ..., phi_p(A)` for a dense matrix `A`
(`p >= 1`) using the scaling-and-recovering algorithm of Al-Mohy and Liu
(arXiv:2506.01193). `out` must be a length-`p+1` vector of `size(A)` matrices; on
return `out[j+1] == phi_j(A)`.

The cost is `O(p n^3)`, in contrast to the `O(n (n+p)^3)` of the basis-vector
approach in [`phi!`](@ref).
"""
function _phi_almohy!(
        out::AbstractVector{<:AbstractMatrix}, A::AbstractMatrix{T},
        p::Integer
    ) where {T}
    n = size(A, 1)
    m, s, tau = _select_parameters_phi(A, p)
    As = A ./ exp2(s)
    Nc, Dc = _phi_pade_coef(m, p)
    Nm, Dm = _paterson_stockmeyer(As, Nc, Dc, tau)

    Rm = Vector{Matrix{T}}(undef, p + 1)
    Rm[p + 1] = Dm \ Nm
    # Recurrence (2.9): R^{(j)} = As R^{(j+1)} + I/j!, j = p-1 : -1 : 0.
    for k in p:-1:1
        R = As * Rm[k + 1]
        f = 1 / _factf(k - 1)
        @inbounds for d in 1:n
            R[d, d] += f
        end
        Rm[k] = R
    end

    # Undo the scaling with the double-argument formula (2.10), s times.
    for _ in 1:s
        for j in p:-1:1
            M = Rm[1] * Rm[j + 1]
            for k in 1:j
                M .+= (1 / _factf(j - k)) .* Rm[k + 1]
            end
            M ./= exp2(j)
            Rm[j + 1] = M
        end
        Rm[1] = Rm[1] * Rm[1]
    end

    for j in 0:p
        copyto!(out[j + 1], Rm[j + 1])
    end
    return out
end
