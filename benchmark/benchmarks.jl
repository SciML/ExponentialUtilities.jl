using ExponentialUtilities, BenchmarkTools
using LinearAlgebra, SparseArrays, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

N = 100
A = rand(rng, N, N)
b = rand(rng, N)
As = sprand(rng, 500, 500, 0.01)
bs = rand(rng, 500)

# =============================================================================
# Dense matrix exponential
# =============================================================================

SUITE["expm"] = BenchmarkGroup()

SUITE["expm"]["dense"] = @benchmarkable exponential!(A) setup = (A = copy($A))
SUITE["expm"]["higham2005"] = @benchmarkable exponential!(
    A, ExpMethodHigham2005()
) setup = (A = copy($A))
SUITE["expm"]["alloc_mem"] = @benchmarkable ExponentialUtilities.alloc_mem(
    $A, ExpMethodHigham2005()
)

# =============================================================================
# Krylov subspace methods
# =============================================================================

SUITE["krylov"] = BenchmarkGroup()

SUITE["krylov"]["arnoldi"] = @benchmarkable arnoldi($As, $bs)
SUITE["krylov"]["expv_dense"] = @benchmarkable expv(0.5, $A, $b)
SUITE["krylov"]["expv_sparse"] = @benchmarkable expv(0.5, $As, $bs)
SUITE["krylov"]["expv_timestep"] = @benchmarkable expv_timestep(
    0.5, $As, $bs; adaptive = true
)

# =============================================================================
# phi functions
# =============================================================================

SUITE["phi"] = BenchmarkGroup()

SUITE["phi"]["phiv"] = @benchmarkable phiv(0.5, $As, $bs, 3)
SUITE["phi"]["phi_dense"] = @benchmarkable phi($(A * 0.1), 3)
