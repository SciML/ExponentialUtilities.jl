# Expv: Matrix Exponentials Times Vectors

The main functionality of ExponentialUtilities is the computation of matrix-phi-vector products. The phi functions are defined as

```
ϕ_0(z) = exp(z)
ϕ_(k+1)(z) = (ϕ_k(z) - 1) / z
```

In exponential algorithms, products in the form of `ϕ_m(tA)b` is frequently encountered. Instead of computing the matrix function first and then computing the matrix-vector product, the common alternative is to construct a [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) `K_m(A,b)` and then approximate the matrix-phi-vector product.

### Support for matrix-free operators

You can use any object as the "matrix" `A` as long as it implements the following linear operator interface:

  - `Base.eltype(A)`
  - `Base.size(A, dim)`
  - `LinearAlgebra.mul!(y, A, x)` (for computing `y = A * x` in place).
  - `LinearAlgebra.opnorm(A, p=Inf)`. If this is not implemented or the default implementation can be slow, you can manually pass in the operator norm (a rough estimate is fine) using the keyword argument `opnorm`.
  - `LinearAlgebra.ishermitian(A)`. If this is not implemented or the default implementation can be slow, you can manually pass in the value using the keyword argument `ishermitian`.

## Core API

```@docs
expv
phiv
expv!
phiv!
exp_timestep
phiv_timestep
exp_timestep!
phiv_timestep!
phi
```

## Caches

```@docs
ExpvCache
PhivCache
```
