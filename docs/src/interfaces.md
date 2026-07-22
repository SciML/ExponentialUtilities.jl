# Matrix-Free Operator Interface

The Krylov-vector APIs accept matrix-free linear operators. This is an
extension interface for callers implementing their own operator type; it is
not necessary to subtype an ExponentialUtilities type.

## Required operations

For an operator `A` and vector `x`, implement these generic Julia and
LinearAlgebra interfaces in the module that owns the operator type:

  - `Base.eltype(A)`, returning the scalar type produced by `A`.
  - `Base.size(A, dim)`, with a square operator shape. `size(A, 1)` must equal
    the length of vectors accepted by `A`.
  - `LinearAlgebra.mul!(y, A, x)`, overwriting `y` with `A * x`. It must accept
    vectors with the promoted element type used by the Krylov basis and return
    `y`.

The same operator can be passed to `arnoldi`, `expv`, `phiv`,
`expv_timestep`, and `phiv_timestep`. The methods do not require indexing or
materializing a matrix.

## Optional operations

`LinearAlgebra.ishermitian(A)` selects the Lanczos path when it returns `true`.
Only report `true` when the operator is Hermitian for the scalar product used
by `mul!`; an incorrect answer invalidates the approximation.

`LinearAlgebra.opnorm(A, p)` is optional. `arnoldi`, `expv`, and `phiv` accept
an `opnorm` keyword instead, which may be a scalar bound or callable
`opnorm(A, p)`. The adaptive timestep APIs estimate their default scale from
the Arnoldi Hessenberg and therefore do not call `opnorm(A, Inf)` unless an
`opnorm` override is supplied.

## Example

```@example matrixfree
using ExponentialUtilities, LinearAlgebra

module MatrixFreeExample
using LinearAlgebra
export Operator

struct Operator{T, M <: AbstractMatrix{T}}
    data::M
end

Base.size(A::Operator, dim::Integer) = size(A.data, dim)
Base.eltype(::Operator{T}) where {T} = T
LinearAlgebra.mul!(y, A::Operator, x) = mul!(y, A.data, x)
LinearAlgebra.ishermitian(A::Operator) = ishermitian(A.data)
end

A = MatrixFreeExample.Operator([-1.0 1.0; 0.0 -2.0])
b = [1.0, 0.0]
expv(0.1, A, b; m = 2)
```
