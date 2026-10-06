# Agent notes for ExponentialUtilities.jl

## Validation before pushing

- Format: `julia --project=@runic -m Runic --check <files>` (Runic.jl), spelling: `typos src ext test docs`.
- Tests: `GROUP=Core julia --project -e 'using Pkg; Pkg.test()'`; `GROUP=QA` for Aqua/ExplicitImports/JET; `GROUP=GPU` needs CUDA hardware. See `test/test_groups.toml`.
- Docs: `julia --project=docs docs/make.jl` for any docstring or public-API change.

## Temporary workarounds to remove

### Static Padé products bypass StaticArrays' kernel on Julia 1.12

The immutable-matrix `ExpMethodGeneric` path calls the hooks `_mul`, `_square` and `_horner`
(src/exp_generic.jl), whose defaults are the original `*`, `^` and `Base.evalpoly`. The
StaticArrays extension overrides them for non-float element types with a fully unrolled
`muladd` chain per product entry. This works around a Julia 1.12 / LLVM 18 SLP
vectorizer miscompile of StaticArrays' `mul_loop` that corrupts ForwardDiff partials, seen as
`ForwardDiff.jacobian(exponential!, ::SMatrix{4,4,Float32})` being wrong on AVX-512 CI runners
(the "exponential! on immutable (static) matrices" testset failing on about half of
`Core (julia 1, ubuntu-latest)` runs from 2026-08-01).

- Upstream: https://github.com/JuliaLang/julia/issues/62368, fixed in LLVM by
  llvm/llvm-project@5d7cf504, awaiting a 1.12 backport.
- Cost: the unrolled kernel is 1.7x (4×4 `Dual{Float32,16}`) to 4x (6×6 `Dual{Float64,36}`)
  slower than StaticArrays' `*`; end to end the 4×4 Float64 Jacobian is about 1.4x slower and
  the 6×6 about 2x. Plain-float matrices are unaffected and keep StaticArrays' kernel.
- Remove when: the package's minimum supported Julia (`[compat] julia` in Project.toml) is a
  release that contains the backport, i.e. the issue above is closed for 1.12.x or 1.12 is no
  longer supported. To remove, delete the three overrides in the extension, inline the hook
  defaults (`*`, `^`, `Base.evalpoly`) at their call sites in src/exp_generic.jl, and drop
  `:_mul`, `:_square`, `:_horner` from the ExplicitImports ignore lists in test/qa/qa.jl.
- To verify a Julia version is fixed without AVX-512 hardware, run the package-free reproducer
  under Intel SDE Sapphire Rapids emulation, or natively with dense random partials:
  a 4×4 `Dual{Nothing,Float32,16}` `SMatrix` product must match a Float64 reference to ~1e-7.
  `JULIA_LLVM_ARGS="-vectorize-slp=false"` making a failure disappear confirms it is this bug.
