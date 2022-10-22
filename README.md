# ExponentialUtilities

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/ExponentialUtilities/stable/)

[![codecov](https://codecov.io/gh/SciML/ExponentialUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ExponentialUtilities.jl)
[![Build Status](https://github.com/SciML/ExponentialUtilities.jl/workflows/CI/badge.svg)](https://github.com/SciML/ExponentialUtilities.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/7c96b830f694a59b4171d8c20af570381bd557ff1acc1e23f1.svg)](https://buildkite.com/julialang/exponentialutilities-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

ExponentialUtilities is a package of utility functions for matrix functions of exponential type, including functionality
for the matrix exponential and phi-functions. These methods are more numerically stable, generic (thus support a wider 
range of number types), and faster than the matrix exponentiation tools in Julia's Base. The tools are used by the exponential 
integrators in OrdinaryDiffEq. The package has no external dependencies, so it can also be used independently.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://exponentialutilities.sciml.ai/stable/). Use the
[in-development documentation](https://exponentialutilities.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.

## Example

```julia
using ExponentialUtilities

A = rand(2,2)
exponential!(A)

v = rand(2); t = rand()
expv(t,A,v)
```
