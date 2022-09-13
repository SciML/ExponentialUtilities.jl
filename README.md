# ExponentialUtilities

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://exponentialutilities.sciml.ai/stable/)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/dev/modules/ExponentialUtilities/)

[![codecov](https://codecov.io/gh/SciML/ExponentialUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ExponentialUtilities.jl)
[![Build Status](https://github.com/SciML/ExponentialUtilities.jl/workflows/CI/badge.svg)](https://github.com/SciML/ExponentialUtilities.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

ExponentialUtilities is a package of utility functions for matrix functions of exponential type, including functionality
for the matrix exponential and phi-functions. These methods are more numerically stable, generic (thus support a wider 
range of number types), and faster than the matrix exponentiation tools in Julia's Base. The tools are used by the exponential 
integrators in OrdinaryDiffEq. The package has no external dependencies, so it can also be used independently.

## Tutorials and Documentation

For information on using the package,
[see the in-development documentation](https://exponentialutilities.sciml.ai/dev/).

## Example

```julia
using ExponentialUtilities

A = rand(2,2)
exponential!(A)

v = rand(2); t = rand()
expv(t,A,v)
```
