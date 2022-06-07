# ExponentialUtilities

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/ExponentialUtilities.jl/workflows/CI/badge.svg)](https://github.com/SciML/ExponentialUtilities.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/SciML/ExponentialUtilities.jl/badge.svg?branch=master)](https://coveralls.io/github/SciML/ExponentialUtilities.jl?branch=master)
[![codecov](https://codecov.io/gh/SciML/ExponentialUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ExponentialUtilities.jl)

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