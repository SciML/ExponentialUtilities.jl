# Internal API

These names are not exported and carry no stability guarantee. They are
documented here so that every docstring in the package appears in the manual;
prefer the exported entry points on the other pages.

## Arnoldi and Lanczos steps

```@docs
ExponentialUtilities.firststep!
ExponentialUtilities.arnoldi_step!
ExponentialUtilities.lanczos_step!
ExponentialUtilities.coeff
ExponentialUtilities.realview
ExponentialUtilities.@diagview
```

## Dense and subspace phi evaluation

```@docs
ExponentialUtilities.phiv_dense
ExponentialUtilities.phiv_dense!
ExponentialUtilities._phi_almohy!
ExponentialUtilities._phi_almohy_generic
ExponentialUtilities.symtridiag_eigen!
ExponentialUtilities.expT!
```

## Caches and method selection

```@docs
ExponentialUtilities.get_expcache!
ExponentialUtilities.pade_order_for_type
```
