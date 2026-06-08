# API

```@meta
CurrentModule = FactorVSA
```

The public surface, grouped by build step. The [Guide](guide.md) shows these in use with
runnable examples; this page is the reference.

```@index
```

## Operators extending Base / LinearAlgebra

`bind` extends `Base.bind` and `factorize` extends `LinearAlgebra.factorize`, so they are
documented here by signature (the rest of the surface is auto-collected below).

```@docs
bind(::HV{BipolarMAP}, ::HV{BipolarMAP})
factorize(::HV{BipolarMAP}, ::Vector{Codebook{BipolarMAP}})
```

## Everything else

```@autodocs
Modules = [FactorVSA]
Private = false
Order   = [:type, :function]
```
