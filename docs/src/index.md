```@meta
CurrentModule = FactorVSA
```

# FactorVSA.jl

Resonator-factored **Vector-Symbolic Architecture** over fixed-width hypervectors —
the vector/hypervector substrate leg (sibling to
[MORKTensorNetworks](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks), the
sparse-tensor/einsum leg). It implements the VSA algebra and resonator factorization of
Goertzel (2026), *"Resonator-Factored Hierarchical Hypervector Embeddings"*. Depends on
[PathMap](https://github.com/CognitiveSubstratesAI/PathMap) +
[MORK](https://github.com/CognitiveSubstratesAI/MORK) only.

It is a **query-faithful cache** of an already-learned factor graph — *not* lossless
storage. The dimension `D` for reliable local queries scales with local crosstalk load
`k` and the log of codebook size, **not** raw object size.

## Status

Steps 0–4 are implemented for the `BipolarMAP` backend, and the **Step-4 margin gate
passes** (see `GATE_RESULT.md` and `bench/gate.jl`):

- **E1** — single-step cleanup tracks Lemma 2's `ℙ[fail] ≤ 2M·exp(−cD/k)` (fitted `c ≈ 0.21`).
- **E4** — resonator success rises 0.26 → 0.985 over `D = 128 … 4096` (3 factors, |𝒞|=10).
- **E3** — the all-path negative control collapses by `N = 32` at fixed `D`, confirming the
  Thm 2 entropy bound (genuinely query-limited, not secretly lossless).

Not yet built: the R-HMH (§8) / ColBaC (§9) application towers (gate-unlocked, separate
brief), the MORK/PathMap MeTTa-integration shim (phase-2 `ASource` over `(VecRef h)`), and
the `PhasorHRR` backend.

## Build order

The package is built in a gated sequence (see `SPEC.md`):

| Step | Content |
|------|---------|
| 0 | `DualIndex` — stable generation-stamped handles, pluggable `ReverseBackend` |
| 1 | VSA algebra — `bind`/`unbind`/`bundle`/`permute`/`proj`/`cleanup` |
| 2 | resonator — `factorize` + `recompose_score` |
| 3 | hierarchical `encode`/`descend` |
| 4 | margin gate — `cleanup_margin` (E1) + `resonator_success_rate` (E4) |

`bind` extends `Base.bind` and `factorize` extends `LinearAlgebra.factorize` (we own
`HV`/`Codebook`, so this is method extension, not type piracy).

## API reference

```@autodocs
Modules = [FactorVSA]
Order = [:type, :function]
```

## Index

```@index
```
