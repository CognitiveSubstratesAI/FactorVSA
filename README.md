# FactorVSA.jl

[![CI](https://github.com/CognitiveSubstratesAI/FactorVSA/actions/workflows/CI.yml/badge.svg)](https://github.com/CognitiveSubstratesAI/FactorVSA/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://cognitivesubstratesai.github.io/FactorVSA/stable/)

**Resonator-factored Vector-Symbolic Architecture** over fixed-width hypervectors — the
vector/hypervector substrate leg, sibling to
[MORKTensorNetworks](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks) (the
sparse-tensor / einsum leg). Implements the VSA algebra + **resonator factorization** of
Goertzel (2026), *"Resonator-Factored Hierarchical Hypervector Embeddings"*. Depends on
[PathMap](https://github.com/CognitiveSubstratesAI/PathMap) +
[MORK](https://github.com/CognitiveSubstratesAI/MORK) only.

It is a **query-faithful cache** of an already-learned factor graph — *not* lossless storage.
The dimension `D` for reliable local queries scales with local crosstalk load `k` and the log
of codebook size, **not** raw object size.

| Step | Content |
|------|---------|
| 0 | `DualIndex` — stable generation-stamped handles, pluggable reverse backend |
| 1 | VSA algebra — `bind` / `unbind` / `bundle` / `permute` / `proj` / `cleanup` |
| 2 | resonator — `factorize` + `recompose_score` (the distinctive capability) |
| 3 | hierarchical `encode` / `descend` |
| 4 | margin gate — **PASSED** (recovery tracks Lemma 2 `2M·exp(−cD/k)`) |

The R-HMH episodic-memory and ColBaC-HDC application layers built on this substrate live in
the sibling package **[HMH.jl](https://github.com/CognitiveSubstratesAI/HMH)**.

## Install

```julia
pkg> add PathMap MORK FactorVSA      # dev the local checkouts until registered
```

## Quick example — the resonator

Factor a product hypervector back into its factor atoms in time `∝ Σ|𝒞_f|`, not `∏|𝒞_f|`:

```julia
using FactorVSA, LinearAlgebra, Random
rng = MersenneTwister(1)
D = 4096

cbs = [random_codebook(BipolarMAP, D, 10; rng=rng) for _ in 1:3]   # 3 factor codebooks
true_idx = [3, 8, 1]
factors  = [HV{BipolarMAP}(cbs[f].atoms[:, true_idx[f]]) for f in 1:3]
H = reduce(bind, factors)                                          # the bound product

recovered, score = factorize(H, cbs; restarts=3)
all(recovered[f] == factors[f] for f in 1:3)                       # true — tuple recovered
```

## MeTTa integration

FactorVSA is callable from MeTTa via MORK's grounded-op channel (upstream-aligned, zero MORK
changes). Dense vectors live in an arena, referenced by a `(VecRef h)` handle string:

```julia
register_factorvsa!()
# then from MeTTa:  (fvsa-random 4096) → (VecRef h),  (fvsa-bind a b),  (fvsa-sim a b),
#                   (fvsa-resonate H c1 c2 …) → recovered factor handles, …
```

See the **[docs](https://cognitivesubstratesai.github.io/FactorVSA/stable/)** for the full
API and `SPEC.md` for the gated build rationale.
