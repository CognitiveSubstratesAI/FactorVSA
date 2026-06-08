```@meta
CurrentModule = FactorVSA
```

# FactorVSA.jl

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
| 0 | [`DualIndex`](@ref) — stable generation-stamped handles, pluggable reverse backend |
| 1 | VSA algebra — `bind` / `unbind` / [`bundle`](@ref) / [`permute`](@ref) / [`proj`](@ref) / [`cleanup`](@ref) |
| 2 | resonator — `factorize` + [`recompose_score`](@ref) (the distinctive capability) |
| 3 | hierarchical [`encode`](@ref) / [`descend`](@ref) |
| 4 | margin gate — **PASSED** (recovery tracks Lemma 2 `2M·exp(−cD/k)`) |

The R-HMH episodic-memory and ColBaC-HDC application layers built on this substrate live in
the sibling package **[HMH.jl](https://github.com/CognitiveSubstratesAI/HMH)**.

## Install

```julia
pkg> add PathMap MORK FactorVSA       # dev the local checkouts until registered
```

```julia
using FactorVSA
```

## 30-second example — the resonator

Factor a product hypervector back into its factor atoms in time `∝ Σ|𝒞_f|`, not `∏|𝒞_f|` —
the exponential-to-linear win that names the package:

```@example quick
using FactorVSA, LinearAlgebra, Random
rng = MersenneTwister(1)
D = 4096

cbs = [random_codebook(BipolarMAP, D, 10; rng=rng) for _ in 1:3]   # 3 factor codebooks
factors = [HV{BipolarMAP}(cbs[f].atoms[:, i]) for (f, i) in enumerate([3, 8, 1])]
H = reduce(bind, factors)                                          # the bound product

recovered, score = factorize(H, cbs; restarts=3)
all(recovered[f] == factors[f] for f in 1:3)                       # true — tuple recovered
```

Next: the [Guide](guide.md) covers the algebra, resonator, hierarchical codes, the dual index,
and the MeTTa integration with runnable examples. Full signatures are in the [API](api.md).

## Capacity

Reliability is governed by a measured law (Lemma 2): single-step cleanup fails with probability
`≤ 2M·exp(−cD/k)`, so required dimension grows with local crosstalk load `k` and the *log* of
codebook size `M` — not raw object size. The Step-4 margin gate (`bench/gate.jl`, run in CI)
confirms this empirically.
