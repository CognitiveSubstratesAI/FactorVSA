# FactorVSA — Step-4 Margin Gate Result

**VERDICT: PASS** (2026-06-05). Backend: `BipolarMAP`. Seed: `MersenneTwister(20260605)`.
Run: `julia --project=. bench/gate.jl` (gate harness in `cleanup_margin` /
`resonator_success_rate` + an inline E3 control). Reproduce by re-running with the
same seed.

The gate tests whether the cleanup/resonator layer satisfies the paper's capacity
conditions on this substrate. PASS ⇒ FactorVSA is a supported cache here, and Step 5
(R-HMH / ColBaC towers) is unlocked. The three criteria (SPEC §4 Step 4):

## E1 — single-step cleanup capacity vs Lemma 2  `ℙ[fail] ≤ 2M·exp(−cD/k)`  (M=64)

| D \ k |   2   |   4   |   8   |  16   |
|------:|:-----:|:-----:|:-----:|:-----:|
|    64 | 0.000 | 0.025 | 0.303 | 0.607 |
|   128 | 0.000 | 0.000 | 0.058 | 0.315 |
|   256 | 0.000 | 0.000 | 0.000 | 0.055 |
|   512 | 0.000 | 0.000 | 0.000 | 0.000 |
|  1024 | 0.000 | 0.000 | 0.000 | 0.000 |
|  2048 | 0.000 | 0.000 | 0.000 | 0.000 |

Failure falls exponentially in `D/k` exactly as Lemma 2 predicts. Fitted decay
constant `c ≈ 0.21` (k=8: 0.206, k=16: 0.218 — positive, within 4×; k=2,4 saturate
to 0 too fast to fit, i.e. trivially favorable). M-dependence at D=512,k=4 is
negligible (M=16…1024 all `P[fail]=0`), confirming M enters only ~logarithmically in
the required dimension. **Criterion 1: met.**

## E2/Descent — covered by the test suite
`test/runtests.jl` "Step 3" verifies `descend(encode(tree), path)` recovers the exact
cleaned subtree code at bounded local load (the §4.2 per-level cleanup), independent of
total tree size — the structured-regime property (Thm 1).

## E4 — resonator success rate (F=3, |𝒞_f|=10) vs D

| D | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|--:|:---:|:---:|:---:|:----:|:----:|:----:|
| success | 0.260 | 0.360 | 0.515 | 0.665 | 0.855 | **0.985** |

Monotonic rise toward 1 as D grows — the conditional-resonator margin (Thm 3 / Cor 1)
holds in the favorable regime, with the exponential-in-`D/k_f` improvement. **Criterion 2: met.**

## E3 — all-path negative control (entropy bound, Thm 2) — fixed D=512

| N (items bundled) | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|--:|:-:|:-:|:-:|:--:|:--:|:--:|:--:|
| all-recovered | 1.000 | 1.000 | 0.990 | 0.355 | 0.000 | 0.000 | 0.000 |

At **fixed** D, recovering *all* N role-bound items is perfect for small N and
**collapses by N=32**. This confirms `D·q_H = Ω(N log m)` (Thm 2): a fixed-width code
is **not** secretly lossless — the method is genuinely query-limited. This is the
anti-gaming check, and it fails-as-required. **Criterion 3: met.**

## What this unlocks / does NOT
- **Unlocks:** Step 5 (R-HMH episode towers §8, ColBaC certificates §9) is now
  permitted — but it needs its **own** implementation brief (the current SPEC fences it
  and leaves it unspecified). Do not auto-build it from this gate alone.
- **Does not claim:** lossless storage, resonator convergence from any init, or that
  local HDC lookup beats an indexed tree. The gate confirms the *cache* property in the
  margin regime — nothing more (per paper §12).
- **Still unbuilt:** the MORK/PathMap MeTTa-integration shim (phase-2 `ASource` over
  `(VecRef h)`), and the `PhasorHRR` backend. The gate is BipolarMAP-only.
