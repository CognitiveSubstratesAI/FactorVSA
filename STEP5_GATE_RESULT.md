# Step-5 (R-HMH) Margin Gate Result

**VERDICT: PASS** (2026-06-06). Backend BipolarMAP. Seed `MersenneTwister(20260606)`.
Run: `julia --project=. bench/step5_gate.jl`. Gate = §8.6 episodic admissibility:
do realistic episodes decode at feasible `D`, and does the system obey the capacity
law (recovery collapses past capacity — the anti-gaming control)?

**Headline:** episodes carry higher crosstalk load (`k_slot + k_rel`) than the bare
resonator, and recovery DOES break past the capacity boundary `k ≈ D/(2·ln M)` (Lemma 2) —
but **realistic episodes (≈10 slots, ≈5 relations) recover ≥0.95 at D=512**, well within
feasible dimension. R-HMH is HDC-admissible; the heavy parts (5c/5d) are unlocked.

> First gate draft gave a FALSE NO-GO: with small load it recovered ~1.0 everywhere and
> the negative control never degraded — the E3 lesson, the gate was only measuring the easy
> regime. Strengthened (higher M, k pushed past `D/(2·ln M)`, lower D) so the control
> actually exercises collapse. The PASS below is from the real test.

## A — recovery vs D × k_slot (k_rel=0, M=64): traces the capacity boundary
| D \ k_slot | 8 | 32 | 64 | 128 | 256 |
|---:|:--:|:--:|:--:|:--:|:--:|
|  256 | 0.975 | 0.466 | 0.252 | 0.136 | 0.078 |
|  512 | 1.000 | 0.768 | 0.463 | 0.238 | 0.136 |
| 1024 | 1.000 | 0.974 | 0.778 | 0.477 | 0.250 |
| 2048 | 1.000 | 1.000 | 0.975 | 0.778 | 0.471 |
| 4096 | 1.000 | 1.000 | 1.000 | 0.979 | 0.784 |

The ~0.47 contour lies at `k ≈ D/(2·ln M)` (M=64 ⇒ 2·ln M ≈ 8.3): break at k≈31 (D=256),
≈62 (512), ≈123 (1024), ≈247 (2048) — exactly Lemma 2. Capacity scales linearly in D.

## B — relation load, swept INDEPENDENTLY (k_slot=4, M=64)
| D | k_rel=0 | 16 | 64 | 128 | 256 |
|---:|:--:|:--:|:--:|:--:|:--:|
|  512 | 1.000 | 0.950 | 0.425 | 0.237 | 0.062 |
| 1024 | 1.000 | 1.000 | 0.800 | 0.494 | 0.244 |
| 2048 | 1.000 | 1.000 | 0.963 | 0.800 | 0.456 |

Relations degrade recovery at a rate comparable to slots — each typed relation is a bound
product (`r_ρ⊗r_i⊗r_j⊗(f_i⊗f_j)`) i.e. one crosstalk term. So `k_rel` matters as much as
`k_slot`; an episode's load is their sum. (This is why §8.6 breaks them out separately.)

## C — negative control (D=512, M=64): collapses as load grows ✓
`k_slot` 8→256: 1.000 → 0.773 → 0.473 → 0.252 → 0.130. The gate is a real test (recovery
is NOT magically lossless; it obeys the entropy/capacity bound).

## D — realistic episode (k_slot=10, k_rel=5, M=16): feasible
`D` 256→4096: 0.902, **0.993, 1.000, 1.000, 1.000**. Reaches ≥0.95 at **D=512**.

## What this unlocks / bounds
- **Unlocks 5c** (consolidation) and **5d** (ColBaC-HDC representation layer) — the episodic
  margin holds for realistic loads.
- **Design bound for builders:** keep per-episode `k_slot + k_rel` within ≈ `D/(2·ln M)`.
  At D=1024, M=64 that's ~120 role-bound terms — comfortable for typed episodes. If an
  application needs far more, raise D or hierarchically nest sub-episodes (don't bundle a
  flat 200-slot episode at D=1024).
- **Does NOT claim** episodes are lossless (control C disproves that) or that arbitrarily
  large episodes decode at fixed D.
