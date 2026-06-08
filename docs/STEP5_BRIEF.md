# Step 5 — R-HMH (§8) + ColBaC-HDC (§9): Implementation Brief

Status: **DRAFT for review — do not implement until approved.** Same brief-first,
gate-first discipline as `SPEC.md` (Steps 0–4). Source theory:
`docs/specs/tensornetworks/hdc_guided_factorization_spec.md` §8 (R-HMH) + §9 (ColBaC).
Builds entirely on FactorVSA's existing primitives (Steps 0–4 + the phase-2/2b shim).

---

## 1. SCOPE (decided 2026-06-06)

**In scope — "Both", honestly read:**
- **R-HMH (§8)** — episodic-memory application: episode encode (Eq 69), resonant partial
  recall + slot completion (Eq 70–73), episodic-semantic consolidation (Eq 77).
- **ColBaC-HDC representation layer (§9, Eq 84–90)** — column hypervector encode, support
  code, certificate hypervectors, and the HDC-derived shell-hygiene audit quantities.

**OUT of scope (fenced / other legs):**
- **The ColBaC *learner*** (HBCML columns/microcolumns/shells/teachers/Bayesian-causal
  training, §9.1–9.2) — that is the CognitiveSubstratesAI **NGC/FabricPC line**, a separate
  architecture. We build the HDC layer it *consumes*, not the learner. ColBaC-HDC here
  operates on a *generic column-structure data type* + synthetic columns for testing.
- **R-HMH neural judgment (§8.4)** — densifier `𝒟_hmh` + a neural judgment model +
  counterfactual factor-swap supervision. Needs neural integration; **deferred behind the
  5-gate** with its own design.

**KEY SIMPLIFIER:** the paper states Eq 84 (column hypervector) *is* Eq 11 (hierarchical
encoder) specialized. So R-HMH and ColBaC-HDC are **two schemas over one machinery** —
typed role-filler encode + resonant recall, which FactorVSA already provides (`encode`,
`descend`, `factorize`, `cleanup`, roles). Build the shared layer once; the two towers are
schema + vocabulary on top.

## 2. HOME (decided)

Build as **modules inside FactorVSA** (`src/RHMH.jl`, `src/ColBaCHDC.jl`), gated. **Extract
to a standalone application package** (`CognitiveMemory.jl`/`HMH.jl` at CognitiveSubstratesAI)
**after the 5-gate passes** — mirroring FactorVSA's own local-first → repo-after-gate arc.
Do not create a separate repo pre-gate.

## 3. SHARED MACHINERY (build first; both towers use it)

A small "typed structure" layer over FactorVSA:
- **Role vocabulary**: deterministic named role atoms (extend `make_roles`) for ontology
  slot-types, relation-roles, anchor-roles (R-HMH) and microcolumn-type / shell-tier / local
  roles (ColBaC). A role is a named `HV{BipolarMAP}` from a registry (reproducible by name).
- **Typed bind-bundle encode**: `H = Normalize(Σ role_i ⊗ filler_i ⊕ …)` — generalize the
  Eq-11 pattern already in `encode`. (No new algebra — bind/bundle/proj from Step 1.)
- **Role-masked recall**: unbind a role, cleanup against the slot's codebook; if the filler
  is itself a product, run `factorize` (the resonator) inside the structure. (Steps 2–3.)
- All vectors stay in the FactorVSA arena, handle-referenced (consistent with the shim).

## 4. GATED BUILD

### 5a — R-HMH episode encode (Eq 68–69) — ACTIONABLE
`Episode` = typed factor graph: `slots::Dict{role => filler-HV}`,
`relations::Vector{(role_i, role_j, relrole)}`, `anchors::Vector{(anchorrole => HV)}`, plus a
schema/event-type code. `encode_episode(E) → H_E` by Eq 69. **Acceptance:** a slot filler is
recovered by `unbind(r_s, H_E)` → `cleanup` at bounded slot-load (independent of total slots,
per Thm 1) — episodic analog of the Step-3 descent test.

### 5b — Resonant partial recall + slot completion (Eq 70–73) — ACTIONABLE
`recall(H_E, query) → completed slots`: role-masked unbind exposes a slot; when the filler
factors as a product, `factorize` recovers/completes it inside the episode (Eq 73).
**Acceptance:** given a partial cue (subset of slots), recover the missing slot fillers with
high rate in the margin regime; spurious completions rejected by recompose score.

### 5-GATE (§8.6 / §9.9 admissibility) — GO/NO-GO
Episodic analog of the Step-4 margin gate. **EXPECT THIS TO BE A REAL TEST, more likely to
fail than Step-4** — an episode bundles many role-bound slots PLUS relations (themselves bound
products) into one fixed-D vector, so its crosstalk load `k_slot + k_rel` is far higher than the
bare 3-factor resonator that passed at D=4096. Measure, as a function of `D`, slot count
`k_slot`, relation count `k_rel`, and codebook size: (1) slot-recovery accuracy, (2) cleanup
margin `Δ_cleanup`, (3) resonator success on product-fillers, (4) a negative control —
recovery must DEGRADE as `k_slot`/`k_rel` grow at fixed `D` (confirms query-limited, not
lossless). **Sweep `k_rel` INDEPENDENTLY of `k_slot`**: relations are bound products of two
fillers + a relation-role (heavier crosstalk than plain slots), so the margin most likely
gives out on relation load first — a low-slot/high-relation episode can fail where slot count
alone looks safe (§8.6 breaks out `k_slot` and `k_rel` separately for exactly this reason). **GO** iff recovery is reliable at realistic slot/relation counts and feasible `D`,
matching Lemma 2 / Thm 3. Output `STEP5_GATE_RESULT.md`. **If FAIL: STOP and report** — the
honest outcome is "episodes at this slot/relation load aren't HDC-admissible at feasible D",
not a patch. Step 5c/5d and the ColBaC-HDC certificates do not proceed until GO.

### 5c — Episodic-semantic consolidation (Eq 77) — after the gate
`consolidate(episodes) → template`: align corresponding slots (unbind/rebind), weighted
bundle into a schema/script/skill template. **Acceptance:** repeated structured episodes
yield a template whose slot pattern is recoverable; idiosyncratic anchors wash out.
**RESOLVED (2026-06-06):** consolidation produces template **HVs** (new immutable arena
handles); it does **NOT** mutate or grow codebooks. To turn consolidated templates into a
cleanup dictionary, build a **new immutable codebook** from them (a new `CodebookRef` —
"change = new handle"), exactly the phase-2b commitment. So 5c stays inside the
immutable-codebook model and never touches the reserved `codebook_version` path. (If a
future feature genuinely needs an in-place GROWING codebook, that is the deferred
mutable-codebook feature via `codebook_version`, not a quiet edit — stop and flag.)
NOTE for in-episode scoring: do NOT reuse Step-4's bare-product thresholds (`score>0.9`)
inside episodes — slots come off a PROJECTED bundle so absolute scores are ~0.5; compare
true-vs-spurious / use recovery-margin, not an absolute bare-product cutoff.

### 5d — ColBaC-HDC representation layer (§9 Eq 84–90) — after the gate
Same machinery, columnar schema:
- **Column hypervector** (Eq 84): `H_m(x) = Normalize(Σ_{u,s,a} r_u ⊗ r_s ⊗ r_a ⊗ motif)`
  over microcolumn-type `u∈{K,L,B}`, shell-tier `s∈{0..3}`, local role `a`.
- **Support code** (Eq 85): `H_S = Normalize(Σ_{m∈S} r_m ⊗ H_m)`.
- **Certificate hypervector** (Eq 86–88): `Z_m` bundling shared/specific/saturation/demotion/
  signature channels; `score(m|x)` for support retrieval.
- **Shell-hygiene HDC quantities** (Eq 89–90): promote/demote signals from reuse, cleanup
  margin `Δ`, confusability `M_conf` — computed from the encoded columns (HDC audit, not the
  learner). Operates on a generic column-structure type + synthetic columns for tests.

### FENCED (do NOT build in Step 5)
- R-HMH neural judgment (§8.4): `𝒟_hmh` + neural model + counterfactual swaps.
- The ColBaC learner (NGC/FabricPC line).

## 5. DEFINITION OF DONE (this brief = 5a/5b → gate → 5c + 5d-repr)
5a/5b implemented with acceptance tests green; 5-gate built + RUN with
`STEP5_GATE_RESULT.md` (pass/fail vs §8.6 checklist + the negative control); on GO, 5c +
the ColBaC-HDC representation layer (5d) with tests on synthetic columns. JET/Aqua/Blue + CI
green throughout. Do NOT build the fenced items. Prefer the paper's equations over invention;
leave `# SPEC?:` rather than guess.
