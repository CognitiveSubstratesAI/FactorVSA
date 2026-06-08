# FactorVSA.jl — Implementation Brief (for a coding agent)

You are implementing `FactorVSA.jl`, a resonator-factored Vector-Symbolic
Architecture over fixed-width hypervectors. This document is your complete task
context — you do **not** need any prior conversation. Read it fully before
writing code. Build in the gated order below; **do not run ahead of the gate.**

- **Source theory:** `docs/specs/tensornetworks/hdc_guided_factorization_spec.md`
  (Goertzel 2026, "Resonator-Factored Hierarchical Hypervector Embeddings").
  Section/Eq/Algorithm references below (§N, Eq N, Alg N) point into it.
- **Skeleton (already created + verified-to-resolve):** `src/FactorVSA.jl` has
  every type and function as a stub that throws `_todo(...)`. You fill the bodies.
  `Project.toml` is set and **confirmed to resolve + precompile**; do not change deps
  without re-resolving. `test/runtests.jl` marks unbuilt steps `@test_broken` — flip
  each to a real `@test` as you implement it.

---

## 1. FIXED CONSTRAINTS (decided — do not re-litigate, do not "improve")

1. **Name / identity:** package is `FactorVSA` (the resonator-factorization is the
   distinctive capability; this is *not* "all of HDC"). It is the **vector/hypervector
   substrate leg**, sibling to `MORKTensorNetworks` (the sparse-tensor/einsum leg).
2. **Dependencies:** `PathMap` + `MORK` **only** (plus stdlib `LinearAlgebra`,
   `Random`). **`MORKTensorNetworks`/`ShardZipper` is NOT a dependency** — it is an
   *optional* `Pkg` extension (weakdep) added later, never a core dep.
3. **Identity-parameterized:** the algebra is generic over the VSA backend
   (`Backend` abstract type). Implement **`BipolarMAP` first** (`{−1,+1}^D`,
   bind = elementwise multiply, self-inverse, `Proj_ℍ = sign`). `PhasorHRR` later.
4. **Query-faithful, NOT lossless.** This is a *cache of an already-learned factor
   graph* (§0, §1). Never claim or design for full reconstruction. The dimension `D`
   scales with local crosstalk load `k` and `log` codebook size — **not** raw object
   size (§5.1 Lemma 2). Any "D independent of N" statement must stay query-limited.
5. **Two-phase MeTTa op-integration** (see §6 below): phase-1 returns *scalars* via
   the existing grounded path; phase-2 dense ops via a new `ASource` subtype over an
   arena handle. Phase-2 is NOT needed for the gate.
6. **Local-first.** Develop in-tree (`packages/FactorVSA`, path-dev'd). Migrate to
   the CognitiveSubstratesAI org (CI/Aqua/BlueStyle/Documenter) **only after the
   Step-4 gate passes** — flip `[sources]` paths to git URLs then.

---

## 2. VERIFIED GROUND TRUTH (checked against the resolved tree — trust these)

These were verified by reading the actual Julia MORK/PathMap source on a resolved
tree (not upstream Rust, not a whitepaper). File:line provenance given. Do **not**
re-derive or "helpfully" reinvent them.

- **MORK's grounding registry is string-typed:** `const GROUNDED_REGISTRY =
  Dict{String, Function}()`, `register_grounded!(name::String, f::Function)`
  (`packages/MORK/src/kernel/Sources.jl:254,263`). It carries **scalar symbolic**
  results only (string in / string out). It **cannot carry dense vector data.**
- **The source/sink extension point is an abstract-type + multiple-dispatch
  mechanism, NOT an enum.** Sources: `abstract type ASource`, dispatched by
  `asource_new(e)::ASource` (`Sources.jl:412`), with concrete exemplars
  `BTMSource`/`ACTSource`/`CmpSource`/`GroundedSource`, each a `struct <: ASource` +
  a `source_factor(s, btm::PathMap{UnitVal})` method. Sinks: `abstract type
  AbstractSink` + `sink_apply!`/`sink_finalize!` (`packages/MORK/src/kernel/Sinks.jl:33`),
  exemplars `AddSink`/`RemoveSink`/`HeadSink`. **To add a dense-op channel you write a
  `struct <: ASource` + a `source_factor` method + a case in `asource_new` — you do
  NOT add an enum variant.**
- **Grounding is itself a source:** `GroundedSource <: ASource`
  (`Sources.jl:281`). So phase-1 (grounded scalar) and phase-2 (custom dense source)
  are **the same mechanism, two source types** — not two different channels.
- **ShardZipper is einsum/tensor-scoped and lives in MORKTensorNetworks**
  (`packages/MORKTensorNetworks/src/shard/ShardZipper.jl`; exports
  `Shard`/`partition_trie`/`materialize!`; "sparse einsums (SpGEMM-like)"). The HDC
  resonator/cleanup path is **not** an einsum cascade, so FactorVSA does **not** need
  ShardZipper. It is the optional extension only.
- **There is no O(1) by-ref graft in the port.** `wz_graft!` exists
  (`packages/MorkServer/src/Commands.jl:187`) but in this port it is an **O(n)
  value-copy**, not the upstream O(1) by-ref share (the "Mirrors wz.graft" comment is
  wrong). **Do not design around O(1) trie sharing.**
- **`Base.hash` IS deterministic across processes** for a given Julia build (verified:
  two processes give identical `hash(Vector{UInt8}(...))`). Not relevant to single-process
  FactorVSA, but do not assume Python-style per-process hash randomization anywhere.

---

## 3. ANTI-INSTRUCTIONS (forbidden — you will be tempted; don't)

- **DO NOT** route dense vector/hypervector data through `GROUNDED_REGISTRY` — it is
  string-typed. Dense ops use the phase-2 `ASource`-over-arena-handle channel.
- **DO NOT** inline dense hypervector data into the MORK trie as content. Vectors live
  in the **FactorVSA arena** (a plain Julia store); MeTTa references them only by an
  opaque handle atom `(VecRef h)`. The trie stores the *handle*, never the buffer.
- **DO NOT** build Step 5 (§8 R-HMH towers / §9 ColBaC certificates) until the Step-4
  margin gate has passed. They are fenced for a reason — they are worthless if the
  underlying cleanup/resonator margins don't hold.
- **DO NOT** add `MORKTensorNetworks`/`ShardZipper` as a core dependency.
- **DO NOT** use an `@enum`/"AFactor enum" for the integration channel (see §2).
- **DO NOT** claim lossless storage or full reconstruction (Constraint 4).
- **DO NOT** use the obsolete `PRIMUS_*` packages (e.g. `PRIMUS_Core`,
  `PRIMUS_Metagraph`) as reference — they are legacy. Reference `MORK`, `PathMap`,
  `MORKTensorNetworks` (the hardened siblings) only.
- **DO NOT** invent a MeTTa→GPU transpiler. Per the MORK authors, full transpilation
  is "expensive, hard, and rarely needed"; the intended pattern is hand-written
  kernels dispatched via the source/sink channel.

---

## 4. BUILD SEQUENCE — gated

Implement strictly in order. Each step: implement the stubs in `src/FactorVSA.jl`,
flip the matching `@test_broken` → `@test` in `test/runtests.jl`, and keep the suite
green before moving on. Backend = `BipolarMAP` throughout Steps 1–4. **Step 0 is the
unconditional foundation; Steps 1–3 sit on top of it; Step 4 is the gate; Step 5 is
fenced.**

### STEP 0 — DUAL INDEX (foundation) — ACTIONABLE, **UNCONDITIONAL (not gated)**
The bidirectional binding between symbolic identity and stored vector — **forward**
(`cid → handle → vector`) and **reverse** (`vector/query → atoms`). This is the
substrate the resonator and every other leg plug into. It must exist before Step 2
has anything to attach to, and it is needed **whether or not the Step-4 gate passes**
(the plain-cleanup and dense-KNN legs use the same index). It is the one piece that
survives a gate failure — so it is *not* behind the gate.

**Why it exists / what it replaces:** it is the consolidated successor to the legacy
trio `PRIMUS_Neural/{VectorSpace.jl, HMHSpace.jl, outside_mode/EmbeddingSpace.jl}` —
three inconsistent implementations of the same idea. The most-developed one,
`VectorSpace.jl:80`, assigns `idx = UInt32(length(data) + 1)` — **identity tied to
insertion order**, so any `deleteat!` silently desyncs `idx_to_cid`. It also hardcodes
`UInt128` CID. **DO NOT copy that positional design.** If you grep `VectorSpace.jl`
(or `HMHSpace.jl`/`EmbeddingSpace.jl`) to see "how it was done", read it as the
**cautionary example — the bug to avoid — not a reference to mirror.** These are
legacy `PRIMUS_*` code; per §3 they are the anti-pattern being replaced, not a template.

Implement (stubs already in `src/FactorVSA.jl`):
- **Stable handles, not positions** — `VectorHandle = UInt32` is a logical slot into a
  pooled `arena::Vector{V}` with a `free` list and a per-slot `gen` counter. The arena
  is **never reordered**; delete returns the slot to `free` and bumps its generation.
- **Generation/ABA guard** — `HandleRef = (handle, generation)`; `deref` returns the
  vector iff `gen[handle] == ref.generation && live[handle]`, else `nothing` (stale).
  This closes the "handle reused after free" hole with zero happy-path cost.
- **Pluggable reverse** — `abstract type ReverseBackend`; ship `ArenaScanBackend`
  (similarity scan over live handles, PathMap+MORK only). Reverse is a trait because it
  differs per leg: KNN (dense) / cleanup (flat HDC) / **resonator (Step 2)**. The
  Step-2 resonator becomes a `ReverseBackend` for the factorized leg — that is the seam
  the Step-4 gate measures against. (A `ShardZipperBackend` over MORKTN is a *future
  optional Pkg extension*, never a core dep.)
- **One generic type** — `DualIndex{Id, V, B}`: `Id` abstract (inject `UInt128` only at
  the PRIMUS shim, never bake it in — the legacy code's mistake), `V` the vector type,
  `B` the reverse backend. Operations: `insert_vector!` (**named to avoid colliding
  with the VSA `bind`** — the thread called it `bind!`), `lookup_vector` (forward),
  `reverse_lookup` (pluggable), `free_vector!`, `deref`, `rebuild_cache!`.

**Two integrity rules (these are what make it correct — assert them as tests):**
1. **Identity is mediated by stable handles, never by arena position.** A handle stays
   valid across any number of later inserts/deletes. (Test: insert A,B,C; free B;
   insert D; A's and C's handles still deref to A and C; B's old `HandleRef` is stale.)
2. **The MORK space is the system of record; the index is a cache.** The `(VecRef h)`
   atom in the trie is authoritative for `cid → handle`; `id_to_handle` is a rebuildable
   cache (`rebuild_cache!`). The arena is authoritative for `handle → vector`. (Test:
   clear `id_to_handle`, `rebuild_cache!` from the atom list, forward lookup still works.)

`codebook_version` keys derived/encoded vectors: a codebook bump invalidates them;
primary (perceptual) vectors are codebook-independent and ignore it.
**Acceptance:** the two integrity-rule tests above pass; no operation assumes positional
order; deletes never corrupt surviving handles.

### STEP 1 — VSA algebra + cleanup (paper §2; Eq 3–4) — ACTIONABLE (on top of Step 0)
Implement `bind`, `unbind`, `bundle`, `permute`/`invpermute`, `proj` (Proj_ℍ),
`cleanup` (hard, Eq 3 argmax), `cleanup_soft` (Eq 3, `𝒞·softmax(β𝒞ᵀz)`), and the
`Codebook` helpers. For `BipolarMAP`: bind = elementwise `*` (self-inverse, so
`unbind == bind`), bundle = sum then optional `sign`, `proj = sign`, permute = cyclic
shift by `ρ`. **Acceptance:** `unbind(a, bind(a, v)) ≈ v` up to crosstalk; a bound
vector is ~orthogonal to its inputs; `cleanup` recovers an atom from `atom + few
crosstalk terms` (this is the seed of the gate).

### STEP 2 — resonator factorization (paper §4.3, Alg 3; Eq 13) — ACTIONABLE
Implement `factorize` (Alg 3: init each `x̂_f` to the normalized codebook sum; iterate
per-slot `z_f = H ⊗ (⊗_{g≠f} proj(x̂_g)†)`, `x̂_f ← cleanup(z_f, 𝒞_f)`; stop when all
hard cleanups stable) and `recompose_score` (Eq 13, `⟨H, x̂₁⊗…⊗x̂_F⟩`). Support
`restarts` and `beam` (multi-start / top-K) and use `recompose_score` to reject
spurious fixed points. **Acceptance:** for a random product `y⋆ = x₁⊗…⊗x_F` with
codebook sizes in the margin-favorable regime, `factorize(y⋆)` returns the true tuple
with high rate; cost scales with `Σ_f|𝒞_f|`, not `∏_f|𝒞_f|`. **Integration:** wrap the
resonator as a `ReverseBackend` (Step 0) for the factorized leg, so the Step-4 gate
exercises it against the real `DualIndex`, not an improvised lookup.

### STEP 3 — hierarchical encode/descend (paper §4.1, Alg 1 & 2; Eq 11) — ACTIONABLE
Implement `encode` (Alg 1, Eq 11: `τ_t⊗u_ν ⊕ Σ r_{t,i}⊗ρ(H_child) ⊕ σ_t⊗S_ν`,
normalized) and `descend` (Alg 2: per step `ρ⁻¹(r†_{t,i}⊗h)` then hard-cleanup vs the
level subtree codebook). The line-5 cleanup is **load-bearing** — without it crosstalk
accumulates to whole-object scale (§4.2). **Acceptance:** on bounded-branching trees
with known level codebooks, `descend(encode(tree), path)` recovers the node at `path`;
per-step load stays local (independent of total `N`) — paper experiment **E2**.

### STEP 4 — THE MARGIN GATE (paper §5.1 Lemma 2; §5.3 Thm 3; experiments E1, E4) — GO/NO-GO
This decides whether the whole approach is supported on this substrate. Build the gate
instrumentation alongside Steps 1–3 (`cleanup_margin`, `resonator_success_rate`).

- **E1 (`cleanup_margin`):** bundle `k` random atoms, unbind target, hard-clean vs a
  size-`M` codebook; measure empirical `ℙ[cleanup fails]` over a sweep of `D, k, M`.
- **E4 (`resonator_success_rate`):** random target products; run `factorize`; count
  recoveries with `recompose_score` above threshold; record mean iterations.

**GATE CRITERION (must be met to proceed to Step 5):**
1. Measured `ℙ[cleanup fails]` tracks **Lemma 2: `≤ 2M·exp(−cD/k)`** — i.e. failure
   falls exponentially in `D/k` and only `log`-linearly in `M`. Fit `c`; it must be a
   stable positive constant across the sweep.
2. Resonator success rate is high in the margin regime and degrades as Cor 1 predicts
   (`1 − 2I Σ_f|𝒞_f|exp(−cD/k_f)`), with recomposition reliably separating true from
   spurious fixed points.
3. The **all-path negative control (E3)** fails as `N` grows at fixed `D` (confirms you
   are not secretly violating the `D·q_H = Ω(N log m)` entropy bound, Thm 2) — i.e. the
   method is genuinely query-limited, not accidentally lossless.

If the gate **passes**, FactorVSA is a supported cache for this substrate → Step 5 is
unlocked. If it **fails** (margins don't hold at feasible `D`), STOP and report — the
honest outcome is "this factorization/substrate is not HDC-admissible" (§6.4), not a
patch. Either way, the gate result is the deliverable that matters.

### STEP 5 — APPLICATIONS (paper §8 R-HMH, §9 ColBaC) — BUILT + EXTRACTED to HMH.jl
The Step-4 gate passed, Step 5 was built (R-HMH episode codes Eq 69, resonant recall/slot
completion §8.3, consolidation Eq 77; ColBaC-HDC column/support/certificate codes Eq 84–88
+ audit quantities), its episodic admissibility gate PASSED, and the whole application layer
was **extracted to the sibling package `HMH.jl`** (github.com/CognitiveSubstratesAI/HMH) to
keep FactorVSA a pure substrate. STILL FENCED there: §8.4 neural judgment (densifier `𝒟_hmh`
+ neural model) and the ColBaC *learner* (the NGC/FabricPC line). FactorVSA itself stops at
Step 4 + the phase-2/2b MeTTa shim.

---

## 5. INTERFACE CONTRACT (already stubbed in `src/FactorVSA.jl` — match these)

Types: `Backend` (`BipolarMAP`, `PhasorHRR`), `HV{B<:Backend}` (buffer lives in a
`DualIndex` arena, addressed by handle), `Codebook{B}` (`D×M` + names);
**Step 0:** `VectorHandle`, `HandleRef`, `ReverseBackend`, `ArenaScanBackend`,
`DualIndex{Id,V,B}`. Functions (signatures in the stub):
**Step 0** `insert_vector!/lookup_vector/reverse_lookup/free_vector!/deref/rebuild_cache!`;
**Step 1** `bind/unbind/bundle/permute/invpermute/proj/cleanup/cleanup_soft`;
**Step 2** `factorize/recompose_score`; **Step 3** `encode/descend`;
**Step 4 gate** `cleanup_margin/resonator_success_rate`. Keep the pure algebra
(Steps 1–3) free of MORK/PathMap — those are touched only at the §6 integration
boundary and via the Step-0 `(VecRef h)` handle binding.

---

## 6. MeTTa OP-INTEGRATION — IMPLEMENTED (`MeTTaShim.jl`)

**CORRECTED after an upstream cross-check (2026-06-06, dev-zone MORK kernel+server+docs).**
The original draft proposed a custom `FactorVSASource <: MORK.ASource`. The cross-check
showed that is *unnecessary* and over-engineered: a custom source would require modifying
MORK's closed `asource_new` dispatch, and it's only the right idiom when dense data must
flow *through* the trie/zipper machinery (the GPU-einsum case). FactorVSA keeps vectors in
its own arena and references them by a tiny `(VecRef h)` handle string — so the existing
string-based grounded channel suffices, with **zero MORK changes**.

Upstream finding: MORK (kernel *and* server) has **no grounding of its own**; the roadmap
delegates grounding to the consuming runtime as "grounded ops inverted into queries". The
Julia port realizes this as `register_grounded!` + `GroundedSource` (an I-pattern source),
and `asource_new` auto-routes any registered name to it. So the shim is upstream-aligned.

**Implemented design** (`src/MeTTaShim.jl`):
- Dense vectors live in a process-global arena (the Step-0 `DualIndex`), referenced from
  MeTTa only by a `(VecRef h)` handle string. Dense data NEVER enters the trie.
- `register_factorvsa!()` registers grounded ops via `register_grounded!`:
  `(fvsa-random D)`, `(fvsa-bind a b)`, `(fvsa-unbind a b)`, `(fvsa-bundle a …)` →
  `(VecRef h)`; `(fvsa-sim a b)` → scalar. Handlers parse handle-strings, op in the arena,
  store dense results, return a fresh handle. Both scalar and dense results go through the
  one string channel (dense via handle).
- **Phase-2b (DONE):** the resonator is callable from MeTTa. Codebooks are **immutable
  `(CodebookRef c)` handles** (a process-global registry; "change" = a new handle, never
  in-place mutation) — which **decouples them from `DualIndex.codebook_version`** (no
  mutation ⇒ nothing to invalidate; the field stays reserved for a future mutable/encoding-
  cache feature). Ops: `(fvsa-codebook D M)`, `(fvsa-codebook-atom c i)`, `(fvsa-cleanup z c)`,
  `(fvsa-resonate H c1 c2 …)` (→ recovered factor tuple, multi-start), `(fvsa-recompose-score
  H f…)`, `(fvsa-free-codebook c)`. So the package's headline capability — resonator
  factorization — is now MeTTa-callable, not just base VSA.

---

## 7. DEFINITION OF DONE (this brief's scope = Steps 0–4)

**Step 0** implemented with the two integrity-rule tests green (stable handles survive
deletes; cache rebuildable from `(VecRef h)` atoms); Steps 1–3 implemented with their
acceptance tests green (no `@test_broken` left for them); **Step 4** gate instrumentation
built and **run**, with the E1 fit, E4 curve, and E3 negative control reported and
compared to Lemma 2 / Thm 3 / Thm 2. Output a short `GATE_RESULT.md` stating pass/fail
with the measured constants and plots/tables. **Do not start Step 5.** If anything here
is ambiguous, prefer the paper's equations over your own design instinct, and leave a
`# SPEC?:` comment rather than inventing.

**Required tooling gate (run on the IMPLEMENTED code, not stubs):**
- **JuliaFormatter** — Blue-92 (`.JuliaFormatter.toml` is present, matching the siblings);
  `format(".")` must be a no-op before commit.
- **Aqua.test_all** green — including `stale_deps` (it currently flags `MORK`/`PathMap`
  because the skeleton doesn't reference them; once Step 0's `(VecRef h)` cache + the
  phase-2 `ASource` are wired, they become real deps and it goes green — do NOT silence
  it with a no-op `using`).
- **JET.report_package** — zero real inference errors (stub `_todo` throws will vanish as
  bodies land; the remaining report must be clean).
- **`@code_warntype` / Cthulhu** on the hot paths (`bind`/`cleanup`/`factorize` inner
  loops, `deref`) — type-stable, no `Any`/`Union` in the kernels (this is a perf substrate).
- **AllocCheck.jl** on the resonator/cleanup inner loops — no unexpected allocations in
  the per-iteration path (the gate's compute accounting, §5.6, depends on it).
None of these are meaningful until the bodies exist — they are the *done* criteria, not
a scaffold check.
