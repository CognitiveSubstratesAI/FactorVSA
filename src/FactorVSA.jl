"""
    FactorVSA

Resonator-factored Vector-Symbolic Architecture over fixed-width hypervectors.

Implements the VSA algebra + resonator factorization of Goertzel (2026),
"Resonator-Factored Hierarchical Hypervector Embeddings" (see SPEC.md). This is
the *vector / hypervector* substrate leg — sibling to MORKTensorNetworks (the
sparse-tensor / einsum leg). It depends on MORK (which brings PathMap transitively)
plus stdlib.

STATUS: Steps 0–4 IMPLEMENTED for the `BipolarMAP` backend (gate-scoped), Step-4
margin gate PASSED, and the phase-2 MeTTa-integration shim is built (`MeTTaShim.jl`
— grounded `(fvsa-*)` ops over a handle-referenced arena via MORK's
`register_grounded!`; cross-checked upstream-aligned, zero MORK changes).
`PhasorHRR` methods remain `_todo` stubs. Step 5 (R-HMH / ColBaC towers) is FENCED.
Phase-2b (codebook-dependent `cleanup`/`resonate` ops) is deferred.
"""
module FactorVSA

using LinearAlgebra
using Random
# Extend (not pirate — we own HV/Codebook) so these work under `using` despite the
# stdlib names `Base.bind` (Channels) and `LinearAlgebra.factorize` (matrices).
import Base: bind
import LinearAlgebra: factorize

# MORK is used by the phase-2 MeTTa-integration shim (MeTTaShim.jl) — grounded ops
# via `register_grounded!`, with dense vectors kept in the FactorVSA arena and
# referenced from MeTTa only by `(VecRef h)` handle strings. The pure VSA algebra
# + the gate (Steps 0-4) don't need it. PathMap comes transitively through MORK
# (FactorVSA has no direct PathMap use), so it is not a direct dependency.

export Backend, BipolarMAP, PhasorHRR
export HV, dim, backend
export Codebook, codebook_matrix, random_hv, random_codebook
export VectorHandle, HandleRef, ReverseBackend, ArenaScanBackend, DualIndex   # Step 0
export insert_vector!, lookup_vector, reverse_lookup, free_vector!, deref, rebuild_cache!
export unbind, bundle, permute, invpermute, proj, cleanup, cleanup_soft   # `bind` extends Base.bind
export recompose_score   # `factorize` extends LinearAlgebra.factorize
export VSATree, Roles, make_roles, encode, descend
export cleanup_margin, resonator_success_rate   # Step-4 gate instrumentation

_todo(what) = error("FactorVSA: NOT IMPLEMENTED — $what. See SPEC.md.")

# ─────────────────────────────────────────────────────────────────────────────
# Backends (paper §2). Identity-parameterized: the algebra is generic over the
# binding model; BipolarMAP is implemented. PhasorHRR is left as stubs.
# ─────────────────────────────────────────────────────────────────────────────
abstract type Backend end
struct BipolarMAP <: Backend end   # {-1,+1}^D, ⊗ = elementwise *, self-inverse
struct PhasorHRR <: Backend end    # complex unit phasors (not implemented)

"""A fixed-width hypervector tagged by its `Backend`. The dense buffer is stored in
a `DualIndex` arena (Step 0) and addressed by a stable `VectorHandle` — it is NEVER
serialized into the MORK trie. The trie holds only the `(VecRef h)` handle atom."""
struct HV{B <: Backend}
    data::Vector{Float64}   # bipolar: ±1.0 for atoms, real for un-projected sums/unbinds
end

dim(h::HV) = length(h.data)
backend(::HV{B}) where {B} = B
Base.:(==)(a::HV{B}, b::HV{B}) where {B} = a.data == b.data

"""A `D×M` codebook (M atoms of dimension D) over a backend `B` plus optional names."""
struct Codebook{B <: Backend}
    atoms::Matrix{Float64}        # D × M
    names::Vector{Symbol}
    # typed inner ctor: narrows JET inference (no Any-arg union-split exploration)
    Codebook{B}(atoms::AbstractMatrix{<:Real}, names::Vector{Symbol}) where {B <: Backend} =
        new{B}(atoms, names)
end
codebook_matrix(c::Codebook) = c.atoms
Base.length(c::Codebook) = size(c.atoms, 2)

# Random generators (BipolarMAP). Atoms are i.i.d. uniform ±1 (Assumption 1).
random_hv(::Type{BipolarMAP}, D::Int, rng::AbstractRNG=Random.default_rng()) =
    HV{BipolarMAP}(rand(rng, (-1.0, 1.0), D))
function random_codebook(::Type{BipolarMAP}, D::Int, M::Int;
    rng::AbstractRNG=Random.default_rng(), prefix::Symbol=:c)
    HV  # touch to keep type inferred; no-op
    atoms = rand(rng, (-1.0, 1.0), D, M)
    names = [Symbol(prefix, i) for i in 1:M]
    Codebook{BipolarMAP}(atoms, names)
end

_softmax(x::AbstractVector{<:Real}) = (m=maximum(x); e=exp.(x .- m); e ./ sum(e))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — DUAL INDEX (UNCONDITIONAL foundation; NOT gated; see SPEC.md §Step 0)
# Consolidated successor to legacy PRIMUS_Neural/{VectorSpace,HMHSpace,Embedding}.
# Stable handles (never positional), generation/ABA guard, pluggable reverse.
# ─────────────────────────────────────────────────────────────────────────────
const VectorHandle = UInt32

"A handle plus the generation it was minted at. Deref fails if the slot was freed/reused."
struct HandleRef
    handle::VectorHandle
    generation::UInt32
end

"Reverse-lookup strategy — the seam. Forward is shared/concrete; reverse is pluggable."
abstract type ReverseBackend end

"""Arena-native default reverse backend: similarity scan over the live arena, keyed by
handle. PathMap+MORK only, NO MORKTensorNetworks. The Step-2 resonator can become a
`ReverseBackend` for the factorized leg; a `ShardZipperBackend` over MORKTN is a future
optional Pkg extension."""
struct ArenaScanBackend <: ReverseBackend end

"""
    DualIndex{Id, V, B<:ReverseBackend}

One generic index collapsing the legacy trio. `Id` = identity type (kept abstract),
`V` = stored vector type (`HV{...}` or a dense vector), `B` = reverse backend. Shared
arena (authoritative `handle → vector`) + a rebuildable `id ↔ handle` cache.
"""
mutable struct DualIndex{Id, V, B <: ReverseBackend}
    arena::Vector{V}                          # handle → vector; NEVER reordered
    gen::Vector{UInt32}                       # per-slot generation (ABA guard)
    live::BitVector                           # slot occupied?
    free::Vector{VectorHandle}                # free-list of reclaimable slots
    id_to_handle::Dict{Id, VectorHandle}      # rebuildable cache over (VecRef h) atoms
    handle_to_id::Dict{VectorHandle, Id}
    codebook_version::UInt32                  # invalidates derived/encoded vectors on bump
    backend::B
end
# codebook_version — DELIBERATELY DORMANT (decided phase-2b, 2026-06-06).
#   It exists to invalidate CACHED ENCODINGS when a codebook MUTATES in place. The
#   phase-2b shim makes codebooks IMMUTABLE handle-referenced objects (a `(CodebookRef c)`
#   never mutates; "change" = register a new handle), and resonate/cleanup produce
#   STANDALONE result HVs that retain no codebook dependency — so there is nothing to
#   invalidate and this field is correctly left unused. Reserved for a future
#   mutable-codebook / encoding-cache feature; do NOT wire it for immutable codebooks.
DualIndex{Id, V}(backend::B=ArenaScanBackend()) where {Id, V, B <: ReverseBackend} =
    DualIndex{Id, V, B}(V[], UInt32[], falses(0), VectorHandle[],
        Dict{Id, VectorHandle}(), Dict{VectorHandle, Id}(), UInt32(0), backend)

"""Insert `v` under identity `id`; allocate (or reuse a freed) slot; return a
generation-stamped `HandleRef`. The handle is stable for the binding's lifetime
regardless of later inserts/deletes. (Named `insert_vector!`, not `bind!`, to avoid
colliding with the VSA `bind`.)"""
function insert_vector!(idx::DualIndex{Id, V}, id::Id, v::V) where {Id, V}
    if isempty(idx.free)
        push!(idx.arena, v)
        push!(idx.gen, UInt32(0))
        push!(idx.live, true)
        h = VectorHandle(length(idx.arena))
    else
        h = pop!(idx.free)
        idx.arena[h] = v
        idx.live[h] = true            # gen[h] was already bumped at free time
    end
    idx.id_to_handle[id] = h
    idx.handle_to_id[h] = id
    HandleRef(h, idx.gen[h])
end

"Forward: `id → handle → vector`, checking liveness. Returns `nothing` if absent/dead."
function lookup_vector(idx::DualIndex{Id}, id::Id) where {Id}
    h = get(idx.id_to_handle, id, nothing)
    h === nothing && return nothing
    idx.live[h] ? idx.arena[h] : nothing
end

"""Reverse lookup, dispatched on the backend. `ArenaScanBackend` = similarity scan
(dot product) over live handles, returning the top-`topk` by score. NEVER assumes
positional order."""
function reverse_lookup(::ArenaScanBackend, idx::DualIndex{Id, HV{B}}, query::HV{B};
    topk::Int=5) where {Id, B}
    scored = Tuple{Float64, VectorHandle}[]
    for h in 1:length(idx.arena)
        idx.live[h] || continue
        push!(scored, (dot(idx.arena[h].data, query.data), VectorHandle(h)))
    end
    sort!(scored; by=first, rev=true)
    [h for (_, h) in scored[1:min(topk, length(scored))]]
end
reverse_lookup(idx::DualIndex, query; kwargs...) =
    reverse_lookup(idx.backend, idx, query; kwargs...)

"Free a slot: clear `live`, bump `gen[handle]` (ABA), drop id-map entries, recycle."
function free_vector!(idx::DualIndex, h::VectorHandle)
    (h <= length(idx.live) && idx.live[h]) || return nothing
    idx.live[h] = false
    idx.gen[h] += UInt32(1)
    id = get(idx.handle_to_id, h, nothing)
    id !== nothing && delete!(idx.id_to_handle, id)
    delete!(idx.handle_to_id, h)
    push!(idx.free, h)
    nothing
end

"Generation-checked deref: the vector iff `gen[handle]==ref.generation && live[handle]`, else `nothing`."
function deref(idx::DualIndex, ref::HandleRef)
    if (
        ref.handle <= length(idx.arena) && idx.live[ref.handle] &&
        idx.gen[ref.handle] == ref.generation
    )
        idx.arena[ref.handle]
    else
        nothing
    end
end

"Rebuild the `id ↔ handle` cache from authoritative `(id, handle)` pairs (cache is recoverable)."
function rebuild_cache!(idx::DualIndex{Id}, pairs) where {Id}
    empty!(idx.id_to_handle)
    empty!(idx.handle_to_id)
    for (id, h) in pairs
        hh = VectorHandle(h)
        idx.id_to_handle[id] = hh
        idx.handle_to_id[hh] = id
    end
    idx
end

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — core algebra (paper §2, Eq 3-4)
# ─────────────────────────────────────────────────────────────────────────────
"Binding `u ⊗ v` (BipolarMAP: elementwise product; self-inverse)."
bind(a::HV{BipolarMAP}, b::HV{BipolarMAP}) = HV{BipolarMAP}(a.data .* b.data)
"Unbinding (BipolarMAP: `a† == a`, so identical to bind)."
unbind(a::HV{BipolarMAP}, b::HV{BipolarMAP}) = HV{BipolarMAP}(a.data .* b.data)
"Bundling `⊕` (superpose): elementwise sum, UN-projected (caller applies `proj`)."
function bundle(hs::HV{BipolarMAP}...)
    isempty(hs) && error("bundle needs ≥1 argument")
    HV{BipolarMAP}(reduce(+, (h.data for h in hs)))
end
"Permutation `ρ` (BipolarMAP: cyclic shift by `k`; depth/order role)."
permute(h::HV{BipolarMAP}, k::Int=1) = HV{BipolarMAP}(circshift(h.data, k))
invpermute(h::HV{BipolarMAP}, k::Int=1) = HV{BipolarMAP}(circshift(h.data, -k))
"Projection to the manifold (Eq 4; BipolarMAP: `sign`, with 0 → +1)."
proj(h::HV{BipolarMAP}) = HV{BipolarMAP}([x >= 0 ? 1.0 : -1.0 for x in h.data])
"Hard cleanup: nearest codebook atom (Eq 3, argmax ⟨c,z⟩)."
cleanup(z::HV{BipolarMAP}, cb::Codebook{BipolarMAP}) =
    HV{BipolarMAP}(cb.atoms[:, _cleanup_idx(z, cb)])
"Soft cleanup: `𝒞·softmax(β 𝒞ᵀz)` (Eq 3, differentiable blend; un-projected)."
cleanup_soft(z::HV{BipolarMAP}, cb::Codebook{BipolarMAP}; beta::Real=1.0) =
    HV{BipolarMAP}(cb.atoms * _softmax(beta .* (cb.atoms' * z.data)))
_cleanup_idx(z::HV{BipolarMAP}, cb::Codebook{BipolarMAP}) = argmax(cb.atoms' * z.data)

# generic (non-BipolarMAP) fallbacks remain unimplemented
bind(::HV{B}, ::HV{B}) where {B} = _todo("Step 1: bind for $B")
proj(::HV{B}) where {B} = _todo("Step 1: proj for $B")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — resonator factorization (paper §4.3, Alg 3; §5.3 margin; Eq 13)
# ─────────────────────────────────────────────────────────────────────────────
"""
    factorize(H, codebooks; iterations, restarts, beam, beta, rng) → (factors, score)

Resonator (Alg 3): init each `x̂_f` to the normalized codebook sum; alternate per-slot
`z_f = H ⊗ (⊗_{g≠f} Proj(x̂_g)†)`, `x̂_f ← cleanup_soft(z_f, 𝒞_f)`; stop when the hard
argmaxes stabilize. Returns the hard-cleanup factor tuple and the recomposition score
`⟨H, x̂₁⊗…⊗x̂_F⟩/D` (Eq 13). `restarts` runs multi-start and keeps the best score.
Correctness is CONDITIONAL on the §5.3 margin (the gate).
"""
function factorize(H::HV{BipolarMAP}, cbs::Vector{Codebook{BipolarMAP}};
    iterations::Int=50, restarts::Int=1, beam::Int=1, beta::Real=3.0,
    rng::AbstractRNG=Random.default_rng())
    F = length(cbs)
    F == 0 && return (HV{BipolarMAP}[], -Inf)
    best_factors = HV{BipolarMAP}[]
    best_score = -Inf
    for r in 1:restarts
        xhat = Vector{HV{BipolarMAP}}(undef, F)
        for f in 1:F
            xhat[f] = if r == 1
                proj(HV{BipolarMAP}(vec(sum(cbs[f].atoms; dims=2))))
            else
                random_hv(BipolarMAP, dim(H), rng)
            end
        end
        prev = fill(0, F)
        idxs = fill(0, F)
        for _ in 1:iterations
            for f in 1:F
                acc = copy(H.data)
                for g in 1:F
                    g == f && continue
                    acc = acc .* proj(xhat[g]).data
                end
                zf = HV{BipolarMAP}(acc)
                idxs[f] = _cleanup_idx(zf, cbs[f])
                xhat[f] = cleanup_soft(zf, cbs[f]; beta=beta)
            end
            idxs == prev && break
            copyto!(prev, idxs)
        end
        factors = [HV{BipolarMAP}(cbs[f].atoms[:, idxs[f]]) for f in 1:F]
        sc = recompose_score(H, factors)
        if sc > best_score
            best_score = sc
            best_factors = factors
        end
    end
    (best_factors, best_score)
end

"Recomposition score `⟨H, x̂₁⊗…⊗x̂_F⟩ / D` (Eq 13)."
function recompose_score(H::HV{BipolarMAP}, factors::Vector{HV{BipolarMAP}})
    isempty(factors) && return -Inf
    p = reduce((a, b) -> a .* b, (f.data for f in factors))
    dot(H.data, p) / length(H.data)
end

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — hierarchical encode/descend (paper §4.1, Alg 1 & 2; Eq 11)
# ─────────────────────────────────────────────────────────────────────────────
"Role atom sets per level: state role `τ_t`, child-slot roles `r_{t,i}`, summary `σ_t`."
struct Roles
    tau::Vector{HV{BipolarMAP}}
    r::Vector{Vector{HV{BipolarMAP}}}
    sigma::Vector{HV{BipolarMAP}}
    # typed inner ctor: narrows JET inference (no Any-arg union-split exploration)
    Roles(tau::Vector{HV{BipolarMAP}}, r::Vector{Vector{HV{BipolarMAP}}},
        sigma::Vector{HV{BipolarMAP}}) = new(tau, r, sigma)
end

"Generate deterministic random roles for `L+1` levels (0..L) and up to `maxbranch` children."
function make_roles(D::Int, L::Int, maxbranch::Int; rng::AbstractRNG=Random.default_rng())
    tau = [random_hv(BipolarMAP, D, rng) for _ in 0:L]
    r = [[random_hv(BipolarMAP, D, rng) for _ in 1:maxbranch] for _ in 0:L]
    sigma = [random_hv(BipolarMAP, D, rng) for _ in 0:L]
    Roles(tau, r, sigma)
end

"A node: its own prototype `u_ν`, child subtrees, and an optional summary code `S_ν`."
struct VSATree
    u::HV{BipolarMAP}
    children::Vector{VSATree}
    summary::Union{HV{BipolarMAP}, Nothing}
    # typed inner ctor: narrows JET inference (no Any-arg union-split exploration)
    VSATree(u::HV{BipolarMAP}, children::Vector{VSATree},
        summary::Union{HV{BipolarMAP}, Nothing}) = new(u, children, summary)
end
VSATree(u::HV{BipolarMAP}; children::Vector{VSATree}=VSATree[], summary=nothing) =
    VSATree(u, children, summary)

"""ENCODE (Alg 1, Eq 11): `Normalize( τ_t⊗u_ν ⊕ Σ_i r_{t,i}⊗ρ(H_child_i) ⊕ σ_t⊗S_ν )`."""
function encode(node::VSATree, roles::Roles, t::Int=0)
    acc = bind(roles.tau[t + 1], node.u).data
    for (i, c) in enumerate(node.children)
        child = encode(c, roles, t + 1)
        acc = acc .+ bind(roles.r[t + 1][i], permute(child, t + 1)).data
    end
    if node.summary !== nothing
        acc = acc .+ bind(roles.sigma[t + 1], node.summary).data
    end
    proj(HV{BipolarMAP}(acc))
end

"""DESCEND (Alg 2): per step unbind the chosen child role, `ρ⁻¹`, then hard-cleanup
against that level's subtree codebook (line-5 cleanup is load-bearing — §4.2)."""
function descend(H::HV{BipolarMAP}, path::Vector{Int}, roles::Roles,
    level_codebooks::Vector{Codebook{BipolarMAP}})
    h = H
    for (step, i) in enumerate(path)
        t = step - 1
        h = invpermute(unbind(roles.r[t + 1][i], h), t + 1)
        h = cleanup(h, level_codebooks[step])
    end
    h
end

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GATE instrumentation (paper Lemma 2 Eq 17; Thm 3; experiments E1/E4)
# ─────────────────────────────────────────────────────────────────────────────
"""
    cleanup_margin(D, k, M; trials, rng) → empirical P[cleanup fails]

E1 single-step capacity: bundle a target atom with `k-1` independent crosstalk atoms,
hard-clean vs a size-`M` codebook; estimate failure rate. Compare to Lemma 2:
`ℙ[fail] ≤ 2M·exp(−cD/k)`.
"""
function cleanup_margin(D::Int, k::Int, M::Int; trials::Int=1000,
    rng::AbstractRNG=Random.default_rng())
    fails = 0
    for _ in 1:trials
        cb = random_codebook(BipolarMAP, D, M; rng=rng)
        target = rand(rng, 1:M)
        terms = HV{BipolarMAP}[HV{BipolarMAP}(cb.atoms[:, target])]
        for _ in 1:(k - 1)
            push!(terms, random_hv(BipolarMAP, D, rng))
        end
        z = bundle(terms...)
        _cleanup_idx(z, cb) == target || (fails += 1)
    end
    fails / trials
end

"""
    resonator_success_rate(D, factor_sizes; trials, iterations, beta, rng) → (rate, mean_iters)

E4 resonator margin: build a random product `y⋆ = x₁⊗…⊗x_F`, run `factorize`, count
exact recoveries of the full factor tuple. Compare to Thm 3 / Cor 1.
"""
function resonator_success_rate(D::Int, factor_sizes::Vector{Int};
    trials::Int=200, iterations::Int=50, beta::Real=3.0,
    rng::AbstractRNG=Random.default_rng())
    F = length(factor_sizes)
    succ = 0
    for _ in 1:trials
        cbs = [random_codebook(BipolarMAP, D, M; rng=rng) for M in factor_sizes]
        true_idx = [rand(rng, 1:M) for M in factor_sizes]
        factors = [HV{BipolarMAP}(cbs[f].atoms[:, true_idx[f]]) for f in 1:F]
        H = reduce((a, b) -> bind(a, b), factors)
        rec, _ = factorize(H, cbs; iterations=iterations, beta=beta, rng=rng)
        all(rec[f] == factors[f] for f in 1:F) && (succ += 1)
    end
    (succ / trials, Float64(iterations))
end

# ─────────────────────────────────────────────────────────────────────────────
# Phase-2 — MeTTa integration shim (grounded ops over the arena; needs MORK)
# ─────────────────────────────────────────────────────────────────────────────
include("MeTTaShim.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — R-HMH episodic memory (§8); 5a encode + 5b recall + 5-gate instrumentation
# ─────────────────────────────────────────────────────────────────────────────
include("RHMH.jl")

end # module FactorVSA
