# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — R-HMH: Resonant Modular compositional episodic Memory (paper §8).
#
# An episode is a typed factor graph (slots + typed relations + anchors) compiled
# into ONE fixed-width role-filler hypervector H_E (Eq 69), and *resonantly*
# recalled: role-masked unbind + cleanup recovers a slot filler; if the filler is a
# product, the resonator (Step 2 `factorize`) completes it inside the episode (Eq 73).
#
# Built entirely on FactorVSA primitives (bind/bundle/proj/cleanup/factorize). The
# ColBaC-HDC schema (§9) reuses the SAME machinery (Eq 84 IS Eq 11 specialized) — see
# ColBaCHDC.jl (phase 5d, after the gate).
#
# GATE-FIRST: `episode_recovery_rate` instruments the §8.6 admissibility gate. Episodes
# carry MORE crosstalk load (k_slot + k_rel) than the bare resonator, so the 5-gate is a
# real go/no-go — do not build 5c/5d until it passes (see STEP5_BRIEF.md).
# ─────────────────────────────────────────────────────────────────────────────

export RoleBook, role!, Episode, encode_episode, recover_slot, complete_slot
export episode_recovery_rate

"A registry of named role atoms (deterministic per name once created; seed externally)."
mutable struct RoleBook
    dim::Int
    atoms::Dict{Symbol, HV{BipolarMAP}}
end
RoleBook(dim::Int) = RoleBook(dim, Dict{Symbol, HV{BipolarMAP}}())

"Get (or lazily create) the role atom named `name`."
role!(rb::RoleBook, name::Symbol) =
    get!(() -> random_hv(BipolarMAP, rb.dim), rb.atoms, name)

# module-marker role name for a slot's ontology type (m_τ in Eq 69)
_mtype(t::Symbol) = Symbol("mtype_", t)

"""
    Episode

A typed factor graph (paper §8.2, Eq 68):
- `schema`    — schema / event-type code `h_σ(E)`
- `slots`     — `role => (type, filler)`   (role `r_s`, ontology type `τ`, filler `h_{f_s}`)
- `relations` — `(relrole, role_i, role_j)` typed relation between two slots' fillers
- `anchors`   — `anchorrole => payload`     (links to external evidence)
"""
struct Episode
    schema::HV{BipolarMAP}
    slots::Dict{Symbol, Tuple{Symbol, HV{BipolarMAP}}}
    relations::Vector{Tuple{Symbol, Symbol, Symbol}}
    anchors::Dict{Symbol, HV{BipolarMAP}}
    # typed inner ctor: narrows JET inference (no Any-arg union-split exploration)
    Episode(schema::HV{BipolarMAP}, slots::Dict{Symbol, Tuple{Symbol, HV{BipolarMAP}}},
        relations::Vector{Tuple{Symbol, Symbol, Symbol}},
        anchors::Dict{Symbol, HV{BipolarMAP}}) = new(schema, slots, relations, anchors)
end
function Episode(schema::HV{BipolarMAP};
    slots::Dict{Symbol, Tuple{Symbol, HV{BipolarMAP}}}=Dict{
        Symbol, Tuple{Symbol, HV{BipolarMAP}}
    }(),
    relations::Vector{Tuple{Symbol, Symbol, Symbol}}=Tuple{Symbol, Symbol, Symbol}[],
    anchors::Dict{Symbol, HV{BipolarMAP}}=Dict{Symbol, HV{BipolarMAP}}())
    Episode(schema, slots, relations, anchors)
end

# ── 5a — encode (Eq 69) ──────────────────────────────────────────────────────
"""
    encode_episode(E, rb) → H_E

R-HMH episode code (Eq 69):
`Normalize( r_schema⊗h_σ ⊕ Σ_s m_τ(s)⊗r_s⊗h_{f_s} ⊕ Σ_{(ρ,i,j)} r_ρ⊗r_i⊗r_j⊗(h_{f_i}⊗h_{f_j})
            ⊕ Σ_a r_anchor(a)⊗h_a )`.
"""
function encode_episode(E::Episode, rb::RoleBook)
    acc = bind(role!(rb, :schema), E.schema).data
    for (rs, (τ, f)) in E.slots
        acc = acc .+ bind(role!(rb, _mtype(τ)), bind(role!(rb, rs), f)).data
    end
    for (rρ, ri, rj) in E.relations
        fi = E.slots[ri][2]
        fj = E.slots[rj][2]
        acc =
            acc .+
            bind(role!(rb, rρ), bind(role!(rb, ri), bind(role!(rb, rj), bind(fi, fj)))).data
    end
    for (ra, h) in E.anchors
        acc = acc .+ bind(role!(rb, ra), h).data
    end
    proj(HV{BipolarMAP}(acc))
end

# ── 5b — resonant recall / slot completion (Eq 70-73) ────────────────────────
"""
    recover_slot(H_E, role, type, rb, codebook) → HV

Role-masked unbind of slot `role` (type `type`) from `H_E`, then hard-cleanup against
the slot's codebook (other slots/relations act as bounded crosstalk). BipolarMAP: bind
is self-inverse, so `bind(m_τ, bind(r_s, H_E))` exposes the filler.
"""
recover_slot(H_E::HV{BipolarMAP}, role::Symbol, type::Symbol, rb::RoleBook,
    cb::Codebook{BipolarMAP}) =
    cleanup(bind(role!(rb, _mtype(type)), bind(role!(rb, role), H_E)), cb)

"""
    complete_slot(H_E, role, type, rb, factor_codebooks) → (factors, score)

When a slot filler is itself a product `h_{f_s} ≈ c_{s,1}⊗…⊗c_{s,F}` (Eq 72), recover the
noisy filler then run the resonator INSIDE the episode (Eq 73) to recover its factors.
"""
complete_slot(H_E::HV{BipolarMAP}, role::Symbol, type::Symbol, rb::RoleBook,
    factor_cbs::Vector{Codebook{BipolarMAP}}) =
    factorize(
        bind(role!(rb, _mtype(type)), bind(role!(rb, role), H_E)), factor_cbs; restarts=3
    )

# ── 5-GATE instrumentation (§8.6 admissibility) ──────────────────────────────
"""
    episode_recovery_rate(D, k_slot, k_rel, M; trials, rng) → mean slot-recovery accuracy

Build random episodes with `k_slot` typed slots (each filler an atom from its own size-`M`
codebook) and `k_rel` random typed relations; encode (Eq 69); recover every slot by
role-masked cleanup; return the mean fraction recovered correctly. The gate sweeps `D`,
`k_slot`, and `k_rel` (the last INDEPENDENTLY — relations are heavier crosstalk).
"""
function episode_recovery_rate(D::Int, k_slot::Int, k_rel::Int, M::Int;
    trials::Int=100, rng::AbstractRNG=Random.default_rng())
    total = 0.0
    for _ in 1:trials
        rb = RoleBook(D)
        cbs = Dict{Symbol, Codebook{BipolarMAP}}()
        truth = Dict{Symbol, Int}()
        slots = Dict{Symbol, Tuple{Symbol, HV{BipolarMAP}}}()
        roles = Symbol[]
        for s in 1:k_slot
            r = Symbol("slot", s)
            τ = Symbol("type", s)
            cb = random_codebook(BipolarMAP, D, M; rng=rng)
            idx = rand(rng, 1:M)
            cbs[r] = cb
            truth[r] = idx
            slots[r] = (τ, HV{BipolarMAP}(cb.atoms[:, idx]))
            push!(roles, r)
        end
        rels = Tuple{Symbol, Symbol, Symbol}[]
        if k_slot >= 2
            for t in 1:k_rel
                i, j = rand(rng, roles), rand(rng, roles)
                push!(rels, (Symbol("rel", t), i, j))
            end
        end
        E = Episode(random_hv(BipolarMAP, D, rng); slots=slots, relations=rels)
        H = encode_episode(E, rb)
        correct = 0
        for r in roles
            τ = slots[r][1]
            got = recover_slot(H, r, τ, rb, cbs[r])
            got == HV{BipolarMAP}(cbs[r].atoms[:, truth[r]]) && (correct += 1)
        end
        total += correct / k_slot
    end
    total / trials
end
