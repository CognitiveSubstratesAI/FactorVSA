# ─────────────────────────────────────────────────────────────────────────────
# Step 5d — ColBaC-HDC representation layer (paper §9, Eq 84-90).
#
# The HDC layer a ColBaC/HBCML learner CONSUMES — "ColBaC learns causal columns; HDC makes
# their reusable contents addressable, compositional, and auditable" (Eq 83). The paper
# states Eq 84 (column hypervector) IS Eq 11 (the hierarchical encoder) specialized — so this
# reuses the SAME typed bind/bundle/cleanup machinery as R-HMH (RHMH.jl), just a columnar
# schema (microcolumn-type u ∈ {K,L,B} × shell-tier s ∈ 0..3 × local-role a).
#
# OUT OF SCOPE (the NGC/FabricPC line, not this package): the ColBaC LEARNER itself
# (columns/shells/teachers/Bayesian-causal training). This is encoders + audit structures
# on a GENERIC column data type; tested on synthetic columns. The score (Eq 88) and the
# promote/demote rule (Eq 89-90) combine these HDC quantities with learner signals (Δ_fact),
# so only the HDC-computable parts (Δ, M_conf, reuse) live here.
# ─────────────────────────────────────────────────────────────────────────────

export Column, encode_column, recover_motif, support_code, certificate
export cleanup_margin_of, confusability, reuse_score, promote_signal

# role names for the columnar schema
_urole(u::Symbol) = Symbol("u_", u)
_srole(s::Integer) = Symbol("s_", s)
_arole(a::Symbol) = Symbol("a_", a)

"""
    Column

A ColBaC column's active motifs (§9.3), each triple-tagged: microcolumn type `u ∈ {:K,:L,:B}`,
shell tier `s ∈ 0:3` (0 = hard kernel), local role `a`, and the motif hypervector.
"""
struct Column
    motifs::Vector{Tuple{Symbol, Int, Symbol, HV{BipolarMAP}}}
    Column(motifs::Vector{Tuple{Symbol, Int, Symbol, HV{BipolarMAP}}}) = new(motifs)
end

"""
    encode_column(C, rb) → H_m

Column hypervector (Eq 84): `Normalize( Σ_{u,s,a} r_u ⊗ r_s ⊗ r_a ⊗ c_{m,u,s,a} )`.
Same unbind-and-cleanup machinery as the R-HMH encoder reads any tagged piece.
"""
function encode_column(C::Column, rb::RoleBook)
    acc = zeros(Float64, rb.dim)
    for (u, s, a, m) in C.motifs
        acc =
            acc .+
            bind(
                role!(rb, _urole(u)),
                bind(role!(rb, _srole(s)), bind(role!(rb, _arole(a)), m))
            ).data
    end
    proj(HV{BipolarMAP}(acc))
end

"Recover a tagged motif from a column code (role-masked unbind + cleanup), like `recover_slot`."
recover_motif(H_m::HV{BipolarMAP}, u::Symbol, s::Integer, a::Symbol, rb::RoleBook,
    cb::Codebook{BipolarMAP}) =
    cleanup(
        bind(
            role!(rb, _urole(u)),
            bind(role!(rb, _srole(s)), bind(role!(rb, _arole(a)), H_m))
        ),
        cb
    )

"""
    support_code(encoded, support, rb) → H_S

Active-support code (Eq 85): `Normalize( Σ_{m∈S} r_m ⊗ H_m )` over the encoded columns in
`support` (`encoded[m]` is column `m`'s `H_m`). Used for support retrieval / context memory.
"""
function support_code(
    encoded::Dict{Int, HV{BipolarMAP}}, support::Vector{Int}, rb::RoleBook
)
    acc = zeros(Float64, rb.dim)
    for m in support
        acc = acc .+ bind(role!(rb, Symbol("col_", m)), encoded[m]).data
    end
    proj(HV{BipolarMAP}(acc))
end

"""
    certificate(channels, H_m, rb) → Z_m

Structured certificate hypervector (Eq 86-87): `r_signature⊗H_m ⊕ Σ_ch r_ch⊗z_ch` over the
certificate channels (`:shared`, `:specific`, `:saturation`, `:demotion`, …). Each channel is
recovered by unbinding `r_ch` and cleaning against that channel's codebook.
"""
function certificate(
    channels::Dict{Symbol, HV{BipolarMAP}}, H_m::HV{BipolarMAP}, rb::RoleBook
)
    acc = bind(role!(rb, :cert_signature), H_m).data
    for (ch, z) in channels
        acc = acc .+ bind(role!(rb, Symbol("cert_", ch)), z).data
    end
    proj(HV{BipolarMAP}(acc))
end

# ── HDC audit quantities (the HDC-computable inputs to shell hygiene, Eq 89-90) ──
"Cleanup margin Δ (Lemma 1): `⟨c⁺,z⟩ − max_{c≠c⁺}⟨c,z⟩`, normalized by D. Higher = cleaner."
function cleanup_margin_of(z::HV{BipolarMAP}, cb::Codebook{BipolarMAP})
    sc = cb.atoms' * z.data
    p = sortperm(sc; rev=true)
    (sc[p[1]] - sc[p[2]]) / length(z.data)
end

"Confusability M_conf (Eq 54): `#{c : ⟨c,z⟩ ≥ ⟨c⁺,z⟩ − γ·D}`. 1 = unambiguous; higher = more confusable."
function confusability(z::HV{BipolarMAP}, cb::Codebook{BipolarMAP}; gamma::Real=0.05)
    sc = cb.atoms' * z.data
    cmax = maximum(sc)
    count(x -> x >= cmax - gamma * length(z.data), sc)
end

"Reuse signal: max |cosine| of motif `m` against `others` (cross-column shared-abstraction mass)."
reuse_score(m::HV{BipolarMAP}, others::Vector{HV{BipolarMAP}}) =
    if isempty(others)
        0.0
    else
        maximum(abs(dot(m.data, o.data)) / length(m.data) for o in others)
    end

"""
    promote_signal(reuse, margin, m_conf) → Float64

HDC side of the shell-hygiene promote rule (Eq 89): monotone ↑ in reuse and cleanup margin,
↓ in confusability. The LEARNER combines this with its factorization defect `Δ_fact` to decide
promotion/demotion — that combination is out of scope here (NGC line).
"""
promote_signal(reuse::Real, margin::Real, m_conf::Integer) = reuse + margin - log(m_conf)
