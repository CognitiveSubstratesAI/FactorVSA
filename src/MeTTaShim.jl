# ─────────────────────────────────────────────────────────────────────────────
# Phase-2 MeTTa integration shim — grounded ops over an arena, handle-referenced.
#
# UPSTREAM-ALIGNED (cross-checked 2026-06-06 vs dev-zone MORK kernel+server+docs):
# MORK has no grounding of its own; grounding is the consuming runtime's layer,
# realized in the Julia port as `register_grounded!` + `GroundedSource` (an
# I-pattern source — matching the roadmap's "grounded ops inverted into queries").
# `asource_new` auto-routes any registered name to GroundedSource, so this needs
# ZERO MORK changes.
#
# Dense vectors NEVER cross into the trie: they live in a process-global FactorVSA
# arena (the Step-0 DualIndex) and are referenced from MeTTa only by a tiny handle
# string `(VecRef h)`. Grounded handlers take/return s-expression strings — for
# dense results they store into the arena and return a fresh `(VecRef h')`.
#
# NOTE: codebook-dependent ops (cleanup / resonate / factorize) need a CodebookRef
# registry — deferred to phase-2b (marked below). This shim covers create + algebra
# + similarity, which exercise the full handle round-trip.
# ─────────────────────────────────────────────────────────────────────────────

using MORK: register_grounded!, is_grounded, GROUNDED_REGISTRY

export FVSA_ARENA, register_factorvsa!, unregister_factorvsa!, vecref, parse_vecref

# Process-global arena (Step-0 DualIndex). Id == the VecRef integer.
const FVSA_ARENA = DualIndex{Int, HV{BipolarMAP}}()
const _FVSA_NEXT_ID = Ref(0)

"Store `hv` in the arena, return its integer handle id."
function _fvsa_store!(hv::HV{BipolarMAP})::Int
    id = (_FVSA_NEXT_ID[] += 1)
    insert_vector!(FVSA_ARENA, id, hv)
    id
end

"Render a handle id as the MeTTa s-expression `(VecRef h)`."
vecref(id::Integer) = "(VecRef $id)"

"""
    parse_vecref(s) → Int

Parse a `(VecRef h)` s-expression string to its integer handle. Throws on a
malformed handle (fail-loud — a grounded op must not silently mis-route).
"""
function parse_vecref(s::AbstractString)
    m = match(r"^\(\s*VecRef\s+(-?\d+)\s*\)$", strip(s))
    m === nothing && error("FactorVSA: not a (VecRef h) handle: $(repr(s))")
    cap = m.captures[1]                      # Union{Nothing,SubString}; narrow before parse
    cap === nothing && error("FactorVSA: malformed (VecRef h): $(repr(s))")
    parse(Int, cap)
end

_fvsa_get(s::AbstractString)::HV{BipolarMAP} = begin
    v = lookup_vector(FVSA_ARENA, parse_vecref(s))
    v === nothing &&
        error("FactorVSA: dangling handle $(repr(s)) (freed or never stored)")
    v
end

"""
    register_factorvsa!()

Register FactorVSA's grounded ops into MORK's `GROUNDED_REGISTRY` so they are
callable from MeTTa as `(fvsa-* …)` in an I-pattern position. Idempotent.

Ops (handle = `(VecRef h)` string):
- `(fvsa-random D)`            → `(VecRef h)`   random ±1 hypervector of dim D
- `(fvsa-bind  a b)`           → `(VecRef h)`   bind ⊗ (stored)
- `(fvsa-unbind a b)`          → `(VecRef h)`   unbind
- `(fvsa-bundle a b …)`        → `(VecRef h)`   bundle ⊕ then proj
- `(fvsa-sim   a b)`           → scalar string  cosine similarity ⟨a,b⟩/D
"""
function register_factorvsa!()
    register_grounded!(
        "fvsa-random",
        args -> begin
            isempty(args) && error("fvsa-random: needs a dimension D")
            D = parse(Int, strip(args[1]))
            vecref(_fvsa_store!(random_hv(BipolarMAP, D)))
        end
    )
    register_grounded!(
        "fvsa-bind",
        args -> begin
            length(args) == 2 || error("fvsa-bind: needs exactly 2 handles")
            vecref(_fvsa_store!(bind(_fvsa_get(args[1]), _fvsa_get(args[2]))))
        end
    )
    register_grounded!(
        "fvsa-unbind",
        args -> begin
            length(args) == 2 || error("fvsa-unbind: needs exactly 2 handles")
            vecref(_fvsa_store!(unbind(_fvsa_get(args[1]), _fvsa_get(args[2]))))
        end
    )
    register_grounded!(
        "fvsa-bundle",
        args -> begin
            length(args) >= 1 || error("fvsa-bundle: needs ≥1 handle")
            hvs = [_fvsa_get(a) for a in args]
            vecref(_fvsa_store!(proj(bundle(hvs...))))
        end
    )
    register_grounded!(
        "fvsa-sim",
        args -> begin
            length(args) == 2 || error("fvsa-sim: needs exactly 2 handles")
            a = _fvsa_get(args[1])
            b = _fvsa_get(args[2])
            string(dot(a.data, b.data) / length(a.data))
        end
    )
    # phase-2b (deferred): fvsa-cleanup / fvsa-resonate need a CodebookRef registry.
    nothing
end

"Remove FactorVSA's grounded ops from the registry (for test isolation / teardown)."
function unregister_factorvsa!()
    for k in ("fvsa-random", "fvsa-bind", "fvsa-unbind", "fvsa-bundle", "fvsa-sim")
        delete!(GROUNDED_REGISTRY, k)
    end
    nothing
end
