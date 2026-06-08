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
# Covers base algebra (create / bind / bundle / sim) AND the resonator (phase-2b):
# codebooks are immutable `(CodebookRef c)` handles; cleanup/resonate take a codebook
# handle + produce standalone results — so this exposes the package's headline
# capability (resonator factorization) to MeTTa, not just base VSA.
# ─────────────────────────────────────────────────────────────────────────────

using MORK: register_grounded!, is_grounded, GROUNDED_REGISTRY

export FVSA_ARENA, FVSA_CODEBOOKS, register_factorvsa!, unregister_factorvsa!
export vecref, parse_vecref, cbref, parse_cbref

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

# Codebook registry (phase-2b). Codebooks are IMMUTABLE handle-referenced objects:
# a `(CodebookRef c)` never mutates; "change" = register a new handle. Monotonic ids
# (never reused → no ABA), fail-loud lookup. See the codebook_version note in
# FactorVSA.jl for why this decouples from DualIndex.codebook_version.
const FVSA_CODEBOOKS = Dict{Int, Codebook{BipolarMAP}}()
const _FVSA_CB_NEXT_ID = Ref(0)

function _fvsa_store_cb!(cb::Codebook{BipolarMAP})::Int
    id = (_FVSA_CB_NEXT_ID[] += 1)
    FVSA_CODEBOOKS[id] = cb
    id
end

"Render a codebook id as the MeTTa s-expression `(CodebookRef c)`."
cbref(id::Integer) = "(CodebookRef $id)"

"Parse a `(CodebookRef c)` s-expression to its integer id (fail-loud on malformed)."
function parse_cbref(s::AbstractString)
    m = match(r"^\(\s*CodebookRef\s+(-?\d+)\s*\)$", strip(s))
    m === nothing && error("FactorVSA: not a (CodebookRef c) handle: $(repr(s))")
    cap = m.captures[1]
    cap === nothing && error("FactorVSA: malformed (CodebookRef c): $(repr(s))")
    parse(Int, cap)
end

_fvsa_get_cb(s::AbstractString)::Codebook{BipolarMAP} = begin
    cb = get(FVSA_CODEBOOKS, parse_cbref(s), nothing)
    cb === nothing && error("FactorVSA: dangling codebook handle $(repr(s))")
    cb
end

"""
    register_factorvsa!()

Register FactorVSA's grounded ops into MORK's `GROUNDED_REGISTRY` so they are
callable from MeTTa as `(fvsa-* …)` in an I-pattern position. Idempotent.

Base algebra (handle = `(VecRef h)`):
- `(fvsa-random D)`            → `(VecRef h)`   random ±1 hypervector of dim D
- `(fvsa-bind  a b)`           → `(VecRef h)`   bind ⊗ (stored)
- `(fvsa-unbind a b)`          → `(VecRef h)`   unbind
- `(fvsa-bundle a b …)`        → `(VecRef h)`   bundle ⊕ then proj
- `(fvsa-sim   a b)`           → scalar string  cosine similarity ⟨a,b⟩/D

Phase-2b — codebooks + the resonator (handle = `(CodebookRef c)`):
- `(fvsa-codebook D M)`              → `(CodebookRef c)`  random size-M codebook, dim D
- `(fvsa-codebook-atom c i)`         → `(VecRef h)`       i-th atom of codebook c (1-based)
- `(fvsa-cleanup z c)`               → `(VecRef h)`       nearest codebook atom to z
- `(fvsa-resonate H c1 c2 …)`        → `((VecRef h1) …)`  recovered factor tuple (Alg 3)
- `(fvsa-recompose-score H f1 f2 …)` → scalar string      ⟨H, f1⊗…⟩/D (reject spurious)
- `(fvsa-free-codebook c)`           → `()`               drop the codebook handle
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
    # ── phase-2b: codebooks + the resonator (the package's headline capability) ──
    register_grounded!(
        "fvsa-codebook",
        args -> begin
            length(args) == 2 || error("fvsa-codebook: needs D and M")
            D = parse(Int, strip(args[1]))
            M = parse(Int, strip(args[2]))
            cbref(_fvsa_store_cb!(random_codebook(BipolarMAP, D, M)))
        end
    )
    register_grounded!(
        "fvsa-codebook-atom",
        args -> begin
            length(args) == 2 ||
                error("fvsa-codebook-atom: needs (CodebookRef c) and index i")
            cb = _fvsa_get_cb(args[1])
            i = parse(Int, strip(args[2]))
            (1 <= i <= length(cb)) ||
                error("fvsa-codebook-atom: index $i out of range 1..$(length(cb))")
            vecref(_fvsa_store!(HV{BipolarMAP}(cb.atoms[:, i])))
        end
    )
    register_grounded!(
        "fvsa-cleanup",
        args -> begin
            length(args) == 2 ||
                error("fvsa-cleanup: needs (VecRef z) and (CodebookRef c)")
            vecref(_fvsa_store!(cleanup(_fvsa_get(args[1]), _fvsa_get_cb(args[2]))))
        end
    )
    register_grounded!(
        "fvsa-resonate",
        args -> begin
            length(args) >= 2 ||
                error("fvsa-resonate: needs (VecRef H) and ≥1 (CodebookRef c)")
            H = _fvsa_get(args[1])
            cbs = [_fvsa_get_cb(a) for a in args[2:end]]
            factors, _ = factorize(H, cbs; restarts=3)   # multi-start (paper §4.3 robustness)
            # one s-expr list of the recovered factor handles
            "(" * join((vecref(_fvsa_store!(f)) for f in factors), " ") * ")"
        end
    )
    register_grounded!(
        "fvsa-recompose-score",
        args -> begin
            length(args) >= 2 ||
                error("fvsa-recompose-score: needs (VecRef H) and ≥1 factor handle")
            H = _fvsa_get(args[1])
            fs = [_fvsa_get(a) for a in args[2:end]]
            string(recompose_score(H, fs))
        end
    )
    register_grounded!(
        "fvsa-free-codebook",
        args -> begin
            length(args) == 1 || error("fvsa-free-codebook: needs (CodebookRef c)")
            delete!(FVSA_CODEBOOKS, parse_cbref(args[1]))
            "()"
        end
    )
    nothing
end

"Remove FactorVSA's grounded ops from the registry (for test isolation / teardown)."
function unregister_factorvsa!()
    for k in ("fvsa-random", "fvsa-bind", "fvsa-unbind", "fvsa-bundle", "fvsa-sim",
        "fvsa-codebook", "fvsa-codebook-atom", "fvsa-cleanup", "fvsa-resonate",
        "fvsa-recompose-score", "fvsa-free-codebook")
        delete!(GROUNDED_REGISTRY, k)
    end
    nothing
end
