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

using MORK: register_grounded!, is_grounded, GROUNDED_REGISTRY, sexpr_to_expr

export FVSA_ARENA, FVSA_CODEBOOKS, register_factorvsa!, unregister_factorvsa!
export vecref, parse_vecref, cbref, parse_cbref
export FVSA_EMBED, FVSA_EMBED_DIM, content_id, content_hv   # dual index: MORK content-id ↔ vector

# Process-global arena (Step-0 DualIndex). Id == the VecRef integer.
# Concurrency: FVSA_ARENA is internally lock-guarded (see DualIndex). The id counter is a
# Threads.Atomic so concurrent grounded-op eval can't hand two stores the same id (a
# non-atomic `+= 1` loses updates → aliased handles). MORK eval is single-threaded today,
# so this is latent — but the arena is process-global, so the moment anything parallelizes
# it must hold. atomic_add! returns the OLD value, hence `+ 1` for the fresh monotonic id.
const FVSA_ARENA = DualIndex{Int, HV{BipolarMAP}}()
const _FVSA_NEXT_ID = Threads.Atomic{Int}(0)

"Store `hv` in the arena, return its integer handle id."
function _fvsa_store!(hv::HV{BipolarMAP})::Int
    id = Threads.atomic_add!(_FVSA_NEXT_ID, 1) + 1
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

# ── DUAL INDEX: MORK content-id ↔ vector (per Hyperon WP §2/§7.8 — symbolic and vector
# representations share ONE content-addressed identity). MORK has NO separate CID; an atom's
# identity IS its canonical serialized byte path (`sexpr_to_expr(atom).buf`) — verified vs the
# MORK kernel (Frontend.jl:232 / Expr.jl). We key by the FULL bytes, so identity is EXACT
# (content-addressed, no hash → no collision question). `fvsa-embed` is the seam: the same atom
# always maps to the same vector slot, making the FactorVSA index genuinely dual to MORK's
# symbolic trie. (Scratch vectors from bind/bundle keep allocation-identity handles — a
# deliberately DIFFERENT identity model; see the codebook/arena notes.)
const FVSA_EMBED = Dict{Vector{UInt8}, Int}()   # canonical atom bytes → VecRef id
const _FVSA_EMBED_LOCK = ReentrantLock()        # guards FVSA_EMBED (get-or-create is atomic)
const FVSA_EMBED_DIM = Ref(1024)                # shared dim for atom embeddings (must be one D)

"""
    content_id(sexpr) -> Vector{UInt8}

MORK's content-identity for an atom: its MM2 (`Expr`, "Rule of 64") serialized byte buffer.
Deterministic and whitespace-canonical (`sexpr_to_expr` re-parses then re-serializes), so
` dog ` and `dog` collapse to one id, while `dog` (symbol) and `(dog)` (1-expr) stay distinct.

ALIGNMENT WITH CORE (verified vs CoreSpace.jl:352-364): Core stores an atom in its MORK trie at
`space.prefix ++ sexpr_to_expr(to_sexpr(atom)).buf`. So for a GROUND atom in the root space this
id is BYTE-IDENTICAL to Core's atom identity — the dual index lines up exactly. Two caveats by
design: (1) we exclude the space `prefix` (embeddings are content-addressed by the ATOM, not its
scoped location — `dog`'s vector is the same in every space); (2) atoms with `\$x` variables only
align if passed in Core's `__var_x` storage form (MORK would otherwise anonymize the name as a de
Bruijn NewVar). Embeddings target ground/named atoms, so both caveats are moot in practice.
"""
content_id(sexpr::AbstractString)::Vector{UInt8} = sexpr_to_expr(sexpr).buf

# ── Compositional encoding for COMPOUND atoms (HDC paper §4.1, Eq 11) ──────────
# A compound atom's vector is BUILT from its parts' embeddings via encode(VSATree, Roles), so it
# is VSA-decodable (descend/unbind recovers the parts) — not an opaque seed. A LEAF symbol keeps a
# random content-addressed seed; a compound `(h a b)` maps to a VSATree (a fixed expr-node marker
# at each internal node, leaf children = their content-addressed embeddings) and is encoded once at
# the root. Roles + marker are deterministic (fixed seed) so the encoding is reproducible within a
# process. Depth/branch are bounded by the role table; beyond it we fall back to an opaque seed
# (logged) rather than silently mis-encode.
const _EMBED_L_MAX = 8           # max tree depth covered by the role table
const _EMBED_BRANCH_MAX = 16     # max children per node covered by the role table
const _EMBED_ROLES = Ref{Union{Roles, Nothing}}(nothing)
const _EMBED_MARKER = Ref{Union{HV{BipolarMAP}, Nothing}}(nothing)
const _EMBED_ROLES_DIM = Ref(0)

"Deterministic (role table, expr-node marker) for the current embedding dim; rebuilt if dim changes."
function _embed_roles()
    D = FVSA_EMBED_DIM[]
    if _EMBED_ROLES[] === nothing || _EMBED_ROLES_DIM[] != D
        _EMBED_ROLES[] = make_roles(D, _EMBED_L_MAX, _EMBED_BRANCH_MAX;
            rng = Random.MersenneTwister(0xFAC70_01))
        _EMBED_MARKER[] = random_hv(BipolarMAP, D, Random.MersenneTwister(0xFAC70_E0))
        _EMBED_ROLES_DIM[] = D
    end
    _EMBED_ROLES[]::Roles, _EMBED_MARKER[]::HV{BipolarMAP}
end

# minimal s-expression parser: string → (String leaf | Vector{Any} compound)
function _sexpr_tokens(s::AbstractString)
    toks = String[]; buf = IOBuffer()
    flush_buf() = (str = String(take!(buf)); isempty(str) || push!(toks, str))
    for c in s
        if c == '(' || c == ')'
            flush_buf(); push!(toks, string(c))
        elseif isspace(c)
            flush_buf()
        else
            print(buf, c)
        end
    end
    flush_buf(); toks
end
function _parse_node(toks::Vector{String}, pos::Base.RefValue{Int})
    tok = toks[pos[]]
    if tok == "("
        pos[] += 1; node = Any[]
        while pos[] <= length(toks) && toks[pos[]] != ")"
            push!(node, _parse_node(toks, pos))
        end
        pos[] += 1                      # consume ")"
        return node
    else
        pos[] += 1
        return tok                      # leaf
    end
end
function _parse_sexpr(s::AbstractString)
    toks = _sexpr_tokens(s)
    isempty(toks) && return ""          # empty input → empty leaf
    _parse_node(toks, Ref(1))
end

_struct_depth(::AbstractString) = 0
_struct_depth(x::Vector) = isempty(x) ? 1 : 1 + maximum(_struct_depth, x)
_struct_maxbranch(::AbstractString) = 0
_struct_maxbranch(x::Vector) = max(length(x), isempty(x) ? 0 : maximum(_struct_maxbranch, x))

# parsed node → VSATree (leaves resolve to their content-addressed embeddings — so a nested `dog`
# shares the SAME vector as a standalone `(fvsa-embed dog)`; internal nodes use the expr marker).
_to_vsatree(x::AbstractString) = VSATree(lookup_vector(FVSA_ARENA, _embed_id(x))::HV{BipolarMAP})
function _to_vsatree(x::Vector)
    _, marker = _embed_roles()
    VSATree(marker, VSATree[_to_vsatree(e) for e in x], nothing)
end

# Stable seed from content-id bytes — FNV-1a/64. Unlike Base.hash (version-dependent, no guarantee),
# this is a fixed pure function of the bytes → identical seed across processes, Julia versions, and
# machines. So a leaf's vector is a pure function of its atom content (the arena is regeneratable).
function _stable_seed(bytes::Vector{UInt8})::UInt64
    h = 0xcbf29ce484222325                  # FNV-1a 64-bit offset basis
    for b in bytes
        h = (h ⊻ UInt64(b)) * 0x100000001b3  # XOR then prime; wraps mod 2^64
    end
    h
end

# A leaf's HV is a DETERMINISTIC random ±1 vector seeded by its content-id — same atom → same vector
# everywhere, with no shared state. Distinct ids → independent seeds → near-orthogonal vectors.
_seeded_leaf_hv(cid::Vector{UInt8}, D::Int) =
    random_hv(BipolarMAP, D, Random.MersenneTwister(_stable_seed(cid)))

"""
    content_hv(name, D) -> HV{BipolarMAP}

Deterministic, CONTENT-seeded hypervector for a name/symbol/string: the same name maps to the same
vector in every process (FNV-1a64 of the name's bytes → MersenneTwister seed), distinct names to
near-orthogonal vectors. Use for role / filler / schema atoms so R-HMH episodes are reproducible and
shareable across processes (WP §0 reproducible twins, §5.11.3 shared IDs). Contrast `random_hv`, whose
output depends on global-RNG draw order, not content — the source of the RoleBook non-reproducibility.
"""
content_hv(name, D::Int) = _seeded_leaf_hv(Vector{UInt8}(string(name)), D)

# compound → compositional encode (bounded); deterministic given the (deterministic) leaf seeds +
# the fixed-seed role table & marker, so the whole encoding is content-derived and reproducible.
function _compound_hv(node::Vector, D::Int)
    if _struct_depth(node) > _EMBED_L_MAX || _struct_maxbranch(node) > _EMBED_BRANCH_MAX
        @warn "fvsa-embed: atom exceeds encode bounds — opaque seed (not decodable)" maxlog=3
        return _seeded_leaf_hv(Vector{UInt8}(string(node)), D)   # still deterministic by structure
    end
    roles, _ = _embed_roles()
    encode(_to_vsatree(node), roles)
end

"""
Content-addressed atom embedding (the dual-index seam). Returns the SAME `(VecRef h)` for the same
atom on every call — keyed by MORK content-id — so one identity addresses both the symbolic atom
(MORK trie) and its vector (FactorVSA arena). Idempotent; the lock makes get-or-create atomic
(reentrant, so nested leaf embedding composes; lock order embed→arena, never reversed).

DETERMINISTIC / content-derived: a LEAF's vector is a random ±1 HV seeded by a STABLE hash of its
content-id (FNV-1a), and a COMPOUND is COMPOSITIONALLY encoded from its parts (`encode(VSATree,
Roles)`, HDC paper §4.1/Eq 11) over a fixed-seed role table. So the embedding is a PURE FUNCTION of
atom content — identical across processes/machines, and the arena is regeneratable (no persistence
needed). The cache (`FVSA_EMBED`) is therefore a memoization, not the source of identity. Compounds
are VSA-decodable (unbind a part's role to recover it); structure-similar atoms get similar codes;
nested leaves are shared. Bounded depth ≤ $_EMBED_L_MAX / branch ≤ $_EMBED_BRANCH_MAX (fallback logged).
"""
_fvsa_embed!(sexpr::AbstractString, D::Int=FVSA_EMBED_DIM[])::Int = _embed_id(sexpr, D)

function _embed_id(sexpr::AbstractString, D::Int=FVSA_EMBED_DIM[])::Int
    cid = content_id(sexpr)
    @lock _FVSA_EMBED_LOCK begin
        existing = get(FVSA_EMBED, cid, nothing)
        existing !== nothing && return existing
        node = _parse_sexpr(sexpr)
        hv = node isa Vector ? _compound_hv(node, D) : _seeded_leaf_hv(cid, D)
        id = _fvsa_store!(hv)
        FVSA_EMBED[cid] = id
        return id
    end
end

# Codebook registry (phase-2b). Codebooks are IMMUTABLE handle-referenced objects:
# a `(CodebookRef c)` never mutates; "change" = register a new handle. Monotonic ids
# (never reused → no ABA), fail-loud lookup. See the codebook_version note in
# FactorVSA.jl for why this decouples from DualIndex.codebook_version.
const FVSA_CODEBOOKS = Dict{Int, Codebook{BipolarMAP}}()
const _FVSA_CB_NEXT_ID = Threads.Atomic{Int}(0)
const _FVSA_CB_LOCK = ReentrantLock()   # guards the FVSA_CODEBOOKS dict (store/get/delete)

function _fvsa_store_cb!(cb::Codebook{BipolarMAP})::Int
    id = Threads.atomic_add!(_FVSA_CB_NEXT_ID, 1) + 1
    @lock _FVSA_CB_LOCK (FVSA_CODEBOOKS[id] = cb)
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
    cb = @lock _FVSA_CB_LOCK get(FVSA_CODEBOOKS, parse_cbref(s), nothing)
    cb === nothing && error("FactorVSA: dangling codebook handle $(repr(s))")
    cb
end

"""
    register_factorvsa!()

Register FactorVSA's grounded ops into MORK's `GROUNDED_REGISTRY` so they are
callable from MeTTa as `(fvsa-* …)` in an I-pattern position. Idempotent.

Base algebra (handle = `(VecRef h)`):
- `(fvsa-embed atom [D])`      → `(VecRef h)`   CONTENT-ADDRESSED embedding of an atom — same
                                                atom ⇒ same handle (the dual index ↔ MORK id)
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
        "fvsa-embed",
        args -> begin
            (1 <= length(args) <= 2) ||
                error("fvsa-embed: needs an atom and an optional dim D")
            D = length(args) == 2 ? parse(Int, strip(args[2])) : FVSA_EMBED_DIM[]
            vecref(_fvsa_embed!(args[1], D))
        end
    )
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
            @lock _FVSA_CB_LOCK delete!(FVSA_CODEBOOKS, parse_cbref(args[1]))
            "()"
        end
    )
    nothing
end

"Remove FactorVSA's grounded ops from the registry (for test isolation / teardown)."
function unregister_factorvsa!()
    for k in ("fvsa-embed", "fvsa-random", "fvsa-bind", "fvsa-unbind", "fvsa-bundle", "fvsa-sim",
        "fvsa-codebook", "fvsa-codebook-atom", "fvsa-cleanup", "fvsa-resonate",
        "fvsa-recompose-score", "fvsa-free-codebook")
        delete!(GROUNDED_REGISTRY, k)
    end
    nothing
end
