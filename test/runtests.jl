using Test
using Random
using LinearAlgebra
using Base.Threads               # concurrency testset (DualIndex thread safety)
using FactorVSA
using MORK: GROUNDED_REGISTRY, is_grounded   # shim tests inspect the grounded registry

rng() = MersenneTwister(0xF00D)

@testset "FactorVSA" begin

    @testset "Step 0 — dual index" begin
        @test VectorHandle === UInt32
        @test ArenaScanBackend <: ReverseBackend
        idx = DualIndex{UInt128, HV{BipolarMAP}}()
        @test idx isa DualIndex{UInt128, HV{BipolarMAP}, ArenaScanBackend}

        a = HV{BipolarMAP}([1.0, -1.0, 1.0, -1.0])
        b = HV{BipolarMAP}([-1.0, -1.0, 1.0, 1.0])
        c = HV{BipolarMAP}([1.0, 1.0, 1.0, 1.0])
        ra = insert_vector!(idx, UInt128(10), a)
        rb = insert_vector!(idx, UInt128(20), b)
        rc = insert_vector!(idx, UInt128(30), c)

        @test lookup_vector(idx, UInt128(10)) == a
        @test deref(idx, ra) == a

        # Integrity rule 1: stable handles survive deletes (no positional desync)
        free_vector!(idx, rb.handle)                 # free B
        rd = insert_vector!(idx, UInt128(40), c)      # reuses B's slot
        @test rd.handle == rb.handle                  # slot recycled
        @test deref(idx, rb) === nothing              # B's old ref is now STALE (ABA guard)
        @test deref(idx, ra) == a                     # A unaffected by B's free + D's insert
        @test deref(idx, rc) == c                     # C unaffected
        @test lookup_vector(idx, UInt128(20)) === nothing  # B is gone

        # Integrity rule 2: the cache is rebuildable from authoritative pairs
        empty!(idx.id_to_handle)
        @test lookup_vector(idx, UInt128(10)) === nothing  # cache cold
        rebuild_cache!(idx, [(UInt128(10), ra.handle), (UInt128(30), rc.handle)])
        @test lookup_vector(idx, UInt128(10)) == a         # recovered

        # reverse lookup returns handles by similarity, never by position
        hits = reverse_lookup(idx, a; topk=1)
        @test hits[1] == ra.handle
    end

    @testset "Step 1 — algebra" begin
        r = rng()
        D = 1024
        a = random_hv(BipolarMAP, D, r)
        v = random_hv(BipolarMAP, D, r)
        # self-inverse unbind: unbind(a, bind(a,v)) == v exactly (MAP)
        @test unbind(a, bind(a, v)) == v
        # bound vector ~orthogonal to inputs
        bound = bind(a, v)
        @test abs(dot(bound.data, a.data)) / D < 0.15
        # proj is sign; bundle sums then proj recovers majority
        s = proj(bundle(a, a, v))
        @test all(x -> x == 1.0 || x == -1.0, s.data)
        # cleanup recovers an atom from atom + light crosstalk
        cb = random_codebook(BipolarMAP, D, 32; rng=r)
        target = HV{BipolarMAP}(cb.atoms[:, 7])
        noisy = bundle(target, random_hv(BipolarMAP, D, r))
        @test cleanup(noisy, cb) == target
    end

    @testset "Step 2 — resonator" begin
        r = rng()
        D = 2048
        sizes = [12, 12, 12]
        cbs = [random_codebook(BipolarMAP, D, M; rng=r) for M in sizes]
        ti = [3, 8, 1]
        factors = [HV{BipolarMAP}(cbs[f].atoms[:, ti[f]]) for f in 1:3]
        H = reduce(bind, factors)
        rec, score = factorize(H, cbs; iterations=50, beta=4.0, rng=r)
        @test all(rec[f] == factors[f] for f in 1:3)   # true tuple recovered
        @test score > 0.9                               # clean recomposition
    end

    @testset "Step 3 — encode/descend" begin
        r = rng()
        D = 4096
        roles = make_roles(D, 2, 3; rng=r)
        cb_of(hvs) = Codebook{BipolarMAP}(reduce(hcat, (h.data for h in hvs)),
            [Symbol(:s, i) for i in 1:length(hvs)])

        u_a = random_hv(BipolarMAP, D, r)
        u_b = random_hv(BipolarMAP, D, r)
        u_root = random_hv(BipolarMAP, D, r)
        leafA = VSATree(u_a)
        leafB = VSATree(u_b)
        root = VSATree(u_root; children=[leafA, leafB])
        H = encode(root, roles)

        # descend recovers the CLEANED SUBTREE CODE H_ν (Alg 2), cleaned vs the level
        # subtree dictionary 𝒦ᵗˢᵘᵇ (the encoded subtrees) — NOT the prototype u_ν.
        HA = encode(leafA, roles, 1)
        HB = encode(leafB, roles, 1)
        distract = [random_hv(BipolarMAP, D, r) for _ in 1:6]
        sub1 = cb_of([HA, HB, distract...])

        @test descend(H, [1], roles, [sub1]) == HA      # child-1 subtree code recovered
        @test descend(H, [2], roles, [sub1]) == HB      # child-2 subtree code recovered
        # the leaf prototype is then one unbind away from the (leaf) subtree code
        @test unbind(roles.tau[2], HA) == u_a
    end

    @testset "Step 4 — gate instrumentation (smoke)" begin
        r = rng()
        # favorable regime: large D / small k → near-zero failure
        pfail_lo = cleanup_margin(2048, 3, 64; trials=200, rng=r)
        # adverse regime: small D / large k → higher failure
        pfail_hi = cleanup_margin(128, 12, 64; trials=200, rng=r)
        @test pfail_lo < 0.05
        @test pfail_hi > pfail_lo            # failure rises as D/k falls (Lemma 2 direction)

        rate, _ = resonator_success_rate(2048, [10, 10, 10]; trials=40, beta=4.0, rng=r)
        @test rate > 0.8                     # resonator works in the margin regime
    end

    @testset "Phase-2 — MeTTa shim (grounded ops over handle-arena)" begin
        register_factorvsa!()
        reg = GROUNDED_REGISTRY
        @test haskey(reg, "fvsa-bind") && haskey(reg, "fvsa-sim")

        # create → handle round-trips through (VecRef h)
        a = reg["fvsa-random"](["1024"])
        b = reg["fvsa-random"](["1024"])
        @test occursin(r"^\(VecRef \d+\)$", a)
        @test parse_vecref(a) isa Int
        @test vecref(parse_vecref(a)) == a

        # similarity: self ≈ 1, distinct ~ 0 (scalar string result)
        @test parse(Float64, reg["fvsa-sim"]([a, a])) ≈ 1.0
        @test abs(parse(Float64, reg["fvsa-sim"]([a, b]))) < 0.15

        # bind then unbind recovers b (BipolarMAP self-inverse): sim(b, unbind(a,bind(a,b))) ≈ 1
        ab = reg["fvsa-bind"]([a, b])
        b2 = reg["fvsa-unbind"]([a, ab])
        @test parse(Float64, reg["fvsa-sim"]([b, b2])) ≈ 1.0

        # bundle yields a handle
        @test occursin(r"^\(VecRef \d+\)$", reg["fvsa-bundle"]([a, b]))

        # dangling / malformed handles fail loud (never silently mis-route)
        @test_throws Exception reg["fvsa-sim"](["(VecRef 999999)", a])
        @test_throws Exception parse_vecref("(NotAHandle 1)")

        unregister_factorvsa!()
        @test !haskey(reg, "fvsa-bind")
    end

    @testset "Phase-2b — resonator + codebooks over MeTTa" begin
        Random.seed!(20260606)              # deterministic (shim ops use default_rng)
        register_factorvsa!()
        reg = GROUNDED_REGISTRY
        @test haskey(reg, "fvsa-resonate") && haskey(reg, "fvsa-codebook")

        D = "4096"
        c1 = reg["fvsa-codebook"]([D, "10"])
        c2 = reg["fvsa-codebook"]([D, "10"])
        c3 = reg["fvsa-codebook"]([D, "10"])
        @test occursin(r"^\(CodebookRef \d+\)$", c1)

        # pick true factor atoms, bind into a product H
        a1 = reg["fvsa-codebook-atom"]([c1, "3"])
        a2 = reg["fvsa-codebook-atom"]([c2, "7"])
        a3 = reg["fvsa-codebook-atom"]([c3, "1"])
        H = reg["fvsa-bind"]([a1, reg["fvsa-bind"]([a2, a3])])

        # THE RESONATOR over MeTTa: recover the factor tuple (the headline capability)
        res = reg["fvsa-resonate"]([H, c1, c2, c3])
        hs = [m.match for m in eachmatch(r"\(VecRef \d+\)", res)]
        @test length(hs) == 3
        # recovered factors match the true atoms, order-aligned to the codebook args
        @test parse(Float64, reg["fvsa-sim"]([hs[1], a1])) ≈ 1.0
        @test parse(Float64, reg["fvsa-sim"]([hs[2], a2])) ≈ 1.0
        @test parse(Float64, reg["fvsa-sim"]([hs[3], a3])) ≈ 1.0
        # recompose score high for the true tuple (spurious-rejection signal)
        @test parse(Float64, reg["fvsa-recompose-score"]([H, hs...])) > 0.9

        # cleanup: a1 + light crosstalk → nearest atom is a1
        z = reg["fvsa-bundle"]([a1, reg["fvsa-random"]([D])])
        @test parse(Float64, reg["fvsa-sim"]([reg["fvsa-cleanup"]([z, c1]), a1])) ≈ 1.0

        # immutable-codebook lifecycle: free → dangling handle fails loud
        reg["fvsa-free-codebook"]([c3])
        @test_throws Exception reg["fvsa-codebook-atom"]([c3, "1"])

        unregister_factorvsa!()
        @test !haskey(reg, "fvsa-resonate")
    end

    @testset "Grounded-atom types — (VecRef h) is data, fvsa-* is operation" begin
        register_factorvsa!()
        reg = GROUNDED_REGISTRY
        D = "1024"

        # ── (1) ROUTING: only fvsa-* HEAD symbols are grounded operations. asource_new
        # routes on the expr head; a (VecRef h) atom's head is `VecRef`, never registered,
        # so it stays DATA (stored/matched structurally), never evaluated.
        @test is_grounded("fvsa-bind") && is_grounded("fvsa-resonate")
        @test !is_grounded("VecRef")          # the handle constructor is data, not an op
        @test !is_grounded("(VecRef 5)")      # a handle atom is never itself executable
        @test !is_grounded("CodebookRef")

        # ── (2) IDENTITY is the handle, NOT the dense content. Distinct stores get distinct
        # handles even for byte-identical vectors (arena id ≠ content hash) — so handle
        # equality is string/atom equality, decoupled from vector similarity.
        a = reg["fvsa-random"]([D])
        @test occursin(r"^\(VecRef \d+\)$", a)
        b = reg["fvsa-bind"]([a, a])          # a⊗a = all-ones (deterministic content)
        c = reg["fvsa-bind"]([a, a])          # same content, fresh store
        @test b != c                          # different handles…
        @test parse_vecref(b) != parse_vecref(c)
        @test parse(Float64, reg["fvsa-sim"]([b, c])) ≈ 1.0   # …identical content

        # ── (3) ANTI-INSTRUCTION as regression: dense vectors NEVER cross into the grounded
        # registry / trie. The registry is Dict{String,Function} (type-enforced: an HV can't
        # be a value), and every dense-producing op RETURNS a handle STRING — the HV lives
        # only in FVSA_ARENA, reachable by handle. This encodes MeTTaShim's core invariant.
        @test eltype(values(reg)) <: Function          # no HV can ever be a registry value
        for (op, args) in (("fvsa-random", [D]), ("fvsa-bind", [a, a]),
            ("fvsa-unbind", [a, a]), ("fvsa-bundle", [a, a]))
            r = reg[op](args)
            @test r isa AbstractString                 # a STRING crosses the boundary…
            @test !(r isa HV)                          # …never a dense vector
            @test occursin(r"^\(VecRef \d+\)$", r)
        end
        # the dense result is retrievable from the ARENA by handle (it stayed there)
        @test lookup_vector(FVSA_ARENA, parse_vecref(a)) isa HV{BipolarMAP}
        # scalar ops cross as a parseable scalar STRING, not a vector
        s = reg["fvsa-sim"]([a, a])
        @test s isa AbstractString && parse(Float64, s) ≈ 1.0

        unregister_factorvsa!()
        @test !is_grounded("fvsa-bind")        # unregister tears down the routing
    end

    @testset "Dual index — content-addressed embedding ↔ MORK id" begin
        # The COMPLEMENT of the identity test above: scratch vectors have HANDLE identity
        # (same content ⇒ distinct handles), but ATOM EMBEDDINGS have CONTENT identity —
        # the same atom ⇒ the same handle, keyed by MORK's content-id. This is what makes
        # the vector index genuinely DUAL to the symbolic (MORK) index (Hyperon WP §2/§7.8).
        empty!(FVSA_EMBED)                       # isolate from any prior embed state
        register_factorvsa!()
        reg = GROUNDED_REGISTRY
        @test is_grounded("fvsa-embed")

        # ── (1) IDEMPOTENT / content-addressed: same atom ⇒ same handle, every call.
        d1 = reg["fvsa-embed"](["dog"])
        d2 = reg["fvsa-embed"](["dog"])
        @test occursin(r"^\(VecRef \d+\)$", d1)
        @test d1 == d2                                   # SAME handle (vs fvsa-random: distinct)
        @test parse(Float64, reg["fvsa-sim"]([d1, d2])) ≈ 1.0

        # ── (2) CANONICAL: MORK re-serializes, so whitespace variants of the SAME atom are
        # one id — but `dog` (symbol) and `(dog)` (1-expr) are DIFFERENT atoms (content rightly
        # distinguishes them; that's the point of content-addressing).
        @test content_id("dog") == content_id(" dog") == content_id("dog ")   # same symbol
        @test content_id("(pet dog)") == content_id("( pet   dog )")          # same compound
        @test content_id("dog") != content_id("(dog)")                       # symbol ≠ 1-list
        @test reg["fvsa-embed"]([" dog "]) == d1         # same atom, different surface text

        # ── (3) DISTINCT atoms ⇒ distinct handles + near-orthogonal vectors.
        cat = reg["fvsa-embed"](["cat"])
        @test cat != d1
        @test abs(parse(Float64, reg["fvsa-sim"]([d1, cat]))) < 0.15

        # ── (4) the embedding IS a normal (VecRef h) — composes with the rest of the algebra.
        bound = reg["fvsa-bind"]([d1, cat])
        @test parse(Float64, reg["fvsa-sim"]([reg["fvsa-unbind"]([cat, bound]), d1])) ≈ 1.0

        # ── (5) compound atoms are content-addressed too, distinct from their parts.
        comp = reg["fvsa-embed"](["(pet dog)"])
        @test comp == reg["fvsa-embed"](["(pet dog)"])   # idempotent
        @test comp != d1                                 # the compound ≠ its symbol

        # ── (6) the dual map links the MORK content-id to the stored vector slot.
        @test haskey(FVSA_EMBED, content_id("dog"))
        @test lookup_vector(FVSA_ARENA, FVSA_EMBED[content_id("dog")]) isa HV{BipolarMAP}

        unregister_factorvsa!()
        @test !is_grounded("fvsa-embed")
    end

    @testset "Dual index — compositional compound encoding (decodable, Eq 11)" begin
        # A COMPOUND atom's vector is BUILT from its parts (encode(VSATree,Roles)), so it is
        # VSA-decodable and structure-sensitive — not an opaque seed. Seeded: leaf seeds use the
        # default RNG (roles use a fixed internal seed), so this pins the similarity margins.
        Random.seed!(20260612)
        register_factorvsa!()
        reg = GROUNDED_REGISTRY
        empty!(FactorVSA.FVSA_EMBED)
        sim(a, b) = parse(Float64, reg["fvsa-sim"]([a, b]))
        hv(v) = lookup_vector(FVSA_ARENA, parse_vecref(v))
        cossim(a, b) = sum(a.data .* b.data) / length(a.data)

        # parser handles nesting
        @test FactorVSA._parse_sexpr("(pet dog)") == Any["pet", "dog"]
        @test FactorVSA._parse_sexpr("(a (b c) d)") == Any["a", Any["b", "c"], "d"]

        pd = reg["fvsa-embed"](["(pet dog)"])
        pc = reg["fvsa-embed"](["(pet cat)"])
        cw = reg["fvsa-embed"](["(car wheel)"])
        # idempotent + structure-sensitive: sharing a part ⇒ more similar than sharing none
        @test pd == reg["fvsa-embed"](["(pet dog)"])
        @test sim(pd, pc) > sim(pd, cw) + 0.10

        # nested leaves are content-addressed AND shared with the standalone embedding
        dog = reg["fvsa-embed"](["dog"])
        @test FactorVSA._embed_id("dog") == parse_vecref(dog)

        # DECODABLE: unbind dog's position role (index 2) from (pet dog) ⇒ recover dog ≫ a wrong leaf.
        # encode child = proj(τ[2]⊗leaf); parent binds r[1][i]⊗permute(child,1) — invert in reverse.
        roles, _ = FactorVSA._embed_roles()
        rec = unbind(roles.tau[2], invpermute(unbind(roles.r[1][2], hv(pd)), 1))
        fish = reg["fvsa-embed"](["fish"])
        @test cossim(proj(rec), hv(dog)) > 0.15                              # recovers the true part
        @test cossim(proj(rec), hv(dog)) > 3 * abs(cossim(proj(rec), hv(fish)))  # ≫ a wrong part

        unregister_factorvsa!()
    end

    @testset "Concurrency — DualIndex thread safety" begin
        # The arena is lock-guarded; an UNGUARDED version throws BoundsError under a
        # 2-thread insert stress (proven before the fix). This testset is only
        # meaningful with ≥2 threads — CI sets JULIA_NUM_THREADS. Skip-with-log
        # otherwise (never a silent pass).
        nt = nthreads()
        if nt < 2
            @info "Concurrency testset SKIPPED — single-threaded. Re-run with `julia -t auto` (CI sets JULIA_NUM_THREADS)."
            @test_skip nt >= 2
        else
            D = 64
            # Scenario 1 — concurrent insert of distinct ids: invariants on count, handle
            # integrity, and id-map size must hold after the barrier.
            insert_errs() = begin
                idx = DualIndex{Int, HV{BipolarMAP}}()
                per = 4000
                recs = Vector{Vector{Tuple{HandleRef, HV{BipolarMAP}}}}(undef, nt)
                @threads for t in 1:nt
                    mine = Tuple{HandleRef, HV{BipolarMAP}}[]
                    for k in 1:per
                        v = random_hv(BipolarMAP, D)
                        push!(mine, (insert_vector!(idx, (t - 1) * per + k, v), v))
                    end
                    recs[t] = mine
                end
                total = nt * per
                e = String[]
                length(idx.arena) == total ||
                    push!(e, "arena length $(length(idx.arena)) != $total")
                bad = count(((ref, v),) -> deref(idx, ref) != v, Iterators.flatten(recs))
                bad == 0 || push!(e, "$bad/$total handles deref wrong")
                length(idx.id_to_handle) == total ||
                    push!(e, "id_to_handle $(length(idx.id_to_handle)) != $total")
                e
            end
            # Scenario 2 — concurrent insert/free/deref churn: a stale ref must never
            # deref after its slot is freed (ABA guard), and the run must not throw.
            aba_violations() = begin
                idx = DualIndex{Int, HV{BipolarMAP}}()
                per = 4000
                v = Atomic{Int}(0)
                @threads for t in 1:nt
                    for k in 1:per
                        ref = insert_vector!(idx, (t - 1) * per + k, random_hv(BipolarMAP, D))
                        if iseven(k)
                            free_vector!(idx, ref.handle)
                            deref(idx, ref) === nothing || atomic_add!(v, 1)
                        else
                            deref(idx, ref)
                        end
                    end
                end
                v[]
            end

            for _ in 1:2                       # repeat: races are probabilistic
                @test isempty(insert_errs())
                @test aba_violations() == 0
            end
            # Atomic id-counter: concurrent _fvsa_store! on the global arena must mint a
            # UNIQUE id per store (a non-atomic `+= 1` loses updates → fewer new entries).
            # Counts only the delta, so prior testset pollution is irrelevant.
            before = length(FVSA_ARENA.id_to_handle)
            nstores = nt * 1000
            @threads for _ in 1:nt
                for _ in 1:1000
                    FactorVSA._fvsa_store!(random_hv(BipolarMAP, D))
                end
            end
            @test length(FVSA_ARENA.id_to_handle) - before == nstores
        end
    end
end
