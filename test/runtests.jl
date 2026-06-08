using Test
using Random
using LinearAlgebra
using FactorVSA
using MORK: GROUNDED_REGISTRY   # phase-2 shim test inspects the grounded registry

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

    @testset "Step 5a/5b — R-HMH episode encode + resonant recall" begin
        Random.seed!(20260606)              # role! + codebooks use default_rng → deterministic
        D = 4096
        rb = RoleBook(D)
        cbA = random_codebook(BipolarMAP, D, 16)
        cbB = random_codebook(BipolarMAP, D, 16)
        fa = HV{BipolarMAP}(cbA.atoms[:, 5])
        fb = HV{BipolarMAP}(cbB.atoms[:, 11])
        slots = Dict(:actor => (:agent, fa), :object => (:thing, fb))

        # 5a encode → 5b recover each slot filler (role-masked unbind + cleanup)
        H = encode_episode(Episode(random_hv(BipolarMAP, D); slots=slots), rb)
        @test recover_slot(H, :actor, :agent, rb, cbA) == fa
        @test recover_slot(H, :object, :thing, rb, cbB) == fb

        # a typed relation is bounded crosstalk — slots still recover
        H2 = encode_episode(
            Episode(random_hv(BipolarMAP, D); slots=slots,
                relations=[(:acts_on, :actor, :object)]), rb)
        @test recover_slot(H2, :actor, :agent, rb, cbA) == fa
        @test recover_slot(H2, :object, :thing, rb, cbB) == fb

        # product-filler completion (Eq 73): resonate INSIDE the episode
        c1 = random_codebook(BipolarMAP, D, 10)
        c2 = random_codebook(BipolarMAP, D, 10)
        p1 = HV{BipolarMAP}(c1.atoms[:, 3])
        p2 = HV{BipolarMAP}(c2.atoms[:, 8])
        H3 = encode_episode(
            Episode(random_hv(BipolarMAP, D); slots=Dict(:goal => (:plan, bind(p1, p2)))),
            rb)
        facs, score = complete_slot(H3, :goal, :plan, rb, [c1, c2])
        @test facs[1] == p1 && facs[2] == p2     # resonator recovers the true factors
        # NOTE: the slot is unbound from a PROJECTED (sign'd) episode bundle, so the
        # recovered filler is noisy — absolute recompose score is ~0.5, not ~1 (bare-product
        # margins don't apply in-episode). What matters is true-tuple ≫ spurious (~0): the
        # rejection signal still works. (This crosstalk is exactly what the 5-gate measures.)
        @test score > 0.3
    end

    @testset "Step 5c — episodic-semantic consolidation (Eq 77)" begin
        Random.seed!(20260606)
        D = 4096
        rb = RoleBook(D)
        cbAct = random_codebook(BipolarMAP, D, 8)
        cbObj = random_codebook(BipolarMAP, D, 8)
        fa = HV{BipolarMAP}(cbAct.atoms[:, 2])          # COMMON actor across all episodes
        # 8 episodes: same actor, idiosyncratic object each
        eps = [
            Episode(random_hv(BipolarMAP, D);
                slots=Dict(:actor => (:agent, fa),
                    :object => (:thing, HV{BipolarMAP}(cbObj.atoms[:, i])))) for i in 1:8
        ]
        tmpl = consolidate(eps, rb)
        # schema formation: the RECURRING slot (actor) is recoverable from the template…
        @test recover_slot(tmpl, :actor, :agent, rb, cbAct) == fa
        # …and it has a CLEARER margin than the varying slot (idiosyncratic objects wash out)
        m_actor = cleanup_margin_of(
            bind(role!(rb, Symbol("mtype_", :agent)), bind(role!(rb, :actor), tmpl)), cbAct)
        m_object = cleanup_margin_of(
            bind(role!(rb, Symbol("mtype_", :thing)), bind(role!(rb, :object), tmpl)), cbObj
        )
        @test m_actor > m_object
        # immutable: template is a fresh HV, codebooks untouched (a new dict would be a new CodebookRef)
        @test tmpl isa HV{BipolarMAP}
    end

    @testset "Step 5d — ColBaC-HDC representation layer (§9)" begin
        Random.seed!(20260606)
        D = 4096
        rb = RoleBook(D)
        cbM = random_codebook(BipolarMAP, D, 16)
        m1 = HV{BipolarMAP}(cbM.atoms[:, 4])
        m2 = HV{BipolarMAP}(cbM.atoms[:, 9])
        m3 = HV{BipolarMAP}(cbM.atoms[:, 1])
        col = Column([(:K, 0, :center, m1), (:L, 1, :lateral, m2), (:B, 2, :bridge, m3)])
        Hm = encode_column(col, rb)

        # Eq 84 ≡ Eq 11: same unbind-cleanup machinery recovers any triple-tagged motif
        @test recover_motif(Hm, :K, 0, :center, rb, cbM) == m1
        @test recover_motif(Hm, :B, 2, :bridge, rb, cbM) == m3

        # support code (Eq 85): a column is recoverable from the active support
        colB = Column([(:K, 0, :center, HV{BipolarMAP}(cbM.atoms[:, 7]))])
        enc = Dict(1 => Hm, 2 => encode_column(colB, rb))
        HS = support_code(enc, [1, 2], rb)
        @test dot(bind(role!(rb, Symbol("col_", 1)), HS).data, Hm.data) / D > 0.3

        # certificate (Eq 86-87): a channel is recovered from the structured cert hypervector
        cbCh = random_codebook(BipolarMAP, D, 8)
        zshared = HV{BipolarMAP}(cbCh.atoms[:, 3])
        Z = certificate(Dict(:shared => zshared), Hm, rb)
        @test cleanup(bind(role!(rb, :cert_shared), Z), cbCh) == zshared

        # HDC audit quantities: a clean atom is unambiguous (margin>0, M_conf==1)
        @test cleanup_margin_of(m1, cbM) > 0
        @test confusability(m1, cbM; gamma=0.01) == 1
    end
end
