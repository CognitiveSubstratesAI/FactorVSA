using Test
using Random
using LinearAlgebra
using FactorVSA

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
end
