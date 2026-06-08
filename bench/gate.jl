# FactorVSA — Step-4 margin gate driver (paper §5 Lemma 2 / Thm 3 / Thm 2).
# Run: julia --project=. bench/gate.jl   → prints E1/E4/E3 tables + PASS/FAIL verdict.
# Reproduces GATE_RESULT.md (seed fixed).
using FactorVSA, Random, LinearAlgebra, Printf

rng = MersenneTwister(20260605)

println("="^70)
println("FactorVSA — STEP 4 MARGIN GATE  (Lemma 2 / Thm 3 / Thm 2)")
println("="^70)

# ── E1: cleanup capacity vs Lemma 2  P[fail] ≤ 2M·exp(−cD/k) ──────────────────
println("\n## E1 — single-step cleanup capacity (M=64)")
M = 64
ks = [2, 4, 8, 16]
Ds = [64, 128, 256, 512, 1024, 2048]
@printf("%6s", "D\\k")
for k in ks
    @printf("%10d", k)
end
println()
pfail = Dict{Tuple{Int, Int}, Float64}()
for D in Ds
    @printf("%6d", D)
    for k in ks
        p = cleanup_margin(D, k, M; trials=600, rng=rng)
        pfail[(D, k)] = p
        @printf("%10.4f", p)
    end
    println()
end

println("\n  fit  c = −k·slope of log(P[fail]) vs D   (expect ~constant, >0):")
cs = Float64[]
for k in ks
    xs = Float64[]
    ys = Float64[]
    for D in Ds
        p = pfail[(D, k)]
        (0 < p < 0.5) || continue
        push!(xs, D)
        push!(ys, log(p))
    end
    if length(xs) >= 2
        n = length(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(xs .^ 2)
        sxy = sum(xs .* ys)
        slope = (n * sxy - sx * sy) / (n * sxx - sx^2)
        c = -k * slope
        push!(cs, c)
        @printf("    k=%2d : c ≈ %.4f   (%d usable points)\n", k, c, n)
    else
        @printf("    k=%2d : (saturated — too few points in (0,0.5))\n", k)
    end
end
c_ok = !isempty(cs) && all(>(0), cs) && (maximum(cs) / minimum(cs) < 4)
println("  → c values all positive & within 4× of each other: ", c_ok)

println("\n  M-dependence at D=512,k=4 (expect slow ~log growth):")
for Mv in [16, 64, 256, 1024]
    @printf(
        "    M=%5d : P[fail]=%.4f\n", Mv, cleanup_margin(512, 4, Mv; trials=600, rng=rng)
    )
end

# ── E4: resonator success vs D (3 factors, |C_f|=10) ─────────────────────────
println("\n## E4 — resonator success rate (F=3, |C_f|=10)")
e4 = Tuple{Int, Float64}[]
for D in [128, 256, 512, 1024, 2048, 4096]
    rate, _ = resonator_success_rate(D, [10, 10, 10]; trials=200, beta=4.0, rng=rng)
    push!(e4, (D, rate))
    @printf("    D=%5d : success=%.3f\n", D, rate)
end
e4_ok = last(e4)[2] > 0.95 && first(e4)[2] < last(e4)[2]
println("  → high at large D and rises with D: ", e4_ok)

# ── E3: all-path negative control (entropy bound, Thm 2) ─────────────────────
println("\n## E3 — all-path negative control (fixed D=512; recover ALL N items)")
function all_recover_rate(D, N; pool=256, trials=200, rng=rng)
    ok = 0
    for _ in 1:trials
        cb = random_codebook(BipolarMAP, D, pool; rng=rng)
        roles = [random_hv(BipolarMAP, D, rng) for _ in 1:N]
        idx = [rand(rng, 1:pool) for _ in 1:N]
        terms = [bind(roles[i], HV{BipolarMAP}(cb.atoms[:, idx[i]])) for i in 1:N]
        H = proj(bundle(terms...))
        good = all(FactorVSA._cleanup_idx(unbind(roles[i], H), cb) == idx[i] for i in 1:N)
        ok += good
    end
    ok / trials
end
e3 = Tuple{Int, Float64}[]
for N in [2, 4, 8, 16, 32, 64, 128]
    r = all_recover_rate(512, N; trials=200, rng=rng)
    push!(e3, (N, r))
    @printf("    N=%4d : all-recovered=%.3f\n", N, r)
end
e3_ok = first(e3)[2] > 0.9 && last(e3)[2] < 0.1
println("  → perfect at small N, COLLAPSES as N grows at fixed D: ", e3_ok)

# ── verdict ──────────────────────────────────────────────────────────────────
gate = c_ok && e4_ok && e3_ok
println("\n" * "="^70)
println("GATE VERDICT: ", gate ? "PASS" : "FAIL",
    "   (E1 law=$c_ok, E4 resonator=$e4_ok, E3 control=$e3_ok)")
println("="^70)
