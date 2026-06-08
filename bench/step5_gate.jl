# Step-5 (R-HMH) margin gate — §8.6 episodic admissibility. Run: julia --project=. bench/step5_gate.jl
# Go/no-go for building 5c/5d. Episodes carry crosstalk load k_slot + k_rel; recovery holds
# until load exceeds ≈ D/(2·ln M) (Lemma 2). The gate must PUSH PAST that boundary so its
# negative control actually exercises degradation (else it's measuring only the easy regime —
# the E3 anti-gaming lesson from Step-4). Honest-failure: if a REALISTIC episode (k_slot≈10-15,
# k_rel≈5) doesn't recover at feasible D, STOP and report — do not patch.
using FactorVSA, Random, Printf

Random.seed!(20260606)

println("="^72)
println("FactorVSA — STEP-5 R-HMH MARGIN GATE (§8.6 episodic admissibility)")
println("="^72)

# ── A. recovery vs D × k_slot, PUSHED to the boundary (k_rel=0, M=64) ─────────
println("\n## A — recovery vs D × k_slot, pushed past capacity (k_rel=0, M=64)")
Ds = [256, 512, 1024, 2048, 4096]
kslots = [8, 32, 64, 128, 256]
@printf("%7s", "D\\ks")
for ks in kslots
    @printf("%8d", ks)
end
println()
for D in Ds
    @printf("%7d", D)
    for ks in kslots
        @printf("%8.3f", episode_recovery_rate(D, ks, 0, 64; trials=40))
    end
    println()
end

# ── B. relation load swept INDEPENDENTLY (k_slot=4 fixed, M=64) ──────────────
println("\n## B — recovery vs k_rel, pushed (k_slot=4 fixed, M=64) — relations = crosstalk")
for D in [512, 1024, 2048]
    @printf("  D=%5d : ", D)
    for kr in [0, 16, 64, 128, 256]
        @printf("k_rel=%d→%.3f  ", kr, episode_recovery_rate(D, 4, kr, 64; trials=40))
    end
    println()
end

# ── C. negative control: FIXED D=512, M=64 — recovery MUST collapse as load grows ─
println("\n## C — negative control (D=512, M=64): recovery must collapse as k_slot grows")
ctrl = [
    (ks, episode_recovery_rate(512, ks, 0, 64; trials=40)) for ks in [8, 32, 64, 128, 256]
]
for (ks, r) in ctrl
    @printf("    k_slot=%3d : %.3f\n", ks, r)
end
ctrl_ok = first(ctrl)[2] > 0.9 && last(ctrl)[2] < 0.5   # high at low load, collapses at high

# ── D. realistic episode (k_slot=10, k_rel=5, M=16): min D for ≥0.95 ─────────
println("\n## D — realistic episode (k_slot=10, k_rel=5, M=16): recovery vs D")
feasD = 0
for D in [256, 512, 1024, 2048, 4096]
    r = episode_recovery_rate(D, 10, 5, 16; trials=80)
    @printf("    D=%5d : %.3f\n", D, r)
    if feasD == 0 && r >= 0.95
        global feasD = D
    end
end

# ── verdict ──────────────────────────────────────────────────────────────────
println("\n" * "="^72)
feasible = feasD != 0 && feasD <= 8192
gate = feasible && ctrl_ok
@printf("realistic episode ≥0.95 at D=%s (feasible ≤8192: %s)\n",
    feasD == 0 ? ">4096" : string(feasD), feasible)
println("negative control collapses with load (gate is a real test): ", ctrl_ok)
println("STEP-5 GATE VERDICT: ", gate ? "PASS" : "NO-GO")
println("="^72)
