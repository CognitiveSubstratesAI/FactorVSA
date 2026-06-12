#!/usr/bin/env julia
# tools/repl.jl — warm development REPL for FactorVSA (mirrors MORK's tools/repl.jl).
#
# Interactive (recommended — Revise hot-reload, no restart on function edits):
#   julia --project=. -i tools/repl.jl
# Scripted (pipe a targeted snippet — foreground, NO background, NO polling):
#   printf 'include("/tmp/snippet.jl")\n' | julia --project=. -i tools/repl.jl
#
# NEVER cold-start a fresh `julia test/runtests.jl` for iteration — run a TARGETED snippet here
# and debug with @show / println / @info. Re-run t() only for the final full-suite gate.
#
# Dev tools live in the GLOBAL env (~/.julia/environments/v#.#), which is on the default
# LOAD_PATH — so they load here WITHOUT being FactorVSA dependencies. Revise auto-loads via
# ~/.julia/config/startup.jl in interactive sessions; guarded again here for safety. Heavier
# tools are available on demand (they'd slow every snippet if eager): `using BenchmarkTools`,
# `using JET`, `using Cthulhu`. Profile is stdlib and loaded below.
#
# NOTE (Revise limitation): editing a STRUCT's fields needs a fresh session; function-body
# edits hot-reload in place. We edit logic far more than structs, so restarts are rare.

try; using Revise; catch; @warn "Revise unavailable — install into the global env for hot-reload"; end

using FactorVSA, MORK
using MORK: GROUNDED_REGISTRY, is_grounded
using Test, Profile

FactorVSA.register_factorvsa!()

"The MORK grounded-op registry (fvsa-* ops are registered)."
reg() = GROUNDED_REGISTRY
"Run the full FactorVSA test suite from the warm session."
t() = include(joinpath(dirname(@__DIR__), "test", "runtests.jl"))

if isinteractive()
    println("FactorVSA REPL ready — Revise tracking src/, fvsa-* registered.")
    println("  reg()  — grounded registry   t()  — full suite")
    println("  on-demand: `using BenchmarkTools` / `using JET` / `using Cthulhu` (global env)")
end
