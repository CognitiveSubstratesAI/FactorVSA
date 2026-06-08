# Guide

Every `@example` block below is executed at documentation build time, so the printed results
are the real output of the current code.

```@setup guide
using FactorVSA, LinearAlgebra, Random
Random.seed!(1)
```

## Hypervectors and the algebra

A hypervector is a fixed-width sign vector under a [`Backend`](@ref) (here [`BipolarMAP`](@ref):
`±1` components). [`random_hv`](@ref) draws one; a [`Codebook`](@ref) is a labelled bank of
atoms. The MAP algebra gives three operations:

- **`bind`** — componentwise product. Self-inverse (`unbind == bind`) and similarity-randomizing:
  the product looks unrelated to its factors. This is the role⊗filler operator.
- [`bundle`](@ref) — superposition (elementwise sum, un-projected). After [`proj`](@ref) it stays
  *similar* to each summand, so it is the set / "this AND that" operator.
- [`permute`](@ref) — a fixed cyclic shift, the cheap "quote / protect" operator that makes a
  vector recoverably dissimilar to itself.

Similarity between two hypervectors is the normalized dot product:

```@example guide
D = 4096
sim(x, y) = dot(x.data, y.data) / dim(x)

a = random_hv(BipolarMAP, D)
b = random_hv(BipolarMAP, D)

bound = bind(a, b)
println("product is dissimilar to its factors:  sim(bind(a,b), a) = ", round(sim(bound, a); digits=3))
println("unbind is the exact inverse:            unbind(bind(a,b), b) == a  →  ", unbind(bound, b) == a)

sup = proj(bundle(a, b))   # projected superposition of two atoms
println("bundle stays partly similar:            sim(proj(bundle(a,b)), a) = ", round(sim(sup, a); digits=3))
```

[`cleanup`](@ref) snaps a noisy vector back to the nearest codebook atom — the denoising step
that makes the algebra usable for query:

```@example guide
cb    = random_codebook(BipolarMAP, D, 12)        # 12 named atoms
atom  = HV{BipolarMAP}(cb.atoms[:, 7])
noisy = bundle(atom, random_hv(BipolarMAP, D))    # atom buried under a distractor
cleanup(noisy, cb) == atom                        # recovered
```

## The resonator — factoring a product

A product of `F` factors, one drawn from each of `F` codebooks, can be decomposed back into its
factor atoms by `factorize` (see the [API](api.md#Operators-extending-Base-/-LinearAlgebra)).
Naively this is a `∏|𝒞_f|` search; the resonator runs it as
alternating cleanup projections in time `∝ Σ|𝒞_f|` — the linear-not-exponential win:

```@example guide
cbs     = [random_codebook(BipolarMAP, D, 10) for _ in 1:3]
true_ix = [3, 8, 1]
factors = [HV{BipolarMAP}(cbs[f].atoms[:, true_ix[f]]) for f in 1:3]
H       = reduce(bind, factors)                   # the bound tuple

recovered, score = factorize(H, cbs; restarts=3)
println("all three factors recovered: ", all(recovered[f] == factors[f] for f in 1:3))
println("recompose score: ", round(recompose_score(H, recovered); digits=3))
```

[`recompose_score`](@ref)`(H, factors)` re-binds the recovered factors and measures agreement
with the target — the convergence witness the resonator reports.

## Hierarchical codes — encode and descend

A [`VSATree`](@ref) is a typed tree: each node carries its own prototype hypervector `u` and a
`Vector` of child subtrees. [`encode`](@ref) folds it into a single hypervector (role-bind the
prototype, role-bind each permuted child code, bundle, project, recurse — Alg 1, Eq 11).
[`make_roles`](@ref)`(D, L, maxbranch)` mints the per-level role atoms.

```@example guide
roles = make_roles(D, 2, 3)                       # up to 2 levels deep, ≤3 children/node
u_a, u_b, u_root = (random_hv(BipolarMAP, D) for _ in 1:3)

leafA = VSATree(u_a)
leafB = VSATree(u_b)
root  = VSATree(u_root; children=[leafA, leafB])
H     = encode(root, roles)                        # the whole tree in one hypervector
nothing # hide
```

[`descend`](@ref) walks a path of **child indices**, unbinding the child role and cleaning up at
each step. It recovers the *cleaned subtree code* `H_ν` against that level's dictionary of
encoded subtrees (not the bare prototype — Alg 2, §4.2). The level dictionary must hold the
encoded child subtrees:

```@example guide
HA = encode(leafA, roles, 1)                       # child subtree codes (at level 1)
HB = encode(leafB, roles, 1)
distract = [random_hv(BipolarMAP, D) for _ in 1:6]
level1 = Codebook{BipolarMAP}(reduce(hcat, (h.data for h in [HA, HB, distract...])),
    [Symbol(:s, i) for i in 1:8])

println("child-1 subtree recovered: ", descend(H, [1], roles, [level1]) == HA)
println("its leaf prototype is one unbind away: ", unbind(roles.tau[2], HA) == u_a)
```

## Step 0 — the dual index

[`DualIndex`](@ref) hands out stable, generation-stamped [`HandleRef`](@ref)s for vectors so the
rest of the system refers to them by handle, never by raw array. The reverse backend is
pluggable ([`ArenaScanBackend`](@ref) by default) for content → handle lookup:

```@example guide
idx = DualIndex{Symbol, HV{BipolarMAP}}()          # id type, stored-vector type
v   = random_hv(BipolarMAP, D)
ref = insert_vector!(idx, :v1, v)                  # store under id :v1 → HandleRef

println("deref round-trips to the vector:   ", deref(idx, ref) == v)
println("content scan finds the same slot:  ", first(reverse_lookup(idx, v)) == ref.handle)
```

## MeTTa integration

FactorVSA is callable from MeTTa through MORK's grounded-op channel (upstream-aligned, **zero**
MORK changes). Vectors live in the arena and are referenced from MeTTa by a `(VecRef h)` handle
string; codebooks by `(CodebookRef c)`. Registering the ops:

```julia
register_factorvsa!()
```

makes them available in the MeTTa reader, e.g.:

```metta
!(fvsa-random 4096)            ; → (VecRef h0)         a fresh hypervector
!(fvsa-bind (VecRef h0) (VecRef h1))
!(fvsa-sim  (VecRef h0) (VecRef h1))   ; → similarity scalar
!(fvsa-resonate (VecRef hp) (CodebookRef c0) (CodebookRef c1) …)  ; → recovered factor refs
```

Codebooks are immutable once registered, so handles stay valid for the arena's lifetime.
