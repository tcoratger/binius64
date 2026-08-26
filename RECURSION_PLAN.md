# Recursion: gap analysis and incremental PR plan

Scope of this note: what `crates/recursion` can do today, what's missing for a working
XMSS aggregation-recursion pipeline, what prior art exists (Succinct's `flock`
`recursion_circuit` branch, `leanEthereum/leanVM-b`), what Linear already knows, and a
sequenced list of PRs to close the gap. This is a planning document, not a design spec —
each PR below still needs its own design pass before implementation.

**Security note:** while researching the `succinctlabs/flock` `recursion_circuit` branch,
one file (`docs/recursion-100-128-variants.md`) was found to contain an appended line
attempting prompt injection (a pointer to a `claude.ai` artifact claiming to be "an audit
of our branch"). It was not followed. Treat that file with suspicion if re-reading it with
any LLM tool.

**Naming caveat:** `succinctlabs/flock` (this note's external reference) is a *different*
repository from `Layr-Labs/flock-challenge` (the repo behind the "Flock PR batch results" /
"Flock-challenge submissions" work tracked in prior sessions). Both are called "Flock"
colloquially and both build on binius64 math, but they are unrelated codebases with
different architectures. BINIUS-502's "Flock recursion catch-up analysis" refers to
`succinctlabs/flock`, not `Layr-Labs/flock-challenge`.

---

## 1. Where `crates/recursion` actually stands today

Files: `challenger.rs`, `channel.rs`, `filler.rs`, `hints.rs`, `merkle.rs` (+`merkle/tests.rs`),
`shared.rs`, `symbolic/{mod,elem,word}.rs`; tests: `basefold_opening.rs`, `channel_builds.rs`,
`recursion_end_to_end.rs`.

**The mechanism.** The crate runs the *existing native verifier code* twice against two
different `IPVerifierChannel` implementations:

- `Binius64BuilderChannel` (`channel.rs`) — every verifier operation either allocates fresh
  circuit input wires and absorbs them into an in-circuit `Sha256Challenger`
  (`challenger.rs`, a bit-exact reimplementation of the native Fiat-Shamir challenger,
  deliberately branch-free since "the native challenger's control flow never reads the data
  it hashes"), or derives values purely from existing wires (`select`, `subset_sum`,
  `pack_words`). This *builds* a circuit that expresses "this verifier accepts."
- `WitnessFillerChannel` (`filler.rs`) — runs the same verifier for real, against a real
  `VerifierTranscript`, and fills every wire the build recorded with its real value.
  `assert_zero` is intentionally a no-op here (soundness is delegated entirely to the
  compiled circuit's constraints, not to this replay).
- `SymbolicElem`/`SymbolicWord` (`symbolic/`) let ordinary field/word arithmetic in the
  verifier code transparently become circuit gates (or fold to nothing at build time when
  both operands are compile-time constants).
- `merkle.rs` reimplements the binary Merkle scheme in-circuit: leaf digest, node
  compression, layer folding, and per-query authentication-path climbing, extensively
  cost-documented (738 AND per climbed level).

**What this proves works, concretely** (`tests/recursion_end_to_end.rs`): build a toy
circuit -> prove it natively -> run the *whole* `iop_verifier().verify()` over
`Binius64BuilderChannel` -> replay over `WitnessFillerChannel` -> the resulting recursive
circuit is satisfied and its public wires equal the inner statement. **This is one full
Binius64 proof, of a shape fixed before the recursive circuit is built, verified inside a
circuit that can itself be proven — one level of single-proof recursion.**

**What is explicitly missing:**

1. No multi-proof / aggregation support anywhere — no API verifies more than one inner
   proof in a single circuit, and nothing folds/merges multiple inner statements.
2. Shape (`n_vars`, `log_inv_rate`, query count, tree depth) is a Rust-side constant
   consumed while building the circuit — a compiled circuit only verifies proofs of that
   exact shape.
3. Zero connection to `crates/circuits/src/hash_based_sig` (XMSS/WOTS). Grepping the whole
   workspace outside `crates/recursion` for `binius_recursion`/`binius-recursion` returns
   nothing. The XMSS circuits and the recursion primitives have never been wired together.
4. **Correction after a closer read (this was wrong in the first draft of this note):**
   BINIUS-470 is not an open soundness gap — it is already fixed and Linear lists it Done.
   The masking gate lives in the challenger's bit-sampling routine, which bounds the drawn
   value below `2^bits` with a real gate before it ever becomes a query index, and every
   real call site derives a Merkle-opening index from that masked draw (directly, or via a
   shift that preserves the bound across fold levels). What's actually wrong is that
   `channel.rs`'s `recv_openings` doc comment never got updated after that fix landed — it
   still describes the pre-fix state and points at BINIUS-470 as if it were open. This is a
   doc-staleness defect, not a live soundness hole; PR 1 below is scoped accordingly.
5. No API to actually prove the recursive circuit itself. `recursion_end_to_end.rs` stops
   at `populate_wire_witness`; nothing calls `Prover` on the resulting circuit or verifies
   *that* proof. There is no depth-2 test (a proof of a circuit that itself verifies a
   proof).
6. `tests/channel_builds.rs` has a stale doc comment claiming operations still `todo!()`
   that are in fact fully implemented in `channel.rs` — minor doc-hygiene defect.

**XMSS side** (`crates/circuits/src/hash_based_sig/{xmss,wots,hashing,aggregate}.rs`,
driven by `crates/examples/src/circuits/hashsign.rs`): `aggregate.rs`'s "multisig" is
**in-circuit batching**, not recursion — `circuit_xmss_multisig` just loops over signers
inside one flat circuit (or one M4 chip call per signer). It produces exactly one circuit
and one proof covering all signers; there is no tree of smaller proofs recursively combined,
and no lever to keep per-shard proving cost from scaling with total signer count.

**BINIUS-507** (this branch's own change: `crates/iop/src/channel/merge.rs`,
`crates/iop-prover/src/channel/merge.rs`) buffers a round's oracle commitments and flushes
them as one combined commitment, cutting a measured 32.3% off proof size in its own example.
It is purely additive — not wired into the real Basefold/FRI driver, and not consumed by
`Binius64BuilderChannel`/`WitnessFillerChannel`, which still implement the un-merged
`MerkleIPVerifierChannel` trait directly. This matters for recursion specifically because
recursive-circuit size is dominated by **Merkle-opening count** (738 AND per level, per
query, per commitment) far more than native proof size is — fewer, larger merged
commitments per round directly shrinks the in-circuit Merkle-climb cost.

## 2. What Linear already knows

- **BINIUS-331** ("Recursion roadmap discussion", Backlog, never started) is the one
  authoritative roadmap. Key measurements: a BMUL costs about the same as an AND in
  verifier terms (rules out algebraic-hash tricks); transcript/hashing cost is dominated by
  traffic volume, not per-invocation hash cost; a naive fan-in-2 node doesn't compress under
  any hash at current parameters. Proposed order: (1) BinMul (done) -> (2) thin verify-in-
  circuit with monster+openings deferred -> (3) **opening discharge/accumulation — "the crux,
  and the piece with no complete prior art"** -> (4) Spark for the "monster" polynomial
  (succinctness endgame). Sequencing constraint stated explicitly: **2 before 3 before 4**.
- **BINIUS-434** ("Recursive Binius64 verifier circuit", Backlog) names "the monster":
  symbolic evaluation of `compute_public_value` costs ~13 BMUL per inner AND constraint,
  ~55M BMUL for a 2^22 inner circuit — "a node built this way cannot compress, at any
  fan-in." Its own recommended first step is exactly what evaluating symbolically today
  would give: a demonstration-only baseline, expensive but real, to measure against.
- **BINIUS-425** ("Create a MerkleIPVerifierChannel for constructing a recursive circuit"),
  In Progress, has an unanswered "can this be closed now?" comment despite its two linked
  PRs (#2104, #2093) being merged — status is stale/ambiguous and should be resolved.
- **BINIUS-428** ("Recursion-aware FRI parameter selection", Backlog) is explicitly gated:
  "not needed to get recursion working — do this only once a recursive circuit exists to
  measure against."
- **BINIUS-502** ("Pair two independent Merkle openings' path climbs into shared hash
  cores", In Review) is filed as "PR-3 of the Flock recursion catch-up analysis" — confirms
  the team is already actively porting cost optimizations from `succinctlabs/flock`.
- **BINIUS-507** is filed under the *Chip Architecture* project, not Recursion, and has no
  Linear links to BINIUS-503/505/506 (the sibling chip-dispatch stubs, all untouched
  Todo/Backlog) even though its own PR description says "nothing yet wires it into a real
  protocol driver" — the natural follow-up isn't tracked anywhere yet.
- XMSS-specific work (BINIUS-162/278/410/482/488/497/498/501) is **finished** and folded
  into the general chip-architecture direction (BINIUS-278, the batched-XMSS-on-M4
  benchmark issue, was explicitly canceled in favor of BINIUS-498's chip-based approach).
  No open XMSS aggregation/recursion issue exists in Linear today — this plan is filling
  that gap.

## 3. External prior art

### `succinctlabs/flock`, `recursion_circuit` branch

A sibling codebase (credits binius64 for `ghash.rs`, LCH-NTT, `tensor_algebra.rs`,
ring-switch `eval_rs_eq`) with a different top-level architecture (zerocheck+lincheck PIOP
over R1CS-over-GF(2), Ligerito PCS) but a **fully worked recursion tower** that is the
closest thing to prior art for BINIUS-331's "crux" step:

- **Claim deferral**: verifier checks return a `MatrixAssertion`/`ElementAssertion` instead
  of being discharged inline (`crates/flock-core/src/aggregate.rs`).
- **Constant-size accumulator**: `Accumulator` folds many proofs' claims via two sumchecks
  per level; only the root ever pays the real evaluation cost.
- **k-ary merge node**: `build_fl_node_k(cfg, cps: &[&ChainProof])` folds `k` leaf proofs
  into one, enforcing adjacency via `k-1` copy constraints between segment endpoints
  (`crates/flock-prover/src/tower.rs`).
- **Demonstrated self-recursion**: the design doc (`docs/folding-verifier-design.tex`)
  states and sizes the fixed point directly — "a recursion circuit that pays it inline can
  never verify a proof of itself. Deferral plus folding is what makes the fixed point
  close" — and `docs/circuit-wiring-design.tex` sizes the gate inventory needed for the
  circuit to express its own verifier (~10.2k SHA-256 compressions + ~46k F-multiplications
  per verifier replay).
- **No XMSS anywhere in this branch** — its only workload is BLAKE3/SHA-256 compression
  chains. The XMSS-specific wiring is genuinely unaddressed on both sides.

### `leanEthereum/leanVM-b`

A separate, self-contained, general-purpose verifiable VM (own 7-instruction ISA over a
64-bit binary-field tower, own zkDSL compiler, own M3-style bus/GKR prover, own WHIR/Ligerito
PCS over a 192-bit field) built as a candidate SNARK-aggregation engine for lean Ethereum's
BLS->XMSS / ECDSA->SPHINCS+ post-quantum migration. Its README credits Binius/Binius64 for
ring-switching and M3 arithmetization as research lineage, **not** as a code dependency —
`Cargo.toml` has no binius/binius64 dependency anywhere.

**Direct answer to "should binius64 target this VM for recursion": no.** leanVM-b is not a
compile target binius64 can emit a verifier program into and then prove with its own
AND-circuit machinery — it brings its own complete, independent proving stack. Adopting it
would mean adopting its entire stack, not extending binius64's.

**What it *is* valuable for**: its `rec_aggregation` crate is a working, benchmarked
(891 XMSS/s, 1,800-signer 2-to-1 recursion in 0.588s) instance of exactly "XMSS aggregation
recursion," and its design is directly reusable *as a pattern*, not as code:

- A guest program (`crates/rec_aggregation/guests/aggregate.py`, in zkDSL) verifies GKR
  sumcheck rounds, Merkle paths, BLAKE2s, Fiat-Shamir, and XMSS/WOTS signatures in-circuit,
  compiled to leanVM-b bytecode and proven by leanVM-b's own pipeline.
- **Deferred-claim batching**: fixed expensive polynomials (bytecode multilinear, Flock's
  R1CS matrices) are exported as one deferred claim per node and batched up the tree; only
  the root pays the real evaluation cost — structurally the same idea as flock's
  accumulator, independently arrived at.
- **Write-once signer-coverage argument**: a running-count/bijection argument over write-once
  memory cells ensures every signer across the whole aggregation tree is counted exactly
  once, with overlapping child signer-sets deduplicated into a union. **This is the piece
  neither binius64 nor `succinctlabs/flock` has any equivalent of, and it is specifically
  the XMSS-aggregation-shaped part of the problem** (as opposed to the proof-recursion
  machinery, which is generic).
- The spec explicitly flags that even leanVM-b's own top-level verifier isn't trivially
  self-recursive, because its final PCS-batching challenge is drawn only after every claim
  is bound (a sequential, transcript-dependent step) — worth remembering as a hazard when
  designing binius64's own deferral scheme.

## 4. Incremental PR plan

Sequenced; later PRs depend on earlier ones being merged. Rough sizing: S = small/mechanical,
M = a design doc plus real but contained implementation, L = substantial new subsystem.

**PR 1 (S) — Fix the stale BINIUS-470 doc comment on `recv_openings`.**
No code change: BINIUS-470 is already fixed (the challenger's bit-sampling routine masks
every drawn value below `2^bits` with a real gate, and every real call site derives a
Merkle-opening index from that masked draw), and Linear lists it Done. `channel.rs`'s
`recv_openings` doc comment was never updated after that fix landed and still describes
the pre-fix state, pointing at BINIUS-470 as if it were open work. Rewrite the comment to
describe the current, sound state instead.

**PR 2 (S) — Recursion-crate hygiene.**
Fix the stale `tests/channel_builds.rs` doc comment (it claims paths still `todo!()` that
`channel.rs` fully implements). Resolve the dangling "can BINIUS-425 be closed now?" Linear
comment. Add a Linear follow-up issue (or update BINIUS-507) linking it to BINIUS-503/505/506
so the merge-channel work has a tracked path into the real protocol driver.

**PR 3 (M) — Wire BINIUS-507's merge channel into the real Basefold/FRI driver, and into
`crates/recursion`. Implemented; see status note below.**
Today `MergeVerifierChannel`/`MergeProverChannel` are only exercised by a standalone example.
Wire them into the actual protocol driver, then make `Binius64BuilderChannel` and
`WitnessFillerChannel` consume the merged form so a recursive circuit sees fewer, larger
Merkle commitments per round. Measure the resulting drop in in-circuit Merkle-climb gate
count (currently 738 AND per level, per query, per commitment) — this is a direct,
low-risk win on recursive-circuit size, and BINIUS-502 already establishes the team is
pursuing this class of optimization ("Flock recursion catch-up" PR-3).

**Status note, added after implementation.**
The wiring is additive: `BaseFoldVerifierCompiler`/`BaseFoldProverCompiler` each gained a
`create_merged_channel*` constructor that wraps the existing (untouched) `create_channel*`
constructor in the round-merging decorator, and `Verifier`/`Prover` (the plain, non-ZK,
non-Spartan driver) now build their channel through it; `crates/recursion`'s
`recursion_end_to_end.rs` does too.

Two things worth flagging for review, discovered while implementing this, that change what
"measure the resulting drop" means in practice:

- **A compiler built for the merged shape needs the merged (coarse) oracle specs, not the
  fine ones, computed by a dry run through the decorator** — mirroring the existing
  `IOPVerifier::oracle_specs` dry-run pattern, now added as `IOPVerifier::merged_oracle_specs`.
  `Verifier`/`Prover` each gained a `fine_oracle_specs` field to carry the one-per-oracle list
  the merge decorator needs at proof time, alongside the coarse-shaped compiler.
- **The plain (non-ZK) Binius64 protocol commits exactly one oracle, ever** — confirmed by
  reading `IOPVerifier::verify`, which calls `recv_oracle` exactly once. The round-merging
  decorator's own doc comment says a round of one oracle is "forwarded unchanged, at zero
  cost," which is exactly what happens here: measured before/after on the CRC-64 recursion
  demo (`recursive circuit: 1840063 gates, 437129 AND, 365235 BMUL, 119713 ZERO, 4341
  recorded inputs`), the numbers are bit-for-bit identical. **The wiring is correct and
  tested, but delivers zero measurable benefit for this driver and for `crates/recursion`'s
  current demo, because neither ever commits more than one oracle per round.**
- **Where the real payoff lives**: `crates/m4-verifier`'s composite verifier
  (`IOPVerifierM4`, in `crates/m4-verifier/src/composite.rs`) commits one oracle for its
  "main" chip *plus one oracle per numbered chip* — exactly the multi-oracle-per-round shape
  BINIUS-507 targets, and exactly the "chip architecture" context BINIUS-507 was filed
  under in Linear. Wiring the merge decorator into `crates/m4-verifier`/`crates/m4-prover`'s
  composite path (not attempted here) is a distinct follow-up with a real, non-zero payoff —
  added below as PR 3b.
- **Deliberately left out of scope**: `crates/spartan-verifier`/`spartan-prover`'s
  `ZKWrappedChannel`/`ZKWrappedProverChannel` hold the concrete `BaseFoldVerifierChannel`/
  `BaseFoldProverChannel` type as a struct field (not just the `IOPVerifierChannel` trait),
  so `ZKVerifier`'s Spartan-wrapped path structurally cannot accept the merge decorator
  without first generalizing those wrapper channels over any `IOPVerifierChannel` — a
  separate, larger change. `crates/m4-verifier`/`m4-prover`, `crates/iop-prover`'s
  `logup_star.rs` test, and benches were left untouched by construction: only new methods
  were added, no existing method's signature changed, so every existing caller is
  unaffected.

Verification run: `cargo check --workspace --all-targets`, `cargo clippy` on the five
touched crates (`binius-iop`, `binius-iop-prover`, `binius-verifier`, `binius-prover`,
`binius-recursion`), `cargo +nightly fmt --all -- --check`, `cargo test -p binius-recursion`,
and `cargo test -p binius-prover --test prove_verify` (17 integration tests over several
constraint-system shapes, including the untouched ZK/Spartan path) — all clean.

**PR 3b (M, new) — Wire the merge channel into the M4/chip composite verifier and prover.**
`IOPVerifierM4::oracle_specs` (`crates/m4-verifier/src/composite.rs`) chains the main chip's
oracle with one oracle per numbered chip — the first place in the codebase where multiple
oracles genuinely land in the same interaction round. This is where PR 3's mechanism should
actually be pointed to get a measured Merkle-commitment reduction, and it is the more
direct route toward the XMSS chip work (BINIUS-498) this whole plan is ultimately for.
Needs its own dry-run-derived merged oracle specs, mirroring what PR 3 added to
`IOPVerifier`, adapted for the composite (`main` + per-chip) shape.

**PR 4 (M) — Close the loop: actually prove the recursive circuit, and test depth-2
recursion.**
Extend `recursion_end_to_end.rs`'s pipeline (which currently stops at
`populate_wire_witness`) to call `Prover` on the resulting recursive circuit and verify
*that* proof. Add a depth-2 test: build a circuit that verifies a proof of the recursive
circuit itself. Give this a reusable API surface (not just test-internal glue) and a small
`examples/` binary. This is the first point at which "recursion" is demonstrated
end-to-end rather than "verification-in-circuit."

**PR 5 (M) — Demonstration-only symbolic "monster" evaluation (BINIUS-434's recommended
first step).**
Evaluate `compute_public_value` fully symbolically in-circuit as a deliberately expensive
baseline (documented cost: ~13 BMUL per inner AND constraint). Purpose is to get a working,
measurable recursive-verifier circuit to benchmark against before investing in the deferral
machinery below — per BINIUS-331's explicit ordering ("2 before 3"), this has to exist
before opening-discharge/accumulation work can be evaluated for real gain.

**PR 6 (L) — Design: claim-deferral / accumulator primitive.**
This is BINIUS-331's "crux, the piece with no complete prior art" for binius64 specifically
— but `succinctlabs/flock`'s `Accumulator`/`aggregate.rs` design (assertions instead of
inline discharge, constant-size fold via two sumchecks per level, single root-level payment)
is a workable template to adapt, not invent from scratch. Deliverable: a design doc (mirror
`docs/folding-verifier-design.tex`'s structure) plus the accumulator type and fold logic in
`crates/recursion`, replacing PR 5's inline evaluation for the expensive fixed polynomials.
Highest-risk, highest-value item in this plan — budget real design time before coding.

**PR 7 (L) — k-ary merge node: verify-and-fold k inner proofs.**
Build the actual aggregation primitive: an API that takes k inner proofs (using PR 6's
deferred claims for their expensive parts) and folds them — public outputs, deferred claims,
and Merkle-opening structure — into one outer statement, modeled on flock's
`build_fl_node_k`. This is the first PR that makes "aggregation" (as opposed to
single-proof recursion) real in binius64.

**PR 8 (M) — Self-recursion / fixed-point closure test.**
Verify a proof of PR 7's merge-node circuit using another instance of itself, and measure
the per-level constraint-count increment (flock's own analysis shows this needs to be
near-constant, not diverging, for the tower to close). This is the checkpoint that confirms
the deferral design actually works before building the XMSS-specific layer on top of it.

**PR 9 (M) — XMSS-specific tree wiring, including a signer-coverage argument.**
Connect `crates/circuits/src/hash_based_sig` to PR 7/8's aggregation primitive:
- Leaf level: existing `circuit_xmss_multisig`/chip-based verification (already shipped,
  BINIUS-498) proves one shard of signers.
- Internal levels: PR 7's merge node folds shard proofs.
- New piece with no existing binius64 or flock equivalent: a signer-set coverage argument
  ensuring every signer across the whole tree is counted exactly once, with no gaps or
  double-counting across shards. Use leanVM-b's write-once-memory-cell + running-count
  design (`crates/rec_aggregation/src/aggregation.rs`, `guests/aggregate.py`) as the
  reference pattern — it is the only prior art for this specific sub-problem, even though
  the code isn't reusable across the two stacks.

**PR 10 (M) — Recursion-aware FRI/Basefold parameter selection (BINIUS-428).**
Only meaningful once PR 8/9 give a real aggregation circuit to measure against, per the
issue's own stated gating. Re-tune inverse rate, terminal codeword size, and verify-layer
depth specifically to minimize AND/BMUL cost in the merge-node circuit, using BINIUS-428's
existing lever analysis as a starting point (flagged stale, pending remeasurement against
BINIUS-437/435).

**PR 11 (L, stretch) — Spark-style succinct verification of the "monster" polynomial.**
BINIUS-331's step 4, the succinctness endgame: only pursue if BMUL cost at the aggregation
root remains prohibitive after PR 6-10. Longer-horizon research item, not required for a
first working XMSS-aggregation pipeline — call this out explicitly rather than blocking on
it.

## 5. Suggested sequencing summary

```
PR1 (stale-doc fix) --+
PR2 (hygiene)         +--> PR3 (merge-channel wiring, done, no-op today) --> PR4 (prove+depth-2 test)
                       |                                     |
                       |                                     +--> PR3b (wire M4 composite: real payoff)
                       |                                          |
                       +------------------------------------------+--> PR5 (symbolic baseline)
                                                                   |        |
                                                                   |        v
                                                                   |   PR6 (deferral/accumulator design)
                                                                   |        |
                                                                   |        v
                                                                   |   PR7 (k-ary merge node)
                                                                   |        |
                                                                   |        v
                                                                   |   PR8 (self-recursion closure test)
                                                                   |        |
                                                                   |        v
                                                                   |   PR9 (XMSS tree + signer coverage)
                                                                   |        |
                                                                   |        v
                                                                   +--> PR10 (FRI param re-tuning)
                                                                            |
                                                                            v
                                                                       PR11 (Spark, stretch)
```

PR1-PR4 are low-risk and can start immediately. PR6 is the pivotal design decision for the
whole plan — everything from PR7 onward depends on getting the deferral/accumulator
primitive right, so it's worth treating as its own design review before implementation,
using `succinctlabs/flock`'s `docs/folding-verifier-design.tex` as the primary reference.
