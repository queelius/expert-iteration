# Solve-claim COMPUTE: substitution-verified solutions for equation goals

Status: spec (nothing built). Motivated by E10 in the experiment ledger
(mini solves casework-v1 directly at 100%; the harness plateaus at 20-25%,
so the entire gap is protocol overhead) and by the probe-3/4/5 failure
taxonomy (the ground-re-expression dance around "solve this equation" is
the largest identified single tax). Adoption is gated on E13, a
pre-registered on/off ablation.

## 1. The problem

Casework's native step is "solve `4x - 16 = 4` for x". Today's COMPUTE lens
forbids unknowns, so the generator must solve the equation silently and
re-emit the answer as ground arithmetic, `(4 + 16)/4 = 5`. Two defects:

1. **Expressiveness tax**: the re-expression is the step the generator
   most often fumbles (evaluating f instead of inverting it, restating the
   equation, plugging a guess back in), and it is pure ceremony.
2. **Verification misses the point**: the verifier checks the
   re-expression's arithmetic, NOT whether 5 solves the case equation. A
   correct-arithmetic wrong-solution fact ("(4 - 16)/4 = -3" for the +4
   case) verifies today and poisons the aggregate.

## 2. The change

COMPUTE targeting a goal whose statement contains an equation in one
unknown MAY return a **solve-claim**: `x = 5`.

Verification of a solve-claim substitutes the claimed value into the
target's parent-goal equation and checks satisfaction symbolically:

    claim "x = 5", case goal "4x - 16 = 4"
    -> substitute: 4*5 - 16 = 4 -> simplify -> True

This is a STRONGER check than today's: it verifies the actual solving
step, where the current check verifies only the arithmetic of a
re-statement the solving already produced.

## 3. Mechanics (no reductio core change)

The reductio `Verifier` protocol already carries a context slot:
`check(statement, verb, expected)`. Today `_do_verify` passes
`expected=None` (decide mode). The change is confined to three files:

1. **`deliberation_mdp._do_verify`**: when the VERIFY target is a derived
   fact whose statement has free symbols, pass the statement of the fact's
   parent goal as `expected` (the substitution context). Ground facts keep
   `expected=None` exactly as today.
2. **`verifier_adapter.MathVerifier.check`**: new branch. If `expected` is
   a goal statement and the claim parses as `Eq(symbol, value)` with that
   symbol free in an equation parsed from `expected`: substitute and
   `simplify`. Guards: the goal equation must be non-tautological and must
   actually contain the claimed symbol; anything unparseable falls through
   to the existing decide path (which fails closed on unknowns, as today).
3. **`lenses._fields_description(COMPUTE)`**: teach the form. When the
   target goal's statement contains an equation with an unknown, the
   statement may be either a ground equation (as today) OR the solution
   directly: `{"statement": "x = 5"}`.

Terminal verdict: unchanged. `x = 5` already yields RHS `5` under the
existing extraction, and syntactic identities are already rejected.

Aggregation: unchanged. Case facts `x = 5` and `x = -2` aggregate at the
root with the same ground forms as today (`5 + (-2) = 3`,
`max(5, -2) = 5`).

Configuration: `DeliberationMDP(solve_claims: bool = False)` plus a runner
flag `--solve-claims {on,off}` (default off). The default flips to on only
if E13 adopts.

## 4. Invariant analysis (what this touches, what it does not)

Untouched, load-bearing:
- Symbolic verifier is the only reward source; terminal +1/-1 at root
  commit against the dataset label; no learned reward anywhere.
- VERIFY/COMMIT/ABANDON make no LLM call.
- Statements sanitized against marker injection.
- Structure never gates.

Touched, and why it is safe:
- **"COMPUTE takes no unknowns" was a lens instruction, not a core
  invariant.** It was introduced as an implementation convenience so that
  `MathVerifier.decide` (ground-truth evaluation) could be the only check.
- **The substitution context is LLM-authored** (the case equation was
  written by the generator during SPLIT). So a solve-claim VERIFY checks
  internal coherence (claim solves case), not external truth. This is the
  same trust level as today's intermediate checks; the external anchor
  remains, as always, the terminal check against the dataset label. A
  generator that writes a wrong case equation and then solves it correctly
  produces a coherent-but-wrong branch, loses at the terminal, and earns
  -1: exactly the behavior the reward design intends.

Honesty note for the paper: intermediate verification under solve-claims
is coherence checking relative to LLM-authored case structure; only the
terminal check is anchored to ground truth. This was already true of
ground-COMPUTE (the RELEVANCE of a ground fact was never checked); the
solve-claim extension narrows the gap between what is checked and what
matters, it does not close it.

## 5. Scope (v1) and deferrals

In scope: single-variable claims `x = value` against a parent goal whose
statement contains exactly one parseable equation in that variable.
Rational and integer values.

Deferred:
- Domain-condition checking (a case's `x > d` side condition): extracting
  conditions from free-text goal statements is fragile; the dead-case
  decision remains the controller's job via `abandon`, as today. v2 could
  carry conditions structurally from SPLIT.
- Multi-variable claims, systems, inequality solutions (interval answers).
- Set-valued claims ("x = 5 or x = -3"): the single-scalar answer
  convention (casework spec constraint 2) still governs.

## 6. Tests

1. MathVerifier substitution: true claim passes; false claim fails; claim
   for a symbol absent from the goal fails; tautological goal equation
   fails; non-equation goal falls through to decide (fails closed).
2. `_do_verify` passes the parent-goal statement exactly when the fact has
   free symbols; ground facts still verify in decide mode.
3. Lens: equation-goal COMPUTE suffix offers both forms; non-equation
   goals keep today's text verbatim.
4. Off by default: with `solve_claims=False`, behavior is byte-identical
   to today (regression guard for E13's control arm).
5. End-to-end: an episode that wins via split -> solve-claim -> verify ->
   commit per case -> root aggregate, under EchoLLM-style scripted client.

## 7. Acceptance

E13 (pre-registered in the ledger): same teacher, interface, and 20
problems; solve-claims on vs off. Adopt (flip the default) iff the on-arm
solve rate is at least 2x the off-arm; otherwise the tax is protocol
length per se and the granularity question reopens at the design level.
