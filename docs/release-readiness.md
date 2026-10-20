# Research Archive Release Readiness

Updated: 2026-10-20. Status: preparation in progress, not published.

## Scope and Progress

Plan: [public archive implementation](superpowers/plans/2026-10-20-public-research-archive.md).
This record tracks work, decisions, and checks. It is not a scientific result.

- Evidence inventory: selected first-release scope recorded.
- Offline table reproduction: eight comparisons reconstructed, 16 tests passed.
- Teaching example: short-budget CLI added; 23 tutorial/classical tests passed.
- Reader guide: overview, findings, reproduction, tutorial, and project map written.
- Standalone packet and publication checks: pending.
- Publication, GitHub read-only archival, and local move: not performed.

Recommended GitHub name: retain `expert-iteration`. Reader-facing title:
**Expert Iteration: Experiments and Lessons**. Preserve existing links.
After publication and verification, mark the GitHub repository archived.
The owner intends to move the local checkout to
`~/github/archives/expert-iteration` afterward; do not move it during this work.

## Execution Decisions

- Ruling: work in the existing `action-space-redesign` checkout. Most of the
  later research is uncommitted, and a new worktree at HEAD would omit the
  evidence being archived. Preserve all existing edits and avoid source moves.
  Cost: final release requires an explicit reviewed snapshot, not just HEAD.
- Ruling: archive selected record-level outcomes first, not a new hosted
  replication. Offline metric reconstruction is weaker than rechecking every
  mathematical answer and replaying every transport receipt. Label that
  boundary explicitly.
- Ruling: select E20, E41/E42, and E47 for the first reader-facing report.
  E19/E22 remain historical context, not headline claims, until their full
  action-label/comparator reproduction is separately prepared.
- Ruling: keep documentation of the original RL-first, search-assisted aim,
  but do not represent behavior-cloning transfer as a test of completed
  off-policy RL. The original full sample-efficiency claim remains untested.
- Ruling: freeze only the packet's required source/evidence plus archive
  entry-point changes in the preparation commit. Do not include unrelated
  slide edits or silently commit all historical working documents. The full
  historical test run describes the working checkout; the standalone packet
  receives its own focused verification.

Initial HEAD was `c3e000d7f1f0eb6e310cbdcfa5ff6a14dd3e0d9c`.
The later research source and selected evidence were largely uncommitted at
the start. The packet builder reads only committed bytes, never unstaged work.

## Evidence Inventory

| Material | Treatment |
| --- | --- |
| Classical Gridworld, value iteration, MCTS, expert iteration | Runnable teaching example, not a novel efficacy claim |
| E20 direct and imitation-controller records | Selected; compare same 20 problems, not full RL |
| E41 worker revision and E42 coordinator routing records | Selected; shared 60-problem development corpus, replayed worker outputs |
| E47P target/locality records | Selected; 12 source states with repeated calls, no independent 72-problem claim |
| E1 and E28 quarantined conclusions | Excluded from public findings; preserve historical exclusion reasons |
| E21 larger-model screen | Excluded from headline findings due to decoding and logging confounds |
| E31 incomplete decomposition run | Preserve partial locally; exclude from headline analysis |
| E18/E19/E22 imitation analyses | Historical context; no new generalization claim |
| v2 contracts and warm-start implementation | Implemented code is not evidence of successful trained v2 control |
| Unfinished paper and old slide decks | Historical drafts, not publication-ready results |
| Sibling decomposition/intervention projects | Separate ownership; links only after checking destinations |

Every selected input will have a SHA-256 in `analysis/claims.json`.
Historical records remain untouched. The publication packet is allowlisted,
not a recursive copy of the dirty checkout.

## Initial Environment Check

The first `uv run pytest -q` created a fresh Python 3.13 environment with only
the base dependency. It reported 207 failures, 61 errors, 482 passes, and 78
skips; missing SymPy caused many failures. This is not evidence of a new code
regression. Install the declared `dev` and `llm` extras and rerun before
assessing the baseline. Historical test counts are not current verification.

The corrected Python 3.13 run was interrupted after 718.69 seconds with 80
passes and 7 optional-dependency skips, while in SymPy. No failing test had
been reported. A new run uses Python 3.11.15 (matching historical logs), four
pytest workers, and a 120-second per-test timeout. It finished in 501.79
seconds: **876 passed, 64 skipped, 7 failed, 1 error**. Four failures and the
error were missing-Torch dependency problems. After installing
`torch==2.14.1+cpu`, those five checks produced **4 passed, 1 skipped**
(the remaining check needs CUDA).

Three historical tests exceeded the 120-second limit, all in
`tests/reasoning_v2/test_boundary_exact_probe.py`:

- `test_prepare_binds_strict_mode_raw_control_and_unchanged_e28_requests`
- `test_strict_corrected_replication_retries_numeric_raw_and_replays_artifact`
- `test_second_numeric_reply_is_terminal_unusable_and_diagnosed`

Their tracebacks were inside repeated SymPy work. This is an unresolved
runtime limitation, not a proven incorrect mathematical result. No timeout
test was counted as a pass. The full suite was not rerun after installing
Torch, so previously skipped neural tests are not newly certified.

Disposition: retain these limitations in the archive. Do not redesign the
historical verifier to make an archive release pass. The compact packet
supports the selected metric reconstruction and classical tutorial, which
have a separately passing test set.

The new archive tests were observed failing before implementation, then
passing. The original long teaching demo also completed, and a short-budget
run is covered by the focused tests. Model inference was not performed.

## Archive Verification

- Focused archive plus classical checks: **59 passed** on Python 3.11.15.
- Branch-aware focused coverage: **98% combined**; reconstruction 99%,
  packet builder 98%, tutorial runner 97%. The uncovered lines are script
  entry guards; actual CLI commands are also exercised separately.
- `CITATION.cff` validates against CFF 1.2.0 using `cffconvert`.
- Eight reader-facing Markdown files were parsed with `markdown-it-py`;
  all local links resolved.
- A separate AI code reviewer found no critical or important issues in the
  archive additions. This is not human peer review or mathematical rescoring.
- `detect-secrets` scanned the 127 then-selected packet files with network
  verification disabled. Nine flags were traced through parsed JSON to
  evidence SHA-256s, model revision IDs, and a package-tree hash. No credential
  was identified by that scan. This is not an audit of reachable Git history,
  nor a guarantee that automated detection finds every secret.
- Source and archived raw outcomes were not repaired or rewritten to obtain
  agreement. The eight reconstructed comparisons match the selected ledger
  counts.
- Kernel network-namespace isolation was attempted but denied by this
  environment (`unshare` could not write the UID map). Standalone verification
  will instead deny Python socket operations and run without provider
  credentials. This is not an OS-level network sandbox.

## Rights and Destination Review

- Owner approved MIT for original code and CC BY 4.0 for new notes.
- The Casework v1 dataset card already declares CC BY 4.0.
- The local `reductio` predecessor at `7c306c5` lists 16 commits by Alex
  Towell (`lex@metafunctor.com`) and no separate license declaration in its
  project metadata. Keep its source attribution and do not claim third-party
  ownership merely because it was vendored.
- GitHub read-only query on 2026-10-20: `queelius/expert-iteration` is private
  and not archived. No visibility change was made.
- Historical manuscripts, third-party content, and model-response
  redistribution are not covered by a blanket new licensing assertion.

## Release Blockers

- Third-party and dataset rights need review; preserve existing notices.
- Sensitive content and reachable Git history need a publication audit before
  changing visibility or uploading the entire repository.
- Existing uncommitted research must not be swept into a commit without review.
- No public version DOI or publication receipt exists for this archive
  preparation. Local source pin and packet verification are recorded below
  when complete.

## Publication Sequence

Finish offline checks; review the exact packet and remaining blockers with the
owner; pin its source; search existing Zenodo records; create or reuse a record
family; publish only authorized contents; download and hash-check the deposit;
link from the central research index; then archive GitHub read-only.
Record public disclosure dates separately from the archive publication date.
