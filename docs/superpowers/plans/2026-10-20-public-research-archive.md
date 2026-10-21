# Public Research Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or superpowers:subagent-driven-development if the user chooses delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make this repository a readable, reproducible research archive with a small teaching example, carefully supported findings, and a release suitable for GitHub and one Zenodo record family.

**Architecture:** Preserve existing code, experiment logs, and historical paths. Add a short reader-facing guide and an offline analysis layer that derives selected findings from preserved evidence. Treat release preparation and public publication as separate stages.

**Tech Stack:** Markdown, existing Python 3.11+ package and pytest suite, JSON/JSONL evidence, Git, GitHub, and the configured Zenodo uploader.

**Spec:** The approved scope below records the repository-organization proposal discussed with the user on 2026-10-20. Existing scientific contracts remain governed by `CLAUDE.md`, `design.md`, and the experiment ledger.

**Status:** The bounded archive and its verification were completed on
2026-10-20. Publication review, GitHub release, Zenodo publication, public
download verification, and central indexing were completed on 2026-10-21.
See `docs/publication.json` and the central record `expert-iteration-lessons`.
GitHub read-only archival follows the final receipt commit; its verified state
is recorded centrally. Remaining unchecked historical-scope steps are not
claims of completed full-suite or independent mathematical verification.

## Approved Scope

Publish a useful research artifact, not a claim of a novel successful algorithm.
Explain the original policy-learning/search idea separately from the later
frozen-LLM orchestration experiments. Preserve negative results and limitations.
Readers should be able to check the reported numbers without buying model
access, and run a small local example to understand the original idea.

Do not restart research, redesign the controller, or complete the paper scaffold
as part of this effort. Do not claim that failed imitation or orchestration
experiments tested and disproved the full proposed RL/search loop.

## Global Constraints

- RL first, MCTS second. MCTS is a strictly additive accelerator, never the spine.
- The symbolic verifier is the ONLY reward source. This archive work introduces no new reward source or changes to historical scoring.
- Python 3.11+. Use structured JSON/JSONL readers and existing typed experiment APIs, not regex extraction of structured records.
- Never write the em-dash glyph in any file.
- Preserve all unrelated edits. `git add` explicit paths only; never `git add -A`, `git add -u`, or `git add .`.
- Do not move or overwrite historical runs, including incomplete or excluded runs. Do not silently repair evidence.
- No new paid inference, model downloads, or training runs are required for this release.
- Existing sibling projects retain ownership of their results. Link them; do not import their findings as this repository's evidence.
- Public findings describe bounded results and lessons. Detailed debugging history remains in the ledger.
- No repository visibility change is authorized by this plan. Review publication destination and release contents before disclosure.
- Read `/home/spinoza/github/research-records/PUBLISHING.md` again before publishing; search existing records before creating a deposition.

## Review Focus

1. Cached outputs, repeated attempts, or reused problems must not be presented as independent fresh evaluations. Tasks 1 and 2 check grouping and provenance.
2. Partial, quarantined, or malformed runs must never silently enter headline totals. Task 2 tests exclusion and fail-closed handling.
3. A working local checkout must not conceal missing files, credentials, model caches, or sibling-repository dependencies. Tasks 3 and 5 test a standalone packet.
4. A release must not expose private operational details, incompatible third-party material, or unrelated work. Tasks 1 and 5 audit files and reachable history.
5. Historical success claims and old test counts must not become current verification claims. Tasks 2, 4, and 5 distinguish measured, implemented, proposed, and unverified work.

## Intended File Layout

Keep `src/`, `tests/`, `experiments/`, `datasets/`, `research-output/`, and
historical document paths intact for the first release.

| File | Responsibility |
| --- | --- |
| `README.md` | Short public entry point, current archive status, three reader routes |
| `docs/research-overview.md` | Questions, original idea, later experiments, limits of what was tested |
| `docs/findings.md` | Selected findings with evidence, controls, limitations, and analysis links |
| `docs/reproduction.md` | Offline table reproduction, local tutorial, optional hosted reruns |
| `docs/tutorial.md` | Small worked example using the existing classical implementation |
| `docs/project-map.md` | Code map, historical documents, sibling projects and their separate questions |
| `docs/release-readiness.md` | Evidence inventory, licensing, verification results, publication checklist |
| `analysis/claims.json` | Structured inventory of selected claims and their evidence |
| `analysis/reproduce.py` | Validate evidence and regenerate selected tables without model calls |
| `analysis/tables.json` | Generated, machine-readable table values |
| `analysis/tables.md` | Generated tables linked from the findings |
| `tests/test_archive_reproduction.py` | Analysis integrity and negative controls |
| `tests/test_archive_tutorial.py` | Tutorial smoke test and no-network check |
| `CITATION.cff` | Canonical authorship and exact release citation |
| `LICENSE`, `LICENSES/README.md` | Approved project license and component-specific rights/attribution |

Use existing runners for the tutorial. Add a new runner only if none can provide
a small, deterministic demonstration without changing the research methods.
Avoid adding a website, a documentation framework, or a general experiment platform.

## Task 1: Inventory the Evidence and Decide the First Release's Scope

**Files:** Create `docs/release-readiness.md` and `analysis/claims.json`.
Read `HANDOFF.md`, `CLAUDE.md`, `design.md`, the full experiment ledger,
`experiments/archive/README.md`, dataset cards, and relevant raw artifacts.

**Interfaces:** The inventory produces the list of claims and required inputs
consumed by Task 2, and the release exclusions consumed by Task 5.

- [ ] Record branch, HEAD, tracked modifications, untracked files, and relevant ignored files. Do not assume the current commit contains the later research.
- [ ] Inventory the classical examples, completed experiments, proposed work, partial runs, quarantined runs, and sibling-owned follow-ups.
- [x] For each candidate public finding, record its precise wording, experiment IDs, raw files, analysis dependencies, independent problem count, repeated-measure structure, and known limitations.
- [ ] Use JSON records with `id`, `statement`, `experiment_ids`, `status`, `inputs`, `analysis`, `unit_of_analysis`, `limitations`, and `exclusions`. Each input records a relative path and SHA-256. Status is one of `candidate`, `verified`, or `excluded`; excluded entries include a reason.
- [ ] Start with three candidate comparisons: E20's imitation-controller transfer test; the E41/E42 fixed-revision and routing comparison; and E47's context-locality and target-selection screen. Add E19/E22 only if the underlying action labels, split, and comparator can be independently checked without new model calls.
- [x] Distinguish recalculating a metric from saved outcomes from replaying and rechecking symbolic correctness. Record which level is possible for each finding; summaries alone do not count as independent reproduction.
- [ ] Audit licenses and provenance of code, vendored code, datasets, saved model responses, and checkpoints. Check repository remote and visibility read-only. Inventory sensitive operational material without copying secrets into the report.
- [ ] Define a small release allowlist and explicit exclusions. If key evidence is missing, narrow the public claim or exclude it rather than repairing the gap by assertion.

**Done when:** Every proposed headline result has a traceable evidence path and
a stated reproduction level. Unknown licensing and missing evidence are named
blockers, not silently marked resolved.

## Task 2: Build and Test Offline Table Reproduction

**Files:** Create `analysis/reproduce.py`, `analysis/tables.json`,
`analysis/tables.md`, and `tests/test_archive_reproduction.py`.
Update the Task 1 manifest and readiness report.

**Interfaces:** `python analysis/reproduce.py --manifest analysis/claims.json
--output-dir PATH` validates listed inputs and writes `tables.json` and
`tables.md`. It must exit nonzero on missing or mismatched evidence. It consumes
no provider credentials and never calls a model. Reuse existing experiment
replay/summary functions where available; implement only the selected analyses.

- [ ] Inspect the selected record schemas and existing replay functions before writing readers. Explicitly dispatch on supported record contracts; reject unknown contracts.
- [ ] Write failing tests for a tiny paired fixture with known counts, including one win, one loss, one tie, and repeated observations from one problem.
- [ ] Write failing tests for missing inputs, hash mismatch, duplicate problem/arm/repetition keys, an unexpected contract, and a partial or quarantined input incorrectly marked as headline evidence.
- [x] Add a test that denies outbound network access during reproduction. Do not merely unset API keys.
- [x] Implement validation and table generation. Derive counts from records; never paste ledger totals into the computation.
- [ ] Preserve paired comparisons and use the original problem/state grouping for uncertainty where applicable. Show unique problem counts separately from calls and repetitions. Label descriptive results and exploratory analyses explicitly.
- [x] Run `uv run pytest -q tests/test_archive_reproduction.py`.
- [ ] Reproduce the selected actual artifacts twice into separate temporary directories and compare generated values. Investigate any discrepancy against the ledger; retain both the old claim and a dated correction if necessary.
- [x] Mark a claim `verified` only after the supported reproduction level succeeds. Record the command, environment, result, and remaining limits.

**Done when:** A reader can regenerate every selected headline number offline,
and damaged, incomplete, or misidentified evidence produces a clear failure.

## Task 3: Provide One Runnable Teaching Example

**Files:** Create `docs/tutorial.md` and `tests/test_archive_tutorial.py`.
Reuse `experiments/run_value_iteration.py`, `experiments/run_mcts.py`,
`experiments/run_expert_iteration.py`, and `tests/test_expert_iteration.py`
as appropriate.

**Interfaces:** The tutorial supplies one exact local command, its dependency
set, expected output structure, seed, and bounded runtime to Task 4.

- [ ] Inspect the existing classical runners and select the smallest worked example that demonstrates policy learning with search as an optional accelerator.
- [ ] Explain the state, available moves, reward, policy, search visits, and learning update in plain language before presenting the general loop.
- [ ] Write a smoke test for the documented command with a fixed seed and small budget, checking finite outputs and completion without network access.
- [ ] Check the example against the existing exact-solver tests. Do not require or imply universal monotonic improvement from one demonstration.
- [x] Run `uv run pytest -q tests/test_archive_tutorial.py tests/test_expert_iteration.py tests/test_mcts.py tests/test_value_iteration.py`.
- [ ] Document that the teaching example illustrates mechanics, not a new empirical claim about LLM training or compute efficiency.

**Done when:** A clean local environment can run the example without API
credentials, hosted models, or dependencies on another repository.

## Task 4: Write the Public Guide and Reconcile Stale Entry Points

**Files:** Create `docs/research-overview.md`, `docs/findings.md`,
`docs/reproduction.md`, and `docs/project-map.md`. Modify `README.md`.
Add brief current-status pointers to `HANDOFF.md`, `PAPER_HANDOFF.md`,
`design.md`, and `research-output/README.md` where needed.
Update project-status text in `AGENTS.md` and `CLAUDE.md` without removing
their scientific, safety, attribution, or preservation rules.

**Interfaces:** These pages consume only verified Task 2 tables, the Task 3
tutorial, and explicit historical/proposal labels from Task 1.

- [x] Replace stale README implementation claims with a concise description of the archive and links for learning, checking findings, and exploring code.
- [ ] Explain the original expert-iteration motivation separately from frozen-LLM orchestration. Distinguish designed RL fine-tuning from executed behavior-cloning tests.
- [x] Write one section per selected finding: question, comparison, evidence, limitation, practical lesson, and reproduction command.
- [ ] State that the tested fixed policy was stronger only within the documented comparison; do not generalize this to all workflows or controllers.
- [ ] Explain why the E47 locality result is not proof of better decomposition or a learnable target selector. Preserve small-sample and reused-development-state caveats.
- [x] Document three reproduction levels: offline analysis, local tutorial, and optional hosted reruns. Make clear which model revisions or provider behavior cannot be recreated exactly.
- [ ] Link sibling projects with verified canonical URLs where public. Do not make local `../` paths necessary for using this archive.
- [ ] Label historical specs and unfinished paper drafts without rewriting their recorded history. Keep debugging detail in the ledger rather than making it the public narrative.
- [ ] Check every reader-facing local link and every numerical statement against the generated tables. Record the review in `docs/release-readiness.md`.

**Done when:** A new reader can distinguish implemented, tested, planned,
excluded, and transferred work without reading the entire experiment ledger.

## Task 5: Verify and Package a Release Candidate

**Files:** Add `CITATION.cff`, `LICENSE`, and `LICENSES/README.md` after the
rights review and license decision. Update `pyproject.toml` only as necessary
for accurate authorship, license metadata, and reproducibility dependencies.
Add `.github/workflows/archive-checks.yml` for the offline checks if appropriate.
Record verification in `docs/release-readiness.md`.

**Interfaces:** Produces an allowlisted, self-contained packet and a pinned
source commit for publication. It must not depend on uncommitted files.

- [ ] Resolve license choices with the owner. Preserve component-specific licenses and predecessor credits; do not blanket-relicense third-party material.
- [x] Use Alexander Richard Towell, ORCID `0000-0001-6443-9897`, and `lex@metafunctor.com` in new citation metadata. Disclose substantial AI assistance and actual review scope.
- [ ] Audit the publication file set and any Git history that would become public for credentials, private operational material, and redistribution problems. Do not rewrite history or change visibility automatically.
- [ ] Establish the required environment from actual imports. Run `uv run pytest -q`; record failures and skips rather than quoting historical test counts.
- [ ] Measure focused coverage using pytest-cov in the verification environment. Cover all new error-handling paths in the analysis and tutorial code; report meaningful gaps rather than imposing an arbitrary repository-wide percentage.
- [x] Make offline analysis and tutorial checks runnable in CI without secrets or model downloads. Avoid silently running hosted experiments.
- [ ] Assemble the packet from explicit source/evidence paths, with environment instructions, license notices, generated tables, and SHA-256 checksums. Include all transitive data dependencies.
- [ ] Test the staged packet in a fresh environment with no sibling directories or model caches and network disabled after dependencies are installed.
- [ ] Stage and commit only reviewed explicit paths, including required existing uncommitted work. Never sweep unrelated edits into the release.
- [ ] Pin the release source commit and reproduce the packet from it. A dirty checkout is not an acceptable release source.

**Done when:** The exact packet passes offline reproduction and the tutorial;
tests, limitations, checksums, attribution, and rights are recorded. No public
upload has occurred merely because this stage is complete.

## Task 6: Publish and Verify the Public Record

**Files:** Finalize `CITATION.cff`, README release links, and the release receipt
in `docs/release-readiness.md`. Add a curated record in
`/home/spinoza/github/research-records` following its current conventions.

**Interfaces:** Consumes the verified Task 5 packet. Produces a public source
release, an exact-version archival DOI, and a central-index link.

- [x] Present the exact release scope, destination, license decisions, and unresolved issues to the user. Resolve any required approval before disclosure, especially repository visibility.
- [x] Check official GitHub/Zenodo instructions at execution time and reread the central publishing guide. Search existing deposits with complete pagination; reuse an appropriate version family.
- [x] Publish a reviewed GitHub release from the pinned commit if the destination is authorized. If the current repository cannot safely be public, propose a clean public export without changing the working repository.
- [x] Create or reuse a Zenodo draft using the configured uploader and existing authentication, without printing tokens. Reserve a DOI before finalizing citation metadata when needed.
- [x] Verify metadata, file contents, original disclosure dates, source links, licenses, and the distinction between research artifact and peer-reviewed publication. Keep one coherent record/version family.
- [x] Publish when authorized and ready, then download every deposited file and compare hashes. Verify the public source, record, and exact-version DOI.
- [x] Update the central index with canonical source and DOI links, evidence scope, disclosure dates, and verification receipt. Citation/index follow-up commits must not silently alter the archived evidence.

**Done when:** Another person can reach the source, cite the exact archived
version, download it, and reproduce the selected findings without private access.

## Execution and Stop Rules

- Implement Tasks 1-5 in order, updating these checkboxes and the readiness report as work finishes. Prefer small explicit-path commits at verified task boundaries.
- The first review point is after Task 1: settle which findings belong in this release before building their presentation.
- Do not expand into new scientific experiments to rescue weak findings.
  Missing evidence means a narrower claim or an explicit exclusion.
- New failures in analysis are release blockers. Preexisting unrelated test
  failures require a documented diagnosis and explicit release disposition,
  not unrelated refactoring.
- Licensing, confidentiality, or publication-destination questions go to the
  user. Other routine implementation decisions can be handled locally.
- Task 6 follows the publication gate, not an assumption that preparing this
  plan authorizes immediate public disclosure.

## Suggested Execution Goal

After the user approves implementation:

> Prepare expert-iteration as a pedagogical, reproducible research archive:
> audit the evidence, regenerate selected findings offline, provide one local
> teaching example, write the public guide, and verify a self-contained release
> candidate. Follow this plan through Task 5. No new paid experiments or public
> publication; report readiness and any decisions needed before Task 6.

This bounded goal can finish successfully even if publication needs a separate
license or destination decision. Publishing is a distinct follow-up goal once
those decisions are settled.
