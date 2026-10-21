# Expert Iteration: Experiments and Lessons

A research archive about learning to guide search and coordinating a frozen
language model. It contains runnable teaching examples, experimental code,
saved model outputs, and lessons from comparisons that did not establish the
hoped-for controller advantage.

**Status:** archival release 1.0.0. Read or cite the exact evidence packet at
[Zenodo, DOI 10.5281/zenodo.23526852](https://doi.org/10.5281/zenodo.23526852).
The repository preserves the wider research history. It contains unfinished
manuscript drafts and does not claim a new successful control algorithm.

## Start Here

- [Understand the research](docs/research-overview.md): what we wanted to learn,
  what we built, and what we actually tested.
- [Read the findings](docs/findings.md): three bounded comparisons and their
  limitations.
- [Check the numbers](docs/reproduction.md): regenerate tables from saved
  records without buying model access.
- [Run the teaching example](docs/tutorial.md): a small policy-learning and
  search loop on a grid.
- [Explore the project](docs/project-map.md): code, historical plans, and
  separate follow-up projects.

## What We Learned

The tested imitation controller did not transfer successfully to a different
worker model. In a later shared-output comparison, always taking an independent
revision produced more correct candidates than the tested coordinator selected.
Focused context helped relative to full context in a small screen, but the
selected fixed child workflow did not beat direct prompting.

These are different experiments, not a single cumulative benchmark. They do
not show that small controllers are inherently unsuitable, or that the original
RL/search idea failed. See the [findings](docs/findings.md) for denominators,
controls, costs, and limits.

## Quick Offline Check

From the repository root, with Python 3.11 or newer:

```bash
python analysis/reproduce.py --output-dir /tmp/expert-iteration-tables
```

This uses only the Python standard library. It checks the hashes of five
selected input files and rebuilds eight descriptive comparisons from individual
records, not copied aggregate totals. It does not rerun models or independently
recheck all mathematical answers.

For the tutorial and focused tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_archive_reproduction.py tests/test_archive_tutorial.py
```

The wider historical suite requires additional optional dependencies; see
[reproduction](docs/reproduction.md). Installation can use the network, but the
table analysis and teaching example do not.

## Scope, Credit, and License

The original aim was **RL first, MCTS second**: improve a learned policy, with
search as an optional accelerator. The tabular teaching implementation exists.
Later frozen-LLM experiments tested teacher control and imitation, but did not
complete the planned off-policy RL study.

Author: **Alexander Richard Towell**, Southern Illinois University Edwardsville.
Substantial AI assistance was used in coding, experimentation, and writing.
Automated tests and AI review are not human peer review.

Original code is [MIT licensed](LICENSE). The new research notes are
[CC BY 4.0](LICENSES/README.md); dataset and third-party terms remain separate.
See [CITATION.cff](CITATION.cff) for the exact release citation. Model outputs
are identified as machine-generated evidence; model and benchmark notices are
retained in [NOTICE](NOTICE).

See [release readiness](docs/release-readiness.md) for verification and
exclusions, and the repository's
[publication metadata](https://github.com/queelius/expert-iteration/blob/main/docs/publication.json)
for the final publication receipt.
