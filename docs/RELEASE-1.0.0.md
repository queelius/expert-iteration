# Expert Iteration: Experiments and Lessons

Version 1.0.0, 2026-10-21. Alexander Richard Towell, Southern Illinois
University Edwardsville. ORCID: https://orcid.org/0000-0001-6443-9897.

Citation: https://doi.org/10.5281/zenodo.23526852.
Source: https://github.com/queelius/expert-iteration.
Exact source commit: `a535c2b34f18894d3228090dce048b30ade3badd`.

This research artifact preserves experiments on search-assisted policy
learning and coordination of a frozen language model. It provides a local
teaching example, saved outcomes, offline analysis, and lessons from results
that did not establish the hoped-for controller advantage. It is a
pedagogical research record, not a peer-reviewed paper.

## Results and Limits

- E20: direct Qwen2.5-1.5B answered 8/20 original problems correctly; the
  tested imitation controller with that worker solved 0/20. This was a
  transfer test of behavior cloning, not completed off-policy RL.
- E41/E42: on 60 reused problems, a fixed independent revision produced
  55/60 correct candidates, while the improved GPT-4.1-mini routing prompt
  selected outcomes solving 51/60. The coordinator replayed saved worker
  outputs and revised fewer problems; this is not an equal-cost comparison
  or fresh end-to-end confirmation.
- E47: child-local context produced 33/72 correct paired evaluation outcomes
  versus full context's 24/72. Those observations came from only 12 states.
  The discovery-selected fixed child tied direct prompting at 10/24, and
  state-specific selection did not meet the registered improvement criteria.

The proposed full study of RL/search sample efficiency was not completed.
These experiments do not establish a general limit on small controllers or
learned orchestration. RL remains the original learning objective, with MCTS
as an optional accelerator. The tabular example illustrates the mechanics.

## Reproduction

Extract `expert-iteration-v1.0.0.zip`, then run from its root:

```bash
python analysis/reproduce.py --output-dir /tmp/expert-iteration-tables
python -m pip install -e ".[dev]"
python -m pytest -q
```

The first command needs only the Python standard library. It validates five
input-file hashes and reconstructs eight descriptive comparisons from saved
individual records. It does not independently rescore all mathematical
answers or rerun model inference. The packet contains 133 source/evidence
files plus `ARCHIVE-MANIFEST.json`, which pins every file and the source commit.

All 59 included tests passed in 4.60 seconds in a fresh Python 3.11.15
environment with provider credentials absent and Python socket operations
denied. Torch, SymPy, running models, and sibling checkouts were absent.
The extracted tables matched the archived copies byte for byte. GitHub CI
also passed on the exact source commit.

The wider historical suite has separate unresolved timeouts in three older
symbolic-replay tests. Its full results and dependency qualifications are
preserved in the source documentation; this release does not certify that
entire historical suite.

## Contents, License, and Credit

The compact ZIP supports the declared reproduction route. The GitHub
repository preserves the wider ledger, scripts, tests, diagnostics, historical
specifications, and unfinished drafts. The latter are not completed papers.

Original code is MIT. New research notes, including this release note, are
CC BY 4.0. Dataset and third-party rights remain separate; retain `NOTICE`
and `LICENSES`. Qwen2.5-3B-Instruct uses the Qwen Research License. No worker
model weights are included in the evidence packet, and its model terms are
not replaced by the code's MIT license. Generated responses are identified
as machine-generated historical evidence.

Substantial AI assistance contributed to implementation, analysis, review,
writing, and publication preparation. A separate AI review found no important
defects within the archive scope. This is not human peer review or independent
verification of all mathematical answers.

The original private experiment/commit dates are development dates. The
public disclosure and DOI publication dates are recorded separately in the
source publication receipt and central research index.

Central record:
https://github.com/queelius/research-records/tree/main/records/expert-iteration-lessons.
