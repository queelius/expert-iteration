# License Scope and Attribution

The owner approved MIT for original code and CC BY 4.0 for the new research
notes on 2026-10-20. This does not override third-party rights.

## Original Code

Original code by Alexander Richard Towell is covered by the root MIT license.
Preserve any separate notices accompanying incorporated material.

The reasoning-state core at `src/verifier_anchored_rl/reductio/` was adapted
from `queelius/reductio`, source commit `7c306c5`, with the Method node layer
removed. This provenance is recorded in the original vendoring plan. Credit
that predecessor when reusing the state representation. The local predecessor
history through that commit lists Alex Towell as its author; this is the same
person as Alexander Richard Towell. Its project metadata declares no separate
license. Preserve any additional third-party notices if later discovered.

Third-party Python dependencies retain their own licenses; listing them as
dependencies does not relicense them under MIT.

## New Research Notes

Copyright 2026 Alexander Richard Towell.

The following new notes are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/):
the root README, `docs/research-overview.md`, `docs/findings.md`,
`docs/reproduction.md`, `docs/tutorial.md`, `docs/project-map.md`,
`docs/release-readiness.md`, `docs/publication-review.md`, and the archive
implementation plan dated 2026-10-20. Cite the author and archive, identify modifications, and retain
the license link. The [legal code](https://creativecommons.org/licenses/by/4.0/legalcode)
is authoritative.

Historical drafts and working documents are not silently relicensed by this
new-notes declaration.

## Data and Generated Outputs

`datasets/casework-v1/README.md` declares CC BY 4.0 for Casework v1.
Retain its dataset card and provenance when redistributing the dataset.

Saved model responses and diagnostic records are historical evidence, not
model weights. The release-specific review in `docs/publication-review.md`
checked provider ownership clauses, source benchmark notices, and the actual
Qwen model terms. Their source attribution and qualifications are retained
in the root `NOTICE`. No blanket license for third-party content or provider
model weights is asserted here.

`Qwen2.5-3B-Instruct.txt` preserves that model's Qwen Research License at the
requested E41 revision; do not describe it as Apache-2.0. `GSM8K.txt` and
`MATH.txt` preserve the benchmark notices for historical material in the wider
repository. These license copies remain under their respective source terms.

## Assistance and Review

Substantial AI assistance contributed to implementation, experiments,
documentation, and archive preparation. The human author remains responsible
for the work. Automated testing, symbolic checks, AI review, and human review
are different kinds of evidence; none alone establishes novelty or peer review.
