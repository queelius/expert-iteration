"""Recount selected historical outcomes, without models or scientific rescoring.

The input adapters are deliberately restricted to the archived contracts.
Checksums bind exact files; record-level counts do not establish that the
historical verifier, experiment design, or model outputs were correct.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, or outside the supported contract."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


@dataclass(frozen=True)
class Observation:
    problem: str
    arm: str
    repetition: object
    correct: int

    def __post_init__(self):
        require(type(self.correct) in (bool, int) and self.correct in (0, 1),
                "correctness must be a binary outcome")


def compare(rows, left, right):
    selected = {}
    for row in rows:
        if row.arm not in (left, right):
            continue
        key = (row.problem, row.repetition, row.arm)
        require(key not in selected, f"duplicate observation: {key}")
        selected[key] = int(row.correct)
    keys = {(problem, rep) for problem, rep, _ in selected}
    require(bool(keys), "no paired observations")
    cells = dict(both=0, left_only=0, right_only=0, neither=0)
    effects = defaultdict(list)
    for problem, rep in sorted(keys):
        require((problem, rep, left) in selected and
                (problem, rep, right) in selected, "unmatched paired observation")
        a, b = selected[problem, rep, left], selected[problem, rep, right]
        cell = "both" if a and b else "left_only" if a else "right_only" if b else "neither"
        cells[cell] += 1
        effects[problem].append(a - b)
    return {
        "left": left, "right": right, "unique_problems": len(effects),
        "paired_observations": len(keys),
        "left_correct": cells["both"] + cells["left_only"],
        "right_correct": cells["both"] + cells["right_only"],
        **cells,
        "problem_effects": {p: sum(v) / len(v) for p, v in sorted(effects.items())},
    }


def load_evidence(root, entry):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    require(path.is_relative_to(root), "input path outside packet")
    require(not path.name.endswith(".partial"), "partial input is not eligible")
    require(entry["disposition"] == "complete", "input must be complete")
    require(path.is_file(), f"missing evidence: {entry['path']}")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == entry["sha256"],
            f"hash mismatch: {entry['path']}")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise EvidenceError(f"invalid JSONL: {entry['path']}") from exc
    require(bool(rows), "empty evidence")
    require(all(isinstance(r, dict) for r in rows), "records must be objects")
    require({r.get("schema_version", "unversioned") for r in rows}
            == set(entry["schemas"]), f"unexpected schema: {entry['path']}")
    return rows


def unique(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        require(key not in result, f"duplicate {field}: {key}")
        result[key] = row
    return result


def records(rows, kind):
    return [r for r in rows if r.get("record_type") == kind]


def header(rows, schema):
    require(rows[0].get("schema_version") == schema, "unsupported header contract")
    require(len(records(rows, "run_header")) == 1, "duplicate run header")
    require(rows[-1].get("record_type") == "run_summary", "missing final run summary")
    require(len(records(rows, "run_summary")) == 1, "duplicate run summary")
    return rows[0]


def result(identifier, rows, left, right):
    return {"id": identifier, **compare(rows, left, right)}


def e20(data):
    direct = unique(data["e20_direct"], "index")
    summaries = unique([r for r in data["e20_controller"] if r.get("summary") is True], "episode")
    turns = defaultdict(list)
    for row in data["e20_controller"]:
        if not row.get("summary"):
            turns[row["episode"]].append(row)
    require(set(direct) == set(summaries) == set(turns), "E20 episode coverage mismatch")
    rows = []
    for index, base in sorted(direct.items()):
        episode = turns[index]
        require([r["turn"] for r in episode] == list(range(len(episode))),
                "E20 duplicate or missing turns")
        require(all(r["problem"] == base["problem"] for r in episode),
                "E20 problem pairing mismatch")
        reward = sum(r["reward"] for r in episode)
        require(reward == summaries[index]["total_reward"], "E20 summary reward mismatch")
        require(len(episode) == summaries[index]["turns"], "E20 turn count mismatch")
        rows.extend([
            Observation(str(index), "direct", 0, base["correct"]),
            Observation(str(index), "imitation_controller", 0, int(reward > 0)),
        ])
    return [result("e20", rows, "direct", "imitation_controller")]


def revision_routing(data):
    h41 = header(data["e41"], "workspace-paired-revision-prompt-header-v1")
    h42 = header(data["e42"], "workspace-paired-routing-prompt-header-v1")
    sources = unique(records(data["e41"], "paired_revision_prompt_source"), "problem_id")
    ids = set(h41["source"]["problem_ids"])
    require(set(sources) == ids == set(h42["source"]["problem_ids"]), "E41/E42 problem pairing mismatch")
    observations = [
        Observation(p, "baseline", 0, r["baseline_symbolic_success"])
        for p, r in sources.items()
    ]
    for r in data["e41"]:
        if r.get("schema_version") != "workspace-paired-revision-prompt-transition-v1":
            continue
        require(r["problem_id"] in ids, "unknown E41 problem")
        require(r["baseline_symbolic_success"] == sources[r["problem_id"]]["baseline_symbolic_success"],
                "baseline outcome mismatch")
        observations.append(Observation(r["problem_id"], r["arm"], 0,
                                        r["revision_symbolic_success"]))
    for r in records(data["e42"], "routing_decision"):
        require(r["schema_version"] == "workspace-paired-routing-prompt-decision-v1", "unsupported routing contract")
        require(r["problem_id"] in ids, "unknown E42 problem")
        observations.append(Observation(r["problem_id"], r["arm"], 0, r["reward"]))
    retained = "workspace-revise-answer-container-v1"
    independent = "workspace-revise-independent-answer-container-v1"
    generic, explicit = "global-planner-v1", "revision-routing-v1"
    expected_arms = {"baseline", retained, independent, generic, explicit}
    require({o.arm for o in observations} == expected_arms, "unexpected revision/routing arm")
    tables = [
        result("e41_revision", observations, independent, retained),
        result("e41_baseline", observations, independent, "baseline"),
        result("e42_routing", observations, explicit, generic),
        result("e42_fixed", observations, independent, explicit),
    ]
    require(all(t["unique_problems"] == len(ids) for t in tables), "incomplete revision/routing pairs")
    return tables


def locality(data):
    rows = data["e47"]
    h = header(rows, "action-lens-factorial-header-v1")
    require(h["contracts"]["analysis"] == "action-lens-pilot-analysis-v2", "unsupported locality analysis")
    settings = h["settings"]
    ids = set(h["source"]["problem_ids"])
    sources = unique(records(rows, "source_state"), "problem_id")
    require(set(sources) == ids, "E47 source coverage mismatch")
    transitions = {}
    discovery = settings["discovery_repetitions"]
    evaluation = settings["evaluation_repetitions"]
    require(set(discovery).isdisjoint(evaluation), "discovery/evaluation overlap")
    require(set(discovery + evaluation) == set(range(settings["repetitions"])), "invalid repetition split")
    for r in records(rows, "lens_transition"):
        require(r["schema_version"] == "action-lens-transition-v1", "unsupported transition contract")
        key = r["problem_id"], r["arm"], r["repetition"]
        require(key not in transitions, "duplicate E47 observation")
        require(r["problem_id"] in ids, "unknown E47 problem")
        require(r["source_state_sha256"] == sources[r["problem_id"]]["state_sha256"], "source state mismatch")
        require(r["split"] == ("discovery" if r["repetition"] in discovery else "evaluation"), "split mismatch")
        transitions[key] = Observation(*key, r["reward"])
    expected = {(p, a, r) for p in ids for a in settings["arms"]
                for r in range(settings["repetitions"])}
    require(set(transitions) == expected, "E47 incomplete or unexpected cells")
    targets = settings["pilot_analysis"]["global_target_tie_order"]

    def score(problem_ids, target):
        return sum(transitions[p, f"local_{target}", r].correct
                   for p in problem_ids for r in discovery)

    fixed = max(targets, key=lambda t: score(ids, t))
    local_pairs, selected_pairs = [], []
    choices = {}
    for p in sorted(ids):
        best = max(score([p], t) for t in targets)
        winners = [t for t in targets if score([p], t) == best]
        chosen = winners[0] if len(winners) == 1 else fixed
        choices[p] = chosen
        for rep in evaluation:
            for target in targets:
                for view in ("local", "full"):
                    value = transitions[p, f"{view}_{target}", rep].correct
                    local_pairs.append(Observation(p, view, (rep, target), value))
            for label, arm in (("fixed_child", f"local_{fixed}"),
                               ("state_selected_child", f"local_{chosen}"),
                               ("direct", "problem_only")):
                selected_pairs.append(Observation(p, label, rep, transitions[p, arm, rep].correct))
    local = result("e47_locality", local_pairs, "local", "full")
    fixed_table = result("e47_fixed", selected_pairs, "fixed_child", "direct")
    adaptive = result("e47_adaptive", selected_pairs, "state_selected_child", "fixed_child")
    fixed_table["discovery_selected_target"] = fixed
    adaptive["discovery_choices"] = choices
    return [local, fixed_table, adaptive]


def reproduce(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema_version"] == "research-archive-claims-v1", "unsupported manifest schema")
    inputs = {key: load_evidence(root, entry) for key, entry in manifest["inputs"].items()}
    adapters = {"e20": e20, "revision_routing": revision_routing, "locality": locality}
    unique(manifest["claims"], "id")
    tables = []
    for claim in manifest["claims"]:
        require(claim["status"] in ("candidate", "verified", "excluded"), "invalid claim status")
        if claim["status"] == "excluded":
            require(bool(claim["exclusions"]), "excluded claim needs a reason")
            continue
        require(claim["analysis"] in adapters, "unsupported analysis contract")
        subset = {key: inputs[key] for key in claim["inputs"]}
        tables.extend(adapters[claim["analysis"]](subset))
    unique(tables, "id")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables.json").write_text(json.dumps(tables, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Reconstructed Historical Outcomes", "",
        manifest["reproduction_level"], "",
        "Counts are descriptive. Repeated observations are not independent problems.",
        "No confidence intervals or new significance tests are computed here.", "",
        "| Comparison | Left | Right | Problems | Paired observations | Left correct | Right correct |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for t in tables:
        lines.append(f"| {t['id']} | {t['left']} | {t['right']} | {t['unique_problems']} | "
                     f"{t['paired_observations']} | {t['left_correct']} | {t['right_correct']} |")
    (output / "tables.md").write_text("\n".join(lines) + "\n")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("claims.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    try:
        tables = reproduce(args.manifest, args.output_dir)
    except (EvidenceError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Evidence check failed: {exc}\n")
    print(f"Reconstructed {len(tables)} comparisons in {args.output_dir}")


if __name__ == "__main__":
    main()
