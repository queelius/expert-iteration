"""Offline archive checks must not need credentials or a model."""

import hashlib
import json
import socket
from pathlib import Path

import pytest

from analysis.reproduce import (
    EvidenceError,
    Observation,
    compare,
    load_evidence,
    reproduce,
    main,
    e20,
    revision_routing,
    locality,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("archive analysis attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def test_comparison_counts_pairs_and_groups_repetitions():
    rows = [
        Observation("a", "left", 0, 1), Observation("a", "right", 0, 0),
        Observation("a", "left", 1, 0), Observation("a", "right", 1, 1),
        Observation("b", "left", 0, 1), Observation("b", "right", 0, 1),
    ]
    result = compare(rows, "left", "right")
    assert result["unique_problems"] == 2
    assert result["paired_observations"] == 3
    assert result["left_correct"] == result["right_correct"] == 2
    assert result["left_only"] == result["right_only"] == result["both"] == 1
    assert result["neither"] == 0
    assert result["problem_effects"] == {"a": 0.0, "b": 0.0}


def test_comparison_rejects_duplicate_keys():
    with pytest.raises(EvidenceError, match="duplicate"):
        compare([Observation("a", "l", 0, 1)] * 2, "l", "r")


def test_comparison_rejects_unmatched_pairs():
    with pytest.raises(EvidenceError, match="paired"):
        compare([Observation("a", "l", 0, 1)], "l", "r")


@pytest.mark.parametrize("value", [None, "1", 2, -1, 0.5])
def test_invalid_rewards_are_not_coerced(value):
    with pytest.raises(EvidenceError):
        Observation("a", "l", 0, value)


def evidence_input(tmp_path, rows=None):
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (rows or [
        {"schema_version": "fixture-v1", "correct": True}
    ])) + "\n")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schemas": ["fixture-v1"],
        "disposition": "complete",
    }


def test_evidence_checks_missing_and_changed_files(tmp_path):
    entry = evidence_input(tmp_path)
    (tmp_path / entry["path"]).unlink()
    with pytest.raises(EvidenceError, match="missing"):
        load_evidence(tmp_path, entry)
    (tmp_path / entry["path"]).write_text("{}\n")
    with pytest.raises(EvidenceError, match="hash"):
        load_evidence(tmp_path, entry)


@pytest.mark.parametrize("disposition", ["partial", "quarantined", "unknown"])
def test_ineligible_evidence_cannot_enter_tables(tmp_path, disposition):
    entry = evidence_input(tmp_path)
    entry["disposition"] = disposition
    with pytest.raises(EvidenceError, match="complete"):
        load_evidence(tmp_path, entry)


def test_partial_filename_rejected_even_if_marked_complete(tmp_path):
    entry = evidence_input(tmp_path)
    (tmp_path / entry["path"]).rename(tmp_path / "run.jsonl.partial")
    entry["path"] = "run.jsonl.partial"
    with pytest.raises(EvidenceError, match="partial"):
        load_evidence(tmp_path, entry)


def test_unknown_schema_rejected(tmp_path):
    entry = evidence_input(tmp_path)
    entry["schemas"] = ["other-v1"]
    with pytest.raises(EvidenceError, match="schema"):
        load_evidence(tmp_path, entry)


def test_evidence_cannot_escape_packet(tmp_path):
    entry = evidence_input(tmp_path)
    entry["path"] = "../run.jsonl"
    with pytest.raises(EvidenceError, match="outside"):
        load_evidence(tmp_path, entry)


def test_checked_in_evidence_reproduces_all_selected_counts(tmp_path):
    manifest = Path(__file__).resolve().parents[1] / "analysis/claims.json"
    result = reproduce(manifest, tmp_path)
    tables = {r["id"]: r for r in result}
    assert (tables["e20"]["left_correct"], tables["e20"]["right_correct"]) == (8, 0)
    assert (tables["e41_revision"]["left_correct"], tables["e41_revision"]["right_correct"]) == (55, 49)
    assert (tables["e41_baseline"]["left_correct"], tables["e41_baseline"]["right_correct"]) == (55, 43)
    assert (tables["e42_routing"]["left_correct"], tables["e42_routing"]["right_correct"]) == (51, 46)
    assert (tables["e42_fixed"]["left_correct"], tables["e42_fixed"]["right_correct"]) == (55, 51)
    assert (tables["e47_locality"]["left_correct"], tables["e47_locality"]["right_correct"]) == (33, 24)
    assert tables["e47_locality"]["unique_problems"] == 12
    assert tables["e47_locality"]["paired_observations"] == 72
    assert (tables["e47_fixed"]["left_correct"], tables["e47_fixed"]["right_correct"]) == (10, 10)
    assert (tables["e47_adaptive"]["left_correct"], tables["e47_adaptive"]["right_correct"]) == (11, 10)
    first = (tmp_path / "tables.json").read_bytes()
    reproduce(manifest, tmp_path)
    assert (tmp_path / "tables.json").read_bytes() == first


def test_malformed_json_is_reported(tmp_path):
    entry = evidence_input(tmp_path)
    path = tmp_path / entry["path"]
    path.write_text("{\n")
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EvidenceError, match="invalid JSONL"):
        load_evidence(tmp_path, entry)


def test_cli_success_and_failure(tmp_path, monkeypatch, capsys):
    manifest = Path(__file__).resolve().parents[1] / "analysis/claims.json"
    monkeypatch.setattr("sys.argv", ["reproduce", "--manifest", str(manifest),
                                  "--output-dir", str(tmp_path)])
    main()
    assert "Reconstructed 8 comparisons" in capsys.readouterr().out
    other = tmp_path / "analysis"
    other.mkdir()
    bad = other / "claims.json"
    bad.write_text('{"schema_version":"unsupported"}')
    monkeypatch.setattr("sys.argv", ["reproduce", "--manifest", str(bad)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Evidence check failed" in capsys.readouterr().err


def test_excluded_claim_is_not_analyzed(tmp_path):
    directory = tmp_path / "analysis"
    directory.mkdir()
    manifest = directory / "claims.json"
    manifest.write_text(json.dumps({
        "schema_version": "research-archive-claims-v1",
        "inputs": {}, "reproduction_level": "test",
        "claims": [{"id": "excluded", "status": "excluded",
                    "exclusions": ["quarantined"]}]
    }))
    assert reproduce(manifest, tmp_path / "out") == []


def historical_data():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "analysis/claims.json").read_text())
    return {name: load_evidence(root, entry) for name, entry in manifest["inputs"].items()}


def test_e20_rejects_mismatched_problem_identity():
    data = historical_data()
    data["e20_direct"][0]["problem"] = "different problem"
    with pytest.raises(EvidenceError, match="pairing"):
        e20(data)


def test_revision_rejects_missing_arm_and_unknown_header():
    data = historical_data()
    rows = data["e42"]
    index = next(i for i, r in enumerate(rows) if r.get("record_type") == "routing_decision")
    del rows[index]
    with pytest.raises(EvidenceError, match="paired"):
        revision_routing(data)
    rows[0]["schema_version"] = "unknown"
    with pytest.raises(EvidenceError, match="header"):
        revision_routing(data)


@pytest.mark.parametrize("change", ["duplicate", "missing", "split", "reward"])
def test_locality_rejects_corrupt_cells(change):
    data = historical_data()
    rows = data["e47"]
    index = next(i for i, r in enumerate(rows) if r.get("record_type") == "lens_transition")
    if change == "duplicate":
        rows.insert(index, dict(rows[index]))
    elif change == "missing":
        del rows[index]
    elif change == "split":
        rows[index]["split"] = "evaluation"
    else:
        rows[index]["reward"] = "1"
    with pytest.raises(EvidenceError):
        locality(data)
