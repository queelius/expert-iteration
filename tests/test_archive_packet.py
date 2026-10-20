import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from analysis.build_packet import build_packet, safe_path, main


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


@pytest.fixture
def packet_repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    (root / "analysis").mkdir()
    (root / "README.md").write_text("committed\n")
    (root / "evidence.jsonl").write_text('{"correct":true}\n')
    claims = {"inputs": {"test": {
        "path": "evidence.jsonl",
        "sha256": hashlib.sha256((root / "evidence.jsonl").read_bytes()).hexdigest()
    }}}
    (root / "analysis/claims.json").write_text(json.dumps(claims))
    config = {"schema_version": "research-archive-packet-v1", "files": ["README.md", "analysis/claims.json"],
              "trees": [], "evidence_manifest": "analysis/claims.json", "scope": "fixture"}
    (root / "analysis/packet.json").write_text(json.dumps(config))
    git(root, "add", "README.md", "evidence.jsonl", "analysis/claims.json", "analysis/packet.json")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    return root, revision


def test_packet_is_pinned_to_commit_and_deterministic(tmp_path, packet_repo):
    root, revision = packet_repo
    (root / "README.md").write_text("dirty working tree\n")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    build_packet(root, revision, a)
    build_packet(root, revision, b)
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as packet:
        assert packet.read("README.md") == b"committed\n"
        receipt = json.loads(packet.read("ARCHIVE-MANIFEST.json"))
        assert receipt["source_commit"] == revision
        for name, digest in receipt["sha256"].items():
            assert hashlib.sha256(packet.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_packet(root, revision, a)


def test_invalid_revision_fails_without_output(tmp_path):
    output = tmp_path / "out.zip"
    with pytest.raises(subprocess.CalledProcessError):
        build_packet(Path(__file__).resolve().parents[1], "nonexistent-archive-revision", output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../secret", "a//b"])
def test_packet_paths_must_be_normal_relative_paths(name):
    with pytest.raises(ValueError, match="unsafe"):
        safe_path(name)


@pytest.mark.parametrize("defect", ["schema", "missing", "symlink", "hash"])
def test_packet_rejects_invalid_committed_inputs(tmp_path, packet_repo, defect):
    root, _ = packet_repo
    config_path = root / "analysis/packet.json"
    config = json.loads(config_path.read_text())
    if defect == "schema":
        config["schema_version"] = "unsupported"
    elif defect == "missing":
        config["files"].append("missing.txt")
    elif defect == "symlink":
        (root / "link").symlink_to("README.md")
        config["files"].append("link")
        git(root, "add", "link")
    else:
        (root / "evidence.jsonl").write_text('{"correct":false}\n')
        git(root, "add", "evidence.jsonl")
    config_path.write_text(json.dumps(config))
    git(root, "add", "analysis/packet.json")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "invalid fixture")
    output = tmp_path / "invalid.zip"
    with pytest.raises(ValueError):
        build_packet(root, "HEAD", output)
    assert not output.exists()


def test_packet_tree_selection_and_cli(tmp_path, packet_repo, monkeypatch, capsys):
    root, _ = packet_repo
    (root / "src").mkdir()
    (root / "src/selected.py").write_text("pass\n")
    (root / "src/omitted.txt").write_text("not in packet\n")
    config_path = root / "analysis/packet.json"
    config = json.loads(config_path.read_text())
    config["trees"] = [{"path": "src", "suffix": ".py"}]
    config_path.write_text(json.dumps(config))
    git(root, "add", "src/selected.py", "src/omitted.txt", "analysis/packet.json")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "tree fixture")
    monkeypatch.setattr("analysis.build_packet.__file__", str(root / "analysis/build_packet.py"))
    output = tmp_path / "tree.zip"
    monkeypatch.setattr("sys.argv", ["build_packet", "--revision", "HEAD", "--output", str(output)])
    main()
    assert "Built" in capsys.readouterr().out
    with zipfile.ZipFile(output) as packet:
        assert "src/selected.py" in packet.namelist()
        assert "src/omitted.txt" not in packet.namelist()
