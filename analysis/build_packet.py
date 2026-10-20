"""Build an allowlisted, deterministic local archive from a Git commit.

No upload or visibility change is performed. Working-tree contents are never
used as source bytes, even if the checkout is dirty.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"unsafe archive path: {name}")
    return path


def build_packet(root, revision, output):
    root, output = Path(root), Path(output)
    commit = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()

    def read(path):
        safe_path(path)
        return git(root, "show", f"{commit}:{path}")

    config = json.loads(read("analysis/packet.json"))
    if config["schema_version"] != "research-archive-packet-v1":
        raise ValueError("unsupported packet schema")
    paths = set(config["files"])
    tracked = git(root, "ls-tree", "-rz", commit).decode().split("\0")
    files = {}
    for row in filter(None, tracked):
        meta, name = row.split("\t", 1)
        mode, kind, _ = meta.split()
        files[name] = (mode, kind)
    for tree in config["trees"]:
        directory = safe_path(tree["path"])
        paths.update(name for name in files
                     if PurePosixPath(name).is_relative_to(directory)
                     and name.endswith(tree["suffix"]))
    evidence = json.loads(read(config["evidence_manifest"]))
    paths.update(entry["path"] for entry in evidence["inputs"].values())
    contents = {}
    for name in sorted(paths):
        safe_path(name)
        if files.get(name) not in {("100644", "blob"), ("100755", "blob")}:
            raise ValueError(f"missing or nonregular committed file: {name}")
        contents[name] = read(name)
    for entry in evidence["inputs"].values():
        if hashlib.sha256(contents[entry["path"]]).hexdigest() != entry["sha256"]:
            raise ValueError(f"evidence hash mismatch: {entry['path']}")
    manifest = {
        "schema_version": "research-archive-receipt-v1",
        "source_commit": commit,
        "scope": config["scope"],
        "sha256": {name: hashlib.sha256(value).hexdigest()
                   for name, value in contents.items()},
    }
    contents["ARCHIVE-MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a previously reviewed packet.
    with output.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as packet:
        for name, value in sorted(contents.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            packet.writestr(entry, value)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="exact reviewed source commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_packet(Path(__file__).resolve().parents[1], args.revision, args.output)
    print(f"Built {args.output} from {receipt['source_commit']} ({len(receipt['sha256'])} source files)")


if __name__ == "__main__":
    main()
