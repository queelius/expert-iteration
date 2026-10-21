"""Build a batched, content-addressed Ollama embedding cache for Phase B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import httpx
import numpy as np

from verifier_anchored_rl.phase_b_features import CachedTextFeatures, node_feature_text


def load_texts(dataset: Path) -> list[str]:
    unique: dict[str, str] = {}
    with dataset.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = [row["state_text"]]
            candidates.extend(node_feature_text(node) for node in row["nodes"])
            for text in candidates:
                key = CachedTextFeatures.key(text)
                previous = unique.setdefault(key, text)
                if previous != text:
                    raise RuntimeError(f"SHA-256 collision for digest {key}")
    return [unique[key] for key in sorted(unique)]


def model_digest(client: httpx.Client, base_url: str, model: str) -> str:
    response = client.get(f"{base_url}/api/tags")
    response.raise_for_status()
    for record in response.json()["models"]:
        if record["name"] == model or record["model"] == model:
            return str(record["digest"])
    raise ValueError(f"model {model!r} not found in Ollama tags")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    texts = load_texts(args.dataset)
    keys = [CachedTextFeatures.key(text) for text in texts]
    batches: list[np.ndarray] = []
    with httpx.Client(timeout=args.timeout) as client:
        digest = model_digest(client, args.base_url, args.model)
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            response = client.post(
                f"{args.base_url}/api/embed",
                json={"model": args.model, "input": batch, "truncate": False},
            )
            response.raise_for_status()
            matrix = np.asarray(response.json()["embeddings"], dtype=np.float32)
            if matrix.shape[0] != len(batch):
                raise RuntimeError("Ollama returned the wrong embedding batch size")
            batches.append(matrix)
            print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}", flush=True)
    embeddings = np.concatenate(batches, axis=0)
    np.savez_compressed(
        args.out,
        model=np.asarray(args.model),
        model_digest=np.asarray(digest),
        keys=np.asarray(keys),
        embeddings=embeddings,
    )
    print(
        json.dumps(
            {
                "texts": len(texts),
                "dimension": int(embeddings.shape[1]),
                "model": args.model,
                "model_digest": digest,
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
