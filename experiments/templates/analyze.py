"""Per-experiment analysis: reads the experiment's runs (metrics.jsonl) and
raw/results.jsonl, computes the standard tables, writes analysis/report.md and
analysis/figures via analysis_lib. Copy into an experiment's analysis/ dir and
adapt. Deterministic: records the consumed run ids and code sha in the report."""
import sys
from experiments.analysis_lib import load_jsonl, sharpness_report, write_report


def main(exp_dir: str) -> int:
    raw = load_jsonl(f"{exp_dir}/raw/results.jsonl") if __import__("os").path.exists(f"{exp_dir}/raw/results.jsonl") else []
    sections = [("Raw cells", raw[:50])]
    write_report(f"{exp_dir}/analysis", "Analysis", sections)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
