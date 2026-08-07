import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import yaml

from pbi_utils.config_parser import Config, YAMLConfig, parse_config
from pipeline.orchestrator import run_full_pipeline

STATS_HEADER = (
    "Date\tCommand\tDescription\tBacteria embedder\tPhages embedder\t"
    "Classifier\tEpochs\tBS\tLR\tWD\tTraining Noise\tEnsemble Size\t"
    "Elapsed time (s)\t"
    "True Positive\tFalse Positive\tFalse Negative\tTrue Negative\t"
    "Accuracy\tWeighted Accuracy\tPrecision\tRecall\tF1 Score\tMCC\t"
    "Elapsed time (s)\t"
    "True Positive\tFalse Positive\tFalse Negative\tTrue Negative\t"
    "Accuracy\tWeighted Accuracy\tPrecision\tRecall\tSpecificity\tF1 Score\tMCC"
)

# Columns (0-indexed) that are numeric and should be aggregated as mean±std.
# Indices 12-34 cover all Train and Test metric columns.
# All other columns (0-11: date, command, model names, hyperparams) are taken verbatim from the last seed.
NUMERIC_COLS = set(range(12, 35))


def _parse_stats_row(stats_tsv_path: Path) -> list[str] | None:
    """Read the data row from a per-seed stats.tsv."""
    lines = stats_tsv_path.read_text(encoding="utf-8").splitlines()
    data_lines = [l for l in lines[1:] if l.strip()]
    if not data_lines:
        return None
    return data_lines[0].split("\t")


def _fmt(value: float) -> str:
    """Format a float number for drive."""
    return f"{value:.4f}".replace(".", ",")


def _aggregate_rows(rows: list[list[str]]) -> list[str]:
    """Produce a single summary row with mean±std for numeric columns."""
    n_cols = len(rows[0])
    last = rows[-1]
    summary = []
    for col_idx in range(n_cols):
        if col_idx in NUMERIC_COLS:
            # Parse values (comma decimal separator → float)
            values = []
            for row in rows:
                try:
                    values.append(float(row[col_idx].replace(",", ".")))
                except (ValueError, IndexError):
                    pass
            if values:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                summary.append(f"{_fmt(mean)}±{_fmt(std)}")
            else:
                summary.append("")
        else:
            summary.append(last[col_idx] if col_idx < len(last) else "")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test a config across multiple seeds and produce aggregated stats."
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("-n", "--n-seeds", type=int, default=10, help="Number of seeds to run (default: 10)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_dict = yaml.safe_load(f)

    top_out_dir = Path(base_dict.get("output_dir", "output/tmp"))
    top_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Seeds: {args.n_seeds}  |  Output: {top_out_dir}")

    # Save training_config.yaml
    shutil.copy(args.config, top_out_dir / "training_config.yaml")

    seed_rows: list[list[str]] = []

    for i in range(args.n_seeds):
        seed = 42 + i
        seed_out_dir = top_out_dir / "runs" / f"seed_{seed}"

        seed_dict = copy.deepcopy(base_dict)
        seed_dict["seed"] = seed
        seed_dict["output_dir"] = str(seed_out_dir)

        print(f"\n[Seed {seed}  ({i + 1}/{args.n_seeds})]  output → {seed_out_dir}")

        config = Config(YAMLConfig(**seed_dict), seed_dict)
        run_full_pipeline(config)

        # Collect the stats row written by run_full_pipeline
        seed_stats_path = seed_out_dir / "stats.tsv"
        if seed_stats_path.exists():
            row = _parse_stats_row(seed_stats_path)
            if row:
                seed_rows.append(row)
        else:
            print(f"  [WARNING] stats.tsv not found for seed {seed}, skipping.")

    if not seed_rows:
        print("\nNo stats rows collected.")
        return

    # Write stats.tsv
    summary_row = _aggregate_rows(seed_rows)
    stats_path = top_out_dir / "stats.tsv"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(STATS_HEADER)
        f.write("\n" + "\t".join(summary_row) + "\n")

    print(f"\nSummary stats saved to {stats_path}")
    print(f"Ran {len(seed_rows)} seeds successfully.")


if __name__ == "__main__":
    main()
