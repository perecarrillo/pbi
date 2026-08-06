from __future__ import annotations
import argparse
import copy
import json
import math
import os
import shutil
import tempfile
import time
from itertools import combinations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from pbi_utils.config_parser import Config, YAMLConfig
from pbi_utils.logging import Logging
from pipeline.orchestrator import run_full_pipeline

logger = Logging()


def calculate_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def compute_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n > 1:
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci95 = float(t_crit * (std / math.sqrt(n)))
    else:
        ci95 = 0.0

    return {"mean": mean, "std": std, "ci95": ci95}


def get_phage_embedding_subsets(base_config: dict) -> List[Tuple[str, dict]]:
    base_models = base_config.get("phages_embedding_models", [])
    if not base_models:
        raise ValueError("base_config has no phages_embedding_models defined")

    subsets = []
    for r in range(1, len(base_models) + 1):
        for combo in combinations(base_models, r):
            names = [m["name"] for m in combo]
            label = "+".join(names)
            if r == len(base_models):
                label += " (baseline)"

            cfg = copy.deepcopy(base_config)
            cfg["phages_embedding_models"] = list(combo)
            subsets.append((label, cfg))
    return subsets


def get_bacteria_embedding_subsets(base_config: dict) -> List[Tuple[str, dict]]:
    base_models = base_config.get("bacteria_embedding_models", [])
    if not base_models:
        raise ValueError("base_config has no bacteria_embedding_models defined")

    subsets = []
    for r in range(1, len(base_models) + 1):
        for combo in combinations(base_models, r):
            names = [m["name"] for m in combo]
            label = "+".join(names)
            if r == len(base_models):
                label += " (baseline)"

            cfg = copy.deepcopy(base_config)
            cfg["bacteria_embedding_models"] = list(combo)
            subsets.append((label, cfg))
    return subsets


def get_kmer_variants(base_config: dict) -> List[Tuple[str, dict]]:
    variants = [
        ("No kmer", []),
        ("k=2", [2]),
        ("k=3", [3]),
        ("k=4 (baseline)", [4]),
        ("k=5", [5]),
        ("k=6", [6]),
        ("k=[3,4]", [3, 4]),
        ("k=[4,5]", [4, 5]),
        ("k=[3,4,5]", [3, 4, 5]),
        ("k=[4,5,6]", [4, 5, 6]),
        ("k=[2,3,4,5,6]", [2, 3, 4, 5, 6]),
    ]

    res = []
    for label, k_vals in variants:
        cfg = copy.deepcopy(base_config)
        cfg["kmer_features"] = {"k_values": k_vals}
        res.append((label, cfg))
    return res


def get_reducer_variants(base_config: dict) -> List[Tuple[str, dict]]:
    variants = [
        ("None", "none", "none"),
        ("PCA-50", "PCA", 50),
        ("PCA-100", "PCA", 100),
        ("PCA-150", "PCA", 150),
        ("PCA-200 (baseline)", "PCA", 200),
        ("PCA-300", "PCA", 300),
        ("PCA-500", "PCA", 500),
        ("UMAP-200", "UMAP", 200),
    ]

    res = []
    for label, reducer_type, n_comp in variants:
        cfg = copy.deepcopy(base_config)
        cfg["reducer_type"] = reducer_type
        cfg["pca_components"] = n_comp
        if "training_config" not in cfg:
            cfg["training_config"] = {}
        cfg["training_config"]["reduce_dimensionality"] = reducer_type
        if n_comp != "none":
            cfg["training_config"]["n_components_bacteria"] = int(n_comp)
            cfg["training_config"]["n_components_phages"] = int(n_comp)
        else:
            cfg["training_config"]["n_components_bacteria"] = None
            cfg["training_config"]["n_components_phages"] = None
        res.append((label, cfg))
    return res


def get_ensemble_variants(base_config: dict) -> List[Tuple[str, dict]]:
    variants = [
        ("1 (no ensemble)", 1),
        ("5", 5),
        ("10 (baseline)", 10),
        ("20", 20),
    ]

    res = []
    for label, ensemble_size in variants:
        cfg = copy.deepcopy(base_config)
        cfg["ensemble_size"] = ensemble_size
        res.append((label, cfg))
    return res


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

# Each entry: (label, strategy_dict)
# strategy_dict is a YAML-compatible dict for the ``strategy`` field of a
# model config.  For strategies with no params we use the short-form string.
_STRATEGY_VARIANTS: List[Tuple[str, Any]] = [
    # TKPert
    ("TKPert-concat-J16",
     {"name": "TKPertStrategy", "params": {"J": 16, "gamma": 20, "merging_strategy": "concat"}}),
    # Top-K (Consensus / Centroid - Concatenated)
    ("TopK-concat-K5",
     {"name": "TopKStrategy", "params": {"K": 5, "merging_strategy": "concat"}}),
    ("TopK-concat-K10",
     {"name": "TopKStrategy", "params": {"K": 10, "merging_strategy": "concat"}}),
    # Inverse Top-K (Divergent / Unique Chunks - Concatenated)
    ("InverseTopK-concat-K5",
     {"name": "InverseTopKStrategy", "params": {"K": 5, "merging_strategy": "concat"}}),
    ("InverseTopK-concat-K10",
     {"name": "InverseTopKStrategy", "params": {"K": 10, "merging_strategy": "concat"}}),
    # Other existing strategies
    ("Max",
     "MaxStrategy"),
    ("TopBottom-Truncate",
     "TopBottomTruncateStrategy"),
]


def _apply_strategy(model_dict: dict, strategy: Any) -> dict:
    """Return a copy of *model_dict* with the strategy field replaced."""
    m = copy.deepcopy(model_dict)
    m["strategy"] = strategy
    return m


def get_merging_strategy_variants(base_config: dict) -> List[Tuple[str, dict]]:
    """Generate merging strategy ablation variants.

    Produces variants starting with the baseline configuration, followed by
    changing 1 model at a time for each strategy:

    **Group 1 – Baseline** (1 variant):
        The base configuration as-is (e.g. all models using TKPertStrategy).

    **Group 2 – Phage branch only, one model at a time** (n_phage × 7 variants):
        For each phage model, sweep all alternative strategies.
        Bacteria models keep whatever strategy the base config specifies.

    **Group 3 – Bacteria branch only, one model at a time** (n_bact × 7 variants):
        Mirror of Group 2 for the bacteria branch.

    With 3 models per branch: 1 + 21 + 21 = **43 unique variants**.
    """
    phage_models = base_config.get("phages_embedding_models", [])
    bact_models = base_config.get("bacteria_embedding_models", [])
    n_phage = len(phage_models)
    n_bact = len(bact_models)

    res: List[Tuple[str, dict]] = []
    seen_labels: set = set()

    # -------------------------------------------------------------------
    # Group 1: Baseline configuration
    # -------------------------------------------------------------------
    res.append(("Baseline", copy.deepcopy(base_config)))
    seen_labels.add("Baseline")

    def _make_variant(
        label: str,
        strategy: Any,
        phage_model_idxs: List[int],
        bact_model_idxs: List[int],
    ) -> Tuple[str, dict]:
        """Build one variant config dict.

        Only models at *phage_model_idxs* / *bact_model_idxs* have their
        strategy replaced; all others keep the base config value.
        """
        cfg = copy.deepcopy(base_config)

        for idx in phage_model_idxs:
            if idx < len(cfg.get("phages_embedding_models", [])):
                cfg["phages_embedding_models"][idx] = _apply_strategy(
                    cfg["phages_embedding_models"][idx], strategy
                )
        for idx in bact_model_idxs:
            if idx < len(cfg.get("bacteria_embedding_models", [])):
                cfg["bacteria_embedding_models"][idx] = _apply_strategy(
                    cfg["bacteria_embedding_models"][idx], strategy
                )

        return label, cfg

    # -------------------------------------------------------------------
    # Group 2: phage branch only, one model at a time
    # -------------------------------------------------------------------
    for model_idx in range(n_phage):
        model_name = phage_models[model_idx].get("name", f"phage_model{model_idx}")
        for strat_label, strategy in _STRATEGY_VARIANTS:
            full_label = f"Phage/{model_name}:{strat_label}"
            if full_label in seen_labels:
                continue
            seen_labels.add(full_label)
            res.append(
                _make_variant(
                    full_label,
                    strategy,
                    phage_model_idxs=[model_idx],
                    bact_model_idxs=[],        # bacteria: keep base config
                )
            )

    # -------------------------------------------------------------------
    # Group 3: bacteria branch only, one model at a time
    # -------------------------------------------------------------------
    for model_idx in range(n_bact):
        model_name = bact_models[model_idx].get("name", f"bact_model{model_idx}")
        for strat_label, strategy in _STRATEGY_VARIANTS:
            full_label = f"Bact/{model_name}:{strat_label}"
            if full_label in seen_labels:
                continue
            seen_labels.add(full_label)
            res.append(
                _make_variant(
                    full_label,
                    strategy,
                    phage_model_idxs=[],        # phage: keep base config
                    bact_model_idxs=[model_idx],
                )
            )

    return res



EXPERIMENT_MAP = {
    "embeddings_phage": get_phage_embedding_subsets,
    "embeddings_bacteria": get_bacteria_embedding_subsets,
    "kmer": get_kmer_variants,
    "reducer": get_reducer_variants,
    "ensemble": get_ensemble_variants,
    "merging_strategy": get_merging_strategy_variants,
}


def run_single_rep(cfg_dict: dict, seed: int) -> Tuple[Dict[str, float], float]:
    temp_dir = tempfile.mkdtemp(prefix="ablation_run_")
    try:
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg["output_dir"] = temp_dir
        run_cfg["seed"] = seed

        yaml_cfg = YAMLConfig(**run_cfg)
        config_obj = Config(yaml_cfg, run_cfg)

        t0 = time.perf_counter()
        run_full_pipeline(config_obj)
        elapsed = time.perf_counter() - t0

        results_path = os.path.join(temp_dir, "results.json")
        with open(results_path, "r") as f:
            results_data = json.load(f)

        test_metrics = results_data.get("test_metrics")
        if not test_metrics or "confusion_matrix" not in test_metrics:
            raise RuntimeError(f"Missing test metrics in run results: {results_path}")

        cm = test_metrics["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        metrics = calculate_metrics(tn, fp, fn, tp)

        return metrics, elapsed
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="PBI Pipeline Ablation Study")
    parser.add_argument(
        "--base_config",
        type=str,
        default="model_configs/best_model_pbip_datasets.yaml",
        help="Path to base configuration YAML file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENT_MAP.keys()),
        help="Experiment to execute",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=10,
        help="Number of repetitions per variant (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_output",
        help="Directory to save ablation results (CSV and JSON)",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for repetitions (rep i uses seed + i)",
    )
    parser.add_argument(
        "--start_variant",
        type=int,
        default=1,
        help="1-based index of the first variant to run (use to resume after a crash).",
    )

    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base_config_dict = yaml.safe_load(f)

    variant_generator = EXPERIMENT_MAP[args.experiment]
    variants = variant_generator(base_config_dict)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"STARTING ABLATION EXPERIMENT: {args.experiment.upper()}")
    print(f"Base Config: {args.base_config}")
    print(f"Total Variants: {len(variants)}")
    if args.start_variant > 1:
        print(f"Resuming from variant {args.start_variant}/{len(variants)}")
    print(f"Repetitions per variant: {args.n_reps}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*80}\n")

    csv_path = os.path.join(args.output_dir, f"{args.experiment}_results.csv")
    json_path = os.path.join(args.output_dir, f"{args.experiment}_raw.json")

    # ------------------------------------------------------------------
    # Resume support: load any already-completed variants from disk so
    # we can skip them and still include them in the final summary.
    # ------------------------------------------------------------------
    completed_labels: set = set()
    summary_rows: List[dict] = []
    raw_results: Dict[str, Any] = {}

    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            for _, r in df_existing.iterrows():
                completed_labels.add(r["Variant"])
                summary_rows.append(r.to_dict())
            print(f"[Resume] Found {len(completed_labels)} already-completed variant(s) in {csv_path}")
        except Exception as exc:
            print(f"[Resume] Could not read existing CSV ({exc}); starting fresh.")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                raw_results = json.load(f)
        except Exception as exc:
            print(f"[Resume] Could not read existing JSON ({exc}); raw results will start fresh.")

    # Variants to actually run (skip already done, then apply --start_variant offset)
    pending_variants = [
        (label, cfg) for label, cfg in variants
        if label not in completed_labels
    ]
    pending_variants = pending_variants[args.start_variant - 1:]

    skipped = len(variants) - len(pending_variants) - (args.start_variant - 1)
    if skipped > 0 or args.start_variant > 1:
        remaining = len(pending_variants)
        print(f"[Resume] Skipping {len(variants) - remaining} variant(s); {remaining} remaining.\n")

    metrics_keys = ["accuracy", "precision", "recall", "specificity", "f1", "mcc"]

    for var_idx, (label, cfg_dict) in enumerate(pending_variants, 1):
        global_idx = len(completed_labels) + (args.start_variant - 1) + var_idx
        print(f"--- Variant [{global_idx}/{len(variants)}]: {label} ---")
        rep_metrics: Dict[str, List[float]] = {m: [] for m in metrics_keys}
        rep_times: List[float] = []

        raw_results[label] = []

        for rep in range(args.n_reps):
            seed = args.base_seed + rep
            print(f"  Repetition {rep+1}/{args.n_reps} (seed={seed})...", end="", flush=True)
            metrics, train_time = run_single_rep(cfg_dict, seed)
            print(f" Done ({train_time:.2f}s) | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")

            for m in metrics_keys:
                rep_metrics[m].append(metrics[m])
            rep_times.append(train_time)

            raw_results[label].append({
                "rep": rep,
                "seed": seed,
                "metrics": metrics,
                "train_time": train_time,
            })

        row = {"Variant": label, "Reps": args.n_reps}

        # Calculate stats for metrics
        for m in metrics_keys:
            st = compute_stats(rep_metrics[m])
            row[f"{m}_mean"] = st["mean"]
            row[f"{m}_std"] = st["std"]
            row[f"{m}_ci95"] = st["ci95"]
            row[f"{m}_str"] = f"{st['mean']:.4f} ± {st['std']:.4f} (±{st['ci95']:.4f})"

        time_st = compute_stats(rep_times)
        row["train_time_mean"] = time_st["mean"]
        row["train_time_std"] = time_st["std"]
        row["train_time_str"] = f"{time_st['mean']:.2f}s ± {time_st['std']:.2f}s"

        summary_rows.append(row)
        print()

        # ---------------------------------------------------------------
        # Incremental save: flush CSV and JSON after every variant so a
        # future crash can resume from where we left off.
        # ---------------------------------------------------------------
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(csv_path, index=False)

        with open(json_path, "w") as f:
            json.dump(raw_results, f, indent=2)

        print(f"  [Saved] {csv_path}")

    # Final summary print (includes previously-completed variants)
    df_summary = pd.DataFrame(summary_rows)

    print(f"{'='*80}")
    print(f"ABLATION EXPERIMENT {args.experiment.upper()} COMPLETE")
    print(f"Results CSV: {csv_path}")
    print(f"Raw JSON:    {json_path}")
    print(f"{'='*80}\n")

    # Format display table
    display_cols = ["Variant", "f1_str", "accuracy_str", "recall_str", "precision_str", "mcc_str", "train_time_str"]
    df_display = df_summary[display_cols].rename(
        columns={
            "f1_str": "F1 (Mean ± Std [CI95])",
            "accuracy_str": "Accuracy",
            "recall_str": "Recall / Sens",
            "precision_str": "Precision",
            "mcc_str": "MCC",
            "train_time_str": "Avg Train Time",
        }
    )

    print(df_display.to_string(index=False))


if __name__ == "__main__":
    main()
