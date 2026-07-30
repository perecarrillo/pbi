import argparse
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import scipy.stats as st
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from gridsearch_classifiers import preload_dataset, _run_single

def calc_stats(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    # 95% CI
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        ci_margin = st.t.ppf(0.975, n - 1) * se
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin
    else:
        ci_lower = mean
        ci_upper = mean
    return mean, std, ci_lower, ci_upper

def _worker_job(args_tuple):
    seed, entry, base_dict, cache_path = args_tuple
    t0 = time.perf_counter()
    metrics = _run_single(entry, seed, base_dict, False, cache_path)
    return seed, metrics, time.perf_counter() - t0

def main():
    parser = argparse.ArgumentParser(description="Test config across multiple seeds with dataset caching and parallel execution.")
    parser.add_argument("-n", "--n-seeds", type=int, default=10, help="Number of seeds to test")
    parser.add_argument("-c", "--config", type=str, default="model_configs/best_model_pbip_datasets.yaml", help="Path to config file")
    parser.add_argument("-w", "--max-workers", type=int, default=4, help="Parallel worker processes. Default: 4")
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    with open(config_path, "r") as f:
        base_dict = yaml.safe_load(f)
        
    out_dir_str = base_dict.get('output_dir', 'output/tmp')
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded configuration from: {config_path}")
    print(f"Testing across {args.n_seeds} seeds using {args.max_workers} worker(s)...")
    print(f"Output directory: {out_dir}")

    dataset_cache_path = preload_dataset(base_dict, False, out_dir)
    
    use_kmer = bool(base_dict.get("kmer_features", {}) and base_dict["kmer_features"].get("k_values"))
    kmer_k_values = base_dict["kmer_features"]["k_values"] if use_kmer else None
    kmer_mlp_sizes = base_dict.get("classifier", {}).get("params", {}).get("kmer_mlp_sizes", [])
    entry = {
        "classifier": base_dict["classifier"],
        "training_overrides": {},
        "use_kmer": use_kmer,
        "kmer_k_values": kmer_k_values,
        "kmer_mlp_sizes": kmer_mlp_sizes,
    }

    metrics_collected = {
        'F1 Score': [],
        'Accuracy': [],
        'Recall': [],
        'Precision': [],
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': []
    }

    work_items = []
    for i in range(args.n_seeds):
        seed = 42 + i
        work_items.append((seed, entry, base_dict, str(dataset_cache_path)))

    completed_count = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_seed = {executor.submit(_worker_job, item): item[0] for item in work_items}
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            completed_count += 1
            try:
                seed_res, metrics, elapsed = future.result()
                f1 = metrics['f1']
                acc = metrics['accuracy']
                recall = metrics['recall']
                tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                metrics_collected['F1 Score'].append(f1)
                metrics_collected['Accuracy'].append(acc)
                metrics_collected['Recall'].append(recall)
                metrics_collected['Precision'].append(precision)
                metrics_collected['TP'].append(tp)
                metrics_collected['FP'].append(fp)
                metrics_collected['TN'].append(tn)
                metrics_collected['FN'].append(fn)
                
                print(f"[{completed_count}/{args.n_seeds}] Seed {seed} done in {elapsed:.1f}s | F1: {f1:.4f}, Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                
                # Write individual seed summary JSON to output folder
                seed_dir = out_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                with open(seed_dir / "results.json", "w") as f_json:
                    json.dump({"seed": seed, "test_metrics": metrics, "elapsed_s": elapsed}, f_json, indent=2)
            except Exception as exc:
                print(f"[{completed_count}/{args.n_seeds}] Seed {seed} generated an exception: {exc}")

    if not metrics_collected['F1 Score']:
        print("\nNo metrics were successfully collected.")
        return

    summary = []
    print("\n" + "="*80)
    print(f"Summary Statistics across {len(metrics_collected['F1 Score'])} seeds:")
    print("="*80)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std Dev':<10} | {'95% CI'}")
    print("-" * 80)
    
    for metric_name, values in metrics_collected.items():
        mean, std, ci_low, ci_high = calc_stats(values)
        summary.append({
            'Metric': metric_name,
            'Mean': mean,
            'Std Dev': std,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high
        })
        print(f"{metric_name:<15} | {mean:<10.4f} | {std:<10.4f} | [{ci_low:.4f}, {ci_high:.4f}]")

    csv_path = out_dir / "summary_statistics.csv"
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary statistics saved to {csv_path}")

if __name__ == "__main__":
    main()
