import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix
import wandb

PRIMARY_METRIC_KEY = "test_exact_match"
PRIMARY_METRIC_READABLE = "Exact-match accuracy (%) on GSM8K test set"

###############################################################################
# Helper utilities                                                             #
###############################################################################

def _save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_wandb_cfg() -> Dict[str, str]:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r") as f:
        root = yaml.safe_load(f)
    wb_cfg = root["wandb"]
    return {"entity": wb_cfg["entity"], "project": wb_cfg["project"]}


def _learning_curve(history: pd.DataFrame, out_dir: str, run_id: str):
    plt.figure(figsize=(8, 5), dpi=300)
    has_data = False
    all_x_values = []

    for key in ["train/loss", "eval/loss"]:
        if key in history.columns:
            # Convert to numeric and drop NaN values
            data = pd.to_numeric(history[key], errors='coerce')
            valid_mask = ~data.isna()
            if valid_mask.sum() > 0:
                # Use _step column for x-axis, fall back to index if not available
                if '_step' in history.columns:
                    x_values = history.loc[valid_mask, '_step'].values
                else:
                    x_values = history.index[valid_mask].values
                y_values = data[valid_mask].values
                all_x_values.extend(x_values)

                # Use different markers for train vs eval
                marker = 'o' if 'train' in key else 's'
                plt.plot(x_values, y_values, label=key, linewidth=2, marker=marker,
                        markersize=6, markeredgewidth=1.5, markeredgecolor='white')
                has_data = True

    if has_data:
        plt.xlabel("Training Step", fontsize=12, fontweight='bold')
        plt.ylabel("Loss", fontsize=12, fontweight='bold')
        plt.title(f"Learning Curve: {run_id}", fontsize=14, fontweight='bold', pad=15)
        plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        # Fix x-axis limits for sparse data
        if len(all_x_values) > 0:
            min_x, max_x = min(all_x_values), max(all_x_values)
            if min_x == max_x:
                # Only one data point - show a reasonable range
                if '_step' in history.columns:
                    total_steps = history['_step'].max()
                    plt.xlim(0, max(10, total_steps * 1.1))
                else:
                    plt.xlim(0, 10)
            else:
                # Multiple points - add 5% padding
                range_x = max_x - min_x
                plt.xlim(max(0, min_x - 0.05 * range_x), max_x + 0.05 * range_x)

        plt.tight_layout()
        path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        print(path)
    else:
        plt.close()
        print(f"Warning: No valid data for learning curve {run_id}")


def _confusion(preds: List[Dict[str, str]], out_dir: str, run_id: str):
    if not preds:
        return
    # Handle case where preds might not be a list of dicts
    if not isinstance(preds, list):
        return
    if len(preds) == 0:
        return
    if not isinstance(preds[0], dict):
        return

    gold = [p["gold"] for p in preds if isinstance(p, dict) and "gold" in p]
    pred = [p["pred"] for p in preds if isinstance(p, dict) and "pred" in p]

    if not gold or not pred or len(gold) != len(pred):
        return

    labels = sorted(set(gold + pred))
    cm = confusion_matrix(gold, pred, labels=labels)

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix: {run_id}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(path)

###############################################################################
# Main evaluation script                                                       #
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs")
    args = parser.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    results_dir = args.results_dir

    wb_cfg = _load_wandb_cfg()
    api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = {}
    primary_vals: Dict[str, float] = {}

    for run_id in run_ids:
        run_out_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_out_dir, exist_ok=True)

        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{run_id}")
        history = run.history(keys=None, pandas=True)
        summary = run.summary._json_dict
        config = dict(run.config)

        # ---------------- Store metrics & figures -----------------------------
        _save_json({"summary": summary, "config": config}, os.path.join(run_out_dir, "metrics.json"))
        _learning_curve(history, run_out_dir, run_id)
        if "predictions" in summary:
            _confusion(summary["predictions"], run_out_dir, run_id)

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated.setdefault(k, {})[run_id] = v
            if k == PRIMARY_METRIC_KEY:
                primary_vals[run_id] = v

    # ---------------- Aggregated comparison ----------------------------------
    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    proposed = {k: v for k, v in primary_vals.items() if "proposed" in k}
    baseline = {k: v for k, v in primary_vals.items() if ("baseline" in k or "comparative" in k)}
    best_prop_id = max(proposed, key=proposed.get) if proposed else None
    best_base_id = max(baseline, key=baseline.get) if baseline else None

    gap = None
    if best_prop_id and best_base_id:
        baseline_val = baseline[best_base_id]
        if baseline_val != 0:
            gap = (proposed[best_prop_id] - baseline_val) / baseline_val * 100
        else:
            gap = None

    agg_json = {
        "primary_metric": PRIMARY_METRIC_READABLE,
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop_id, "value": proposed.get(best_prop_id) if proposed else None},
        "best_baseline": {"run_id": best_base_id, "value": baseline.get(best_base_id) if baseline else None},
        "gap": gap,
    }
    agg_path = os.path.join(comp_dir, "aggregated_metrics.json")
    _save_json(agg_json, agg_path)
    print(agg_path)

    # ---------------- Visual comparison chart --------------------------------
    if primary_vals:
        plt.figure(figsize=(12, 6), dpi=300)
        names, vals = zip(*primary_vals.items())

        # Create shortened labels for better readability
        short_labels = []
        for name in names:
            # Extract key parts: "proposed" or "comparative", and model name
            if "proposed" in name:
                label = "Proposed"
            elif "comparative" in name or "baseline" in name:
                label = "Baseline"
            else:
                label = name
            short_labels.append(label)

        # Create bar chart with better styling
        x_pos = np.arange(len(names))
        colors = ['#2E86AB' if 'proposed' in name else '#A23B72' for name in names]
        bars = plt.bar(x_pos, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)

        # Determine appropriate y-axis range
        max_val = max(vals) if vals else 0
        if max_val == 0:
            # All values are zero - set a reasonable scale for visualization
            y_max = 10
        else:
            # Use 15% padding above the max value
            y_max = max_val * 1.15

        # Add value labels on top of bars (or above baseline for zero values)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            height = bar.get_height()
            if height == 0:
                # For zero values, place text slightly above zero
                y_pos = y_max * 0.03
            else:
                # For non-zero values, place text above the bar
                y_pos = height + y_max * 0.02

            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel(PRIMARY_METRIC_READABLE, fontsize=14, fontweight='bold')
        plt.title("Model Performance Comparison", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x_pos, short_labels, fontsize=13)
        plt.yticks(fontsize=12)

        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)

        # Set y-axis limits
        plt.ylim(0, y_max)

        plt.tight_layout()
        bar_path = os.path.join(comp_dir, "comparison_accuracy_bar_chart.pdf")
        plt.savefig(bar_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(bar_path)

    # ---------------- Statistical significance -------------------------------
    if proposed and baseline:
        t_stat, p_val = stats.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        stats_path = os.path.join(comp_dir, "ttest.json")
        _save_json({"t_stat": float(t_stat), "p_value": float(p_val)}, stats_path)
        print(stats_path)

if __name__ == "__main__":
    main()
