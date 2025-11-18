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
    plt.figure(figsize=(6, 4))
    for key in ["train/loss", "eval/loss"]:
        if key in history.columns:
            sns.lineplot(x=history.index, y=history[key], label=key)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Learning curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    plt.savefig(path)
    plt.close()
    print(path)


def _confusion(preds: List[Dict[str, str]], out_dir: str, run_id: str):
    if not preds:
        return
    gold = [p["gold"] for p in preds]
    pred = [p["pred"] for p in preds]
    labels = sorted(set(gold + pred))
    cm = confusion_matrix(gold, pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    plt.savefig(path)
    plt.close()
    print(path)

###############################################################################
# Main evaluation script                                                       #
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
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
        gap = (proposed[best_prop_id] - baseline[best_base_id]) / baseline[best_base_id] * 100

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
        plt.figure(figsize=(10, 4))
        names, vals = zip(*primary_vals.items())
        sns.barplot(x=list(names), y=list(vals))
        for i, v in enumerate(vals):
            plt.text(i, v + 0.3, f"{v:.2f}", ha="center")
        plt.ylabel(PRIMARY_METRIC_READABLE)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        bar_path = os.path.join(comp_dir, "comparison_accuracy_bar_chart.pdf")
        plt.savefig(bar_path)
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
