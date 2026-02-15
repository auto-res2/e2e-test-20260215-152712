import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64, np.float16, np.int32, np.int64)):
        return obj.item()
    return obj


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    files = []
    if "test_accuracy" not in history.columns:
        return files
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.index, history["test_accuracy"], label="test_accuracy")
    if len(history["test_accuracy"]) > 0:
        ax.scatter([history.index[-1]], [history["test_accuracy"].iloc[-1]], color="red")
        ax.annotate(
            f"{history['test_accuracy'].iloc[-1]:.3f}",
            (history.index[-1], history["test_accuracy"].iloc[-1]),
        )
    ax.set_title("Test Accuracy Over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    fig.savefig(path)
    files.append(path)
    plt.close(fig)
    return files


def plot_probe_metrics(summary: Dict, run_id: str, out_dir: str) -> List[str]:
    keys = ["probe_acc0", "probe_acc1", "probe_delta_accuracy", "probe_harm_rate"]
    values = [summary.get(k) for k in keys if isinstance(summary.get(k), (int, float))]
    if not values:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=keys, y=[summary.get(k, 0.0) for k in keys], ax=ax)
    ax.set_title("Probe Metrics")
    ax.set_ylabel("Value")
    ax.set_xticklabels(keys, rotation=45, ha="right")
    for i, k in enumerate(keys):
        v = summary.get(k, 0.0)
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_probe_metrics_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_probe_confusion(summary: Dict, run_id: str, out_dir: str) -> List[str]:
    cc = summary.get("probe_cc")
    wc = summary.get("probe_wc")
    cw = summary.get("probe_cw")
    ww = summary.get("probe_ww")
    if not all(isinstance(v, (int, float)) for v in [cc, wc, cw, ww]):
        return []
    mat = np.array([[cc, cw], [wc, ww]])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Probe Confusion (Baseline→Demo)")
    ax.set_xlabel("Demo Correct")
    ax.set_ylabel("Baseline Correct")
    ax.set_xticklabels(["Correct", "Wrong"])
    ax.set_yticklabels(["Correct", "Wrong"], rotation=0)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_probe_confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_summary_bar(summary: Dict, run_id: str, out_dir: str) -> List[str]:
    keys = [k for k in summary.keys() if isinstance(summary[k], (int, float))]
    if not keys:
        return []
    values = [summary[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=keys, y=values, ax=ax)
    ax.set_title("Summary Metrics")
    ax.set_ylabel("Value")
    ax.set_xticklabels(keys, rotation=45, ha="right")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_summary_metrics_bar.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def comparison_bar_chart(metrics: Dict[str, Dict[str, float]], out_dir: str) -> List[str]:
    files = []
    for metric_name, values_dict in metrics.items():
        runs = list(values_dict.keys())
        values = [values_dict[r] for r in runs]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=runs, y=values, ax=ax)
        ax.set_title(f"{metric_name} Comparison")
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(runs, rotation=45, ha="right")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        path = os.path.join(out_dir, f"comparison_{metric_name}_bar_chart.pdf")
        fig.savefig(path)
        plt.close(fig)
        files.append(path)
    return files


def comparison_boxplot(metrics: Dict[str, Dict[str, float]], out_dir: str) -> List[str]:
    files = []
    for metric_name, values_dict in metrics.items():
        if len(values_dict) < 2:
            continue
        data = pd.DataFrame({"run_id": list(values_dict.keys()), "value": list(values_dict.values())})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="run_id", y="value", data=data, ax=ax)
        sns.stripplot(x="run_id", y="value", data=data, color="black", size=4, ax=ax)
        ax.set_title(f"{metric_name} Distribution")
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(data["run_id"].tolist(), rotation=45, ha="right")
        fig.tight_layout()
        path = os.path.join(out_dir, f"comparison_{metric_name}_boxplot.pdf")
        fig.savefig(path)
        plt.close(fig)
        files.append(path)
    return files


def save_metrics_table(metrics: Dict[str, Dict[str, float]], out_dir: str) -> List[str]:
    if not metrics:
        return []
    df = pd.DataFrame(metrics).fillna(0.0)
    csv_path = os.path.join(out_dir, "comparison_metrics_table.csv")
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.3 * len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_metrics_table.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    return [csv_path, fig_path]


def bootstrap_gap(
    proposed_vals: List[float],
    baseline_vals: List[float],
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    if len(proposed_vals) == 0 or len(baseline_vals) == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_bootstrap):
        p = np.random.choice(proposed_vals, size=len(proposed_vals), replace=True).mean()
        b = np.random.choice(baseline_vals, size=len(baseline_vals), replace=True).mean()
        diffs.append(p - b)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def permutation_test(
    proposed_vals: List[float],
    baseline_vals: List[float],
    n_perm: int = 1000,
) -> float:
    if len(proposed_vals) == 0 or len(baseline_vals) == 0:
        return 1.0
    combined = np.array(proposed_vals + baseline_vals)
    n_p = len(proposed_vals)
    observed = combined[:n_p].mean() - combined[n_p:].mean()
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        diff = combined[:n_p].mean() - combined[n_p:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def is_minimized_metric(metric_name: str) -> bool:
    lower = metric_name.lower()
    return any(token in lower for token in ["loss", "error", "perplexity"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    run_ids = json.loads(args.run_ids)

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    output_files: List[str] = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(samples=100000)
        summary = run.summary._json_dict
        config = dict(run.config)

        if config.get("mode") == "trial" or config.get("wandb", {}).get("mode") == "disabled":
            raise ValueError(f"Run {run_id} is in trial mode; evaluation requires full mode with WandB enabled.")

        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(
            metrics_path,
            {"history": history.to_dict(orient="list"), "summary": summary, "config": config},
        )
        output_files.append(metrics_path)

        output_files.extend(plot_learning_curve(history, run_id, run_dir))
        output_files.extend(plot_probe_metrics(summary, run_id, run_dir))
        output_files.extend(plot_probe_confusion(summary, run_id, run_dir))
        output_files.extend(plot_summary_bar(summary, run_id, run_dir))

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated_metrics.setdefault(k, {})[run_id] = float(v)

    comparison_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    primary_metric = "accuracy"
    primary_key = "test_accuracy" if "test_accuracy" in aggregated_metrics else primary_metric
    best_proposed = {"run_id": None, "value": -np.inf}
    best_baseline = {"run_id": None, "value": -np.inf}

    for run_id, val in aggregated_metrics.get(primary_key, {}).items():
        if "proposed" in run_id:
            if val > best_proposed["value"]:
                best_proposed = {"run_id": run_id, "value": val}
        if "baseline" in run_id or "comparative" in run_id:
            if val > best_baseline["value"]:
                best_baseline = {"run_id": run_id, "value": val}

    gap = None
    if best_proposed["run_id"] and best_baseline["run_id"] and best_baseline["value"] != 0:
        direction = -1 if is_minimized_metric(primary_metric) else 1
        raw_gap = (best_proposed["value"] - best_baseline["value"]) / best_baseline["value"] * 100.0
        gap = raw_gap * direction

    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    save_json(
        aggregated_path,
        {
            "primary_metric": primary_metric,
            "metrics": aggregated_metrics,
            "best_proposed": best_proposed,
            "best_baseline": best_baseline,
            "gap": gap,
        },
    )
    output_files.append(aggregated_path)

    output_files.extend(comparison_bar_chart(aggregated_metrics, comparison_dir))
    output_files.extend(comparison_boxplot(aggregated_metrics, comparison_dir))
    output_files.extend(save_metrics_table(aggregated_metrics, comparison_dir))

    proposed_vals = [v for k, v in aggregated_metrics.get(primary_key, {}).items() if "proposed" in k]
    baseline_vals = [
        v for k, v in aggregated_metrics.get(primary_key, {}).items() if "baseline" in k or "comparative" in k
    ]
    mean_diff, ci_low, ci_high = bootstrap_gap(proposed_vals, baseline_vals)
    p_value = permutation_test(proposed_vals, baseline_vals)
    stat_path = os.path.join(comparison_dir, "comparison_statistical_tests.json")
    save_json(
        stat_path,
        {
            "metric": primary_key,
            "mean_difference": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_bootstrap": 1000,
            "permutation_p_value": p_value,
        },
    )
    output_files.append(stat_path)

    for p in output_files:
        print(p)


if __name__ == "__main__":
    main()
