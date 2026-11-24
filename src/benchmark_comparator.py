"""
Benchmark Comparison and Lift Analysis Module
Author: Experimentation Engineer
Course: AIE231 Neural Networks Project - Alamein International University

This module implements:
1. Automated Lift Computation: Quantifies the delta improvement of any neural model over baselines (Section 3.3).
2. Benchmark Visualization: Generates publication-ready comparative bar charts across all evaluation metrics.
3. Summary Export: Outputs Markdown and CSV tables for the technical report.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.experiment_tracker import ExperimentTracker


class BenchmarkComparator:
    """
    Compares neural network experiments against trivial and learned baselines.
    Enforces Section 3.3 rule: 'Every later model is reported as an improvement over these numbers.'
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.tracker = ExperimentTracker(csv_path=csv_path)
        self.curves_dir = os.path.join(BASE_DIR, "experiments", "curves")
        os.makedirs(self.curves_dir, exist_ok=True)

    def load_benchmarks(self) -> pd.DataFrame:
        df = self.tracker.load_table()
        if df.empty:
            raise ValueError("No experiments found in experiments.csv.")
        return df

    def compute_lift_table(
        self, baseline_run_id: str = "BASE-LOGREG-001"
    ) -> pd.DataFrame:
        """
        Calculates absolute and percentage lift of all models relative to a baseline.
        """
        df = self.load_benchmarks()
        baseline_match = df[df["run_id"] == baseline_run_id]
        if baseline_match.empty:
            raise ValueError(f"Baseline run '{baseline_run_id}' not found in experiments table.")

        baseline_row = baseline_match.iloc[0]
        metrics = ["val_pr_auc", "val_roc_auc", "val_f1_score", "val_recall", "val_precision", "val_loss"]

        records = []
        for _, row in df.iterrows():
            rec = {
                "run_id": row["run_id"],
                "model_name": row["model_name"],
                "total_params": row["total_params"],
            }
            for m in metrics:
                val = float(row.get(m, 0.0))
                base_val = float(baseline_row.get(m, 0.0))
                rec[m] = round(val, 4)

                # Delta
                delta = val - base_val
                rec[f"{m}_delta"] = round(delta, 4)

                # Percentage lift
                if base_val != 0.0:
                    pct = (delta / abs(base_val)) * 100.0
                    rec[f"{m}_lift_pct"] = round(pct, 2)
                else:
                    rec[f"{m}_lift_pct"] = 0.0

            records.append(rec)

        lift_df = pd.DataFrame(records)
        return lift_df

    def plot_benchmark_bars(
        self,
        candidate_run_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates grouped bar chart comparing key metrics across Trivial,
        Logistic Regression, and candidate Neural Network models.
        """
        df = self.load_benchmarks()

        if candidate_run_ids:
            subset = df[df["run_id"].isin(candidate_run_ids)].copy()
        else:
            # Default: pick baselines and reference runs
            subset = df.drop_duplicates(subset=["model_name"]).copy()

        if subset.empty:
            raise ValueError("No matching runs to plot.")

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=300)

        metrics_to_plot = [
            ("val_pr_auc", "PR-AUC (Primary Metric)", "#1f77b4"),
            ("val_roc_auc", "ROC-AUC", "#2ca02c"),
            ("val_f1_score", "F1-Score (Positive Class)", "#ff7f0e"),
            ("val_recall", "Recall (Deposit Subscribers)", "#d62728"),
        ]

        models = list(subset["model_name"])

        for idx, (metric_col, title, color) in enumerate(metrics_to_plot):
            ax = axes[idx]
            vals = [float(v) for v in subset[metric_col]]
            bars = ax.bar(models, vals, color=color, alpha=0.85, edgecolor="black", width=0.55)

            # Annotate values on top of bars
            for bar, val in zip(bars, vals):
                y_pos = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos + 0.015,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        fig.suptitle("Week 10 Benchmark Comparison: Baselines vs Neural Network", fontsize=15, fontweight="bold", y=1.03)
        plt.tight_layout()

        target_path = save_path or os.path.join(self.curves_dir, "baseline_benchmark_comparison.png")
        plt.savefig(target_path, bbox_inches="tight", dpi=300)
        print(f"[PLOT] Benchmark comparison saved to: {target_path}")

        return fig

    def export_summary_markdown(self, output_path: Optional[str] = None) -> str:
        """Exports a clean Markdown table of benchmarks for inclusion in reports."""
        df = self.load_benchmarks()
        cols = [
            "run_id",
            "model_name",
            "total_params",
            "training_time_sec",
            "val_loss",
            "val_pr_auc",
            "val_roc_auc",
            "val_f1_score",
            "val_recall",
            "val_precision",
            "notes",
        ]
        available_cols = [c for c in cols if c in df.columns]
        summary_table = df[available_cols].to_markdown(index=False)

        md_content = f"""# Benchmark Summary Table (AIE231 Section 3.3)
Generated automatically by `BenchmarkComparator`.

{summary_table}

*Note: All validation metrics were evaluated on the leak-free 15% validation split without test-set leakage.*
"""
        target_path = output_path or os.path.join(BASE_DIR, "experiments", "benchmark_summary.md")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[SUMMARY] Benchmark summary exported to: {target_path}")
        return md_content


def run_benchmark_analysis():
    comparator = BenchmarkComparator()
    comparator.plot_benchmark_bars()
    comparator.export_summary_markdown()


if __name__ == "__main__":
    run_benchmark_analysis()
