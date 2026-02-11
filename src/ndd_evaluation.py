from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COG_COLS = [
    "cog_verbal_language",
    "cog_visuospatial",
    "cog_working_memory",
    "cog_processing_speed",
    "cog_attention",
    "cog_motor",
]


@dataclass
class DatasetDiagnostics:
    n_cases: int
    class_balance: Dict[str, float]
    defer_rate: float
    risk_high_rate: float
    defer_by_profile: Dict[str, float]
    cognitive_means_by_profile: Dict[str, Dict[str, float]]


def compute_diagnostics(df: pd.DataFrame) -> DatasetDiagnostics:
    class_balance = df["true_profile"].value_counts(normalize=True).to_dict()
    defer_rate = float(df["should_defer"].mean())
    risk_high_rate = float(df["risk_high"].mean())
    defer_by_profile = df.groupby("true_profile")["should_defer"].mean().sort_values(ascending=False).to_dict()
    cog_means = df.groupby("true_profile")[COG_COLS].mean().round(2)
    cognitive_means_by_profile = cog_means.to_dict(orient="index")

    return DatasetDiagnostics(
        n_cases=int(len(df)),
        class_balance={k: float(v) for k, v in class_balance.items()},
        defer_rate=defer_rate,
        risk_high_rate=risk_high_rate,
        defer_by_profile={k: float(v) for k, v in defer_by_profile.items()},
        cognitive_means_by_profile={k: {kk: float(vv) for kk, vv in d.items()} for k, d in cognitive_means_by_profile.items()},
    )


def save_metadata_json(diag: DatasetDiagnostics, out_path: Path, dataset: str = "NeuroDevDiff", version: str = "1") -> None:
    meta = {
        "dataset": dataset,
        "version": version,
        "n_cases": diag.n_cases,
        "class_balance": {k: round(v, 4) for k, v in diag.class_balance.items()},
        "defer_rate": diag.defer_rate,
        "risk_high_rate": diag.risk_high_rate,
        "defer_rate_by_profile": {k: round(v, 4) for k, v in diag.defer_by_profile.items()},
        "cognitive_means_by_profile": diag.cognitive_means_by_profile,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["true_profile"].value_counts(normalize=True).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title("NeuroDevDiff – Class Distribution")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


def plot_cognitive_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    means = df.groupby("true_profile")[COG_COLS].mean()

    plt.figure(figsize=(9, 6))
    plt.imshow(means.values, aspect="auto")
    plt.colorbar(label="Mean scaled score")
    plt.xticks(range(len(COG_COLS)), COG_COLS, rotation=45, ha="right")
    plt.yticks(range(len(means.index)), means.index)
    plt.title("Mean Cognitive Profiles by Group")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


def plot_defer_by_profile(df: pd.DataFrame, out_path: Path) -> None:
    rates = df.groupby("true_profile")["should_defer"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(rates.index, rates.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Defer rate")
    plt.title("Defer Rate by Profile")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()