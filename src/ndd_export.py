from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import json
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    seed: int,
    stratify_col: str = "true_profile",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
    )

    # val/test are split 50/50 of temp_df if val_size == test_size
    # otherwise do a proportionate split
    val_frac_of_temp = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac_of_temp),
        random_state=seed,
        stratify=temp_df[stratify_col] if stratify_col in temp_df.columns else None,
    )

    return train_df, val_df, test_df


def _to_jsonl_llm_pairs(df_part: pd.DataFrame, path: Path) -> None:
    """
    JSONL format used for LLM training/eval:
      {"input": "...", "output": {...}, "meta": {...}}
    """
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df_part.iterrows():
            alt = [x.strip() for x in str(r.get("plausible_alternatives", "")).split(",") if x.strip()]

            record = {
                "input": r["vignette_en"],
                "output": {
                    "should_defer": int(r["should_defer"]),
                    "rationale": r["should_defer_rationale_en"],
                    "questions_to_ask": r["questions_to_ask_en"],
                    "differential_hypotheses": [r["true_profile"]] + alt[:2],
                },
                "meta": {
                    "case_id": int(r["case_id"]),
                    "true_profile": r["true_profile"],
                    "risk_high": int(r["risk_high"]),
                    "severity": r["severity"],
                    "age": int(r["age"]),
                    "sex": r["sex"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_jsonl_vignettes(df_full: pd.DataFrame, path: Path) -> None:
    """
    Full vignette view: one case per line (LLM-friendly, but not I/O-paired).
    """
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df_full.iterrows():
            rec = {
                "case_id": int(r["case_id"]),
                "vignette_en": r["vignette_en"],
                "questions_to_ask_en": r["questions_to_ask_en"],
                "should_defer": int(r["should_defer"]),
                "should_defer_rationale_en": r["should_defer_rationale_en"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_bundle(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    version: str = "1",
    seed: int,
    noise_level: float,
    cfg: Optional[Any] = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Writes:
      - neurodevdiff_v{version}_full.csv
      - neurodevdiff_v{version}_train.csv / _val.csv / _test.csv
      - neurodevdiff_v{version}_train.jsonl / _val.jsonl / _test.jsonl
      - neurodevdiff_v{version}_vignettes.jsonl  (full)
      - neurodevdiff_v{version}_metadata.json

    Returns dict of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # CSVs
    full_csv = out_dir / f"neurodevdiff_v{version}_full.csv"
    train_csv = out_dir / f"neurodevdiff_v{version}_train.csv"
    val_csv = out_dir / f"neurodevdiff_v{version}_val.csv"
    test_csv = out_dir / f"neurodevdiff_v{version}_test.csv"

    df.to_csv(full_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    paths["full_csv"] = full_csv
    paths["train_csv"] = train_csv
    paths["val_csv"] = val_csv
    paths["test_csv"] = test_csv

    # JSONL: full vignettes
    vignettes_jsonl = out_dir / f"neurodevdiff_v{version}_vignettes.jsonl"
    _to_jsonl_vignettes(df, vignettes_jsonl)
    paths["vignettes_jsonl"] = vignettes_jsonl

    # JSONL: paired IO for LLM workflows
    train_jsonl = out_dir / f"neurodevdiff_v{version}_train.jsonl"
    val_jsonl = out_dir / f"neurodevdiff_v{version}_val.jsonl"
    test_jsonl = out_dir / f"neurodevdiff_v{version}_test.jsonl"

    _to_jsonl_llm_pairs(train_df, train_jsonl)
    _to_jsonl_llm_pairs(val_df, val_jsonl)
    _to_jsonl_llm_pairs(test_df, test_jsonl)

    paths["train_jsonl"] = train_jsonl
    paths["val_jsonl"] = val_jsonl
    paths["test_jsonl"] = test_jsonl

    # Metadata
    meta_path = out_dir / f"neurodevdiff_v{version}_metadata.json"

    meta = {
        "dataset": "NeuroDevDiff",
        "version": str(version),
        "n_cases": int(len(df)),
        "seed": int(seed),
        "noise_level": float(noise_level),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_balance": df["true_profile"].value_counts(normalize=True).round(4).to_dict(),
        "defer_rate": float(df["should_defer"].mean()),
        "risk_high_rate": float(df["risk_high"].mean()),
    }

    # If you pass a config dataclass (or similar), store it too
    if cfg is not None:
        try:
            meta["config"] = asdict(cfg)  # dataclass
        except Exception:
            # fallback: store repr
            meta["config"] = repr(cfg)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    paths["metadata_json"] = meta_path

    if verbose:
        print(f"✅ Saved bundle to: {out_dir.resolve()}")
        for k, p in paths.items():
            print(f"  - {k}: {p.name}")

    return paths