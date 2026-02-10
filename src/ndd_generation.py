from pathlib import Path
import json
import pandas as pd

def generate_and_save(
    output_dir: str | Path,
    config: Optional[NDDConfig] = None
) -> pd.DataFrame:
    cfg = config or NDDConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_neurodevdiff(cfg)

    csv_path = output_dir / f"neurodevdiff_v{cfg.version}_full.csv"
    jsonl_path = output_dir / f"neurodevdiff_v{cfg.version}_vignettes.jsonl"
    meta_path = output_dir / f"neurodevdiff_v{cfg.version}_metadata.json"

    # CSV
    df.to_csv(csv_path, index=False)

    # JSONL (testuale / LLM-friendly)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "case_id": int(row["case_id"]),
                "vignette": row["vignette_en"],
                "questions_to_ask": row["questions_to_ask_en"],
                "should_defer": int(row["should_defer"]),
                "should_defer_rationale": row["should_defer_rationale_en"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Metadata
    meta = {
        "dataset": "NeuroDevDiff",
        "version": cfg.version,
        "n_cases": len(df),
        "seed": cfg.seed,
        "noise_level": cfg.noise_level,
        "class_balance": df["true_profile"].value_counts(normalize=True).round(4).to_dict(),
        "defer_rate": float(df["should_defer"].mean()),
        "risk_high_rate": float(df["risk_high"].mean()),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--version", type=str, default="1.0")
    args = parser.parse_args()

    cfg = NDDConfig(
        n=args.n,
        seed=args.seed,
        noise_level=args.noise,
        version=args.version
    )

    generate_and_save(args.out, cfg)



### python src/ndd_generation.py --out data/ --n 3000 --version 1.1