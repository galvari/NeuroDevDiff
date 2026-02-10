# NDD Dataset Files

---

## Core datasets (CSV)

- **`neurodevdiff_v1_full.csv`**  
  Full dataset containing all generated cases.  
  Includes structured variables (demographics, symptoms, cognitive profiles, missing information) and generated clinical text.

- **`neurodevdiff_v1_train.csv`**  
- **`neurodevdiff_v1_val.csv`**  
- **`neurodevdiff_v1_test.csv`**  

  Fixed and stratified train/validation/test splits (70/15/15) based on the `true_profile` label, provided for reproducible experiments.

---

## LLM-oriented datasets (JSONL)

- **`neurodevdiff_v1_train.jsonl`**  
- **`neurodevdiff_v1_val.jsonl`**  
- **`neurodevdiff_v1_test.jsonl`**  

  Line-delimited JSON files, one case per line, designed for LLM training and evaluation.

  Each record includes:
  - a narrative clinical vignette (input)
  - a structured decision-support output:
    - `should_defer` (binary)
    - `should_defer_rationale`
    - suggested follow-up questions
    - a short differential hypothesis list
  - lightweight metadata (age, sex, severity, risk flag)

  These files are suitable for supervised fine-tuning, instruction tuning, or evaluation of decision-support behaviors.

---

## Metadata

- **`neurodevdiff_v1_metadata.json`**

  Dataset-level metadata, including:
  - dataset name and version
  - number of cases
  - random seed and noise level used during generation
  - class balance across neurodevelopmental profiles
  - overall defer rate and high-risk rate
  - creation timestamp (UTC)

  This file is intended to support transparency, reproducibility, and version comparison across future releases.

---

## Notes

- All data are **fully synthetic** and generated programmatically.  
- The dataset is **not intended for clinical use** or automated diagnosis.
- Labels represent latent generative profiles, not ground-truth diagnoses.
- The primary focus is on **uncertainty, missing information, and decision deferral**, rather than classification accuracy alone.

For details on dataset generation and design rationale, see the notebook in `../notebooks/00_generate_ndd_v1.ipynb`.