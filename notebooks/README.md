# Notebooks

This folder contains the core notebooks for the NeuroDevDiff project, covering the full pipeline from synthetic data generation to hybrid clinical decision support.

---

## Overview

The project focuses on modeling a defer / not-defer triage decision from textual clinical vignettes, and augmenting predictions with structured reasoning.

The workflow is organized into three main stages:

1. Dataset generation  
2. Supervised classification (DistilBERT)  
3. Hybrid system (classifier + LLM)  

---

## Notebooks

### 00 — Dataset Generation
`00_generate_ndd_v1.ipynb`

- Generates the NeuroDevDiff synthetic dataset
- Creates structured clinical vignettes with:
  - demographic info
  - symptom patterns
  - severity and functional impact
- Produces:
  - CSV splits (train / val / test)
  - JSONL format for LLM usage
  - metadata (class balance, defer rate, etc.)

This notebook defines the data generation logic and experimental setup.

---

### 01 — DistilBERT Classifier
`01_defer-or-not_tf-distilbert.ipynb`

- Implements a DistilBERT-based classifier using TensorFlow
- Task:
  - Binary classification → should_defer ∈ {0,1}

#### Training strategy:
- Stage A: frozen encoder (train classification head)
- Stage B: partial fine-tuning of last transformer layers

#### Outputs:
- Defer probability (confidence)
- Optimized threshold for classification

#### Artifacts saved:
- Trained model (SavedModel / Keras)
- Tokenizer
- Metadata (e.g., max_len, best threshold)

This notebook provides the core predictive model.

---

### 02 — Hybrid System (Classifier + LLM)
`02_hybrid_distilbert_llm.ipynb`

- Combines:
  - DistilBERT classifier → decision + confidence
  - LLM (Mistral-7B-Instruct) → structured reasoning

#### Pipeline:
1. Input vignette  
2. Classifier predicts:
   - should_defer
   - confidence  
3. LLM generates:
   - rationale
   - clarifying_questions
   - differential_hypotheses  

#### Output format:
{
  "should_defer": 1,
  "confidence": 0.78,
  "rationale": "...",
  "clarifying_questions": ["..."],
  "differential_hypotheses": ["..."]
}

This notebook demonstrates the final hybrid decision-support system.

---

## Results

### Classification performance

The DistilBERT classifier achieves strong performance on the defer decision task:

- Accuracy: 0.92  
- Precision: 1.00  
- Recall: 0.85  
- F1-score: 0.92  
- ROC-AUC: 0.82  

The model is intentionally high-precision, favoring conservative defer decisions and minimizing false positives.

---

### LLM Reasoning Layer

The hybrid system augments classifier predictions with structured clinical reasoning outputs, including a rationale, clarifying questions, and differential hypotheses.

#### Structural Evaluation

- JSON parse success rate: 94%  
- Non-empty rationale: 94%  
- Non-empty clarifying questions: 94%  
- Non-empty differential hypotheses: 94%  
- Average number of questions per case: 1.89  
- Average number of differential hypotheses: 2.69  
- Question diversity: 0.90  

These results indicate stable and consistent generation of structured reasoning components, with high diversity in clarifying questions and minimal failure in output formatting.

#### Qualitative Evaluation (LLM-as-Judge)

To assess semantic quality, outputs were evaluated using an auxiliary LLM acting as a judge, scoring multiple dimensions of reasoning quality (scale 1–5):

- Rationale specificity: 2.67  
- Question usefulness: 4.44  
- Differential plausibility: 2.89  
- Overall score: 3.11  

The evaluation highlights a clear strength in the generation of useful clarifying questions, suggesting the model effectively supports uncertainty reduction and information gathering.

However, rationale specificity and differential plausibility remain moderate, indicating that while outputs are generally coherent, they are sometimes generic or insufficiently tailored to individual cases.

Overall, the reasoning layer provides meaningful decision support, particularly in guiding follow-up questioning, while leaving room for improvement in case-specific justification and diagnostic refinement.

---

### Example output

{
  "should_defer": 1,
  "confidence": 0.98,
  "rationale": "Moderate severity and functional impact with diagnostic uncertainty.",
  "clarifying_questions": [
    "When did the symptoms first begin?",
    "Are there any comorbid features such as tics?"
  ],
  "differential_hypotheses": [
    "Obsessive-Compulsive Disorder",
    "Tic Disorder",
    "ADHD"
  ]
}

---

### Key insight

The system is designed to support clinical reasoning under uncertainty:

- detect decision uncertainty
- trigger defer decisions when appropriate
- generate targeted follow-up questions

Rather than forcing early diagnosis, the system models a workflow centered on:

"I don’t know yet — and that’s clinically meaningful."

---

## Execution Environments

The notebooks support 2 modes:

### Kaggle (recommended)
- GPU available  
- Easy dataset/model access  
- Minimal setup  

Paths typically follow:

/kaggle/input/datasets/<username>/...


To run the hybrid notebook, you also need the pretrained DistilBERT classifier artifacts, available on Kaggle in the published [distilbert-classifier](https://www.kaggle.com/datasets/galvari/distilbert-classifier)

---

### Local environment
- Requires:
  - dataset in data/
  - classifier in models/distilbert/

---

## Notes

- Large artifacts (models, datasets) are not stored in this repository  
- Refer to the main README for download instructions  

---

## Suggested usage

If you want to quickly understand the project:

1. Run 02_hybrid_distilbert_llm.ipynb  
2. Inspect example predictions  
3. Review evaluation results  

If you want full reproducibility:

1. Run 00_generate_ndd_v1.ipynb  
2. Train with 01_defer-or-not_tf-distilbert.ipynb  
3. Run the hybrid pipeline  

---

## Summary

These notebooks implement a complete pipeline for:

- Generating clinical-style datasets  
- Learning triage decisions from text  
- Augmenting predictions with structured LLM-based reasoning  
