# Hybrid Dataset-Type Classification Using Ensemble Learning and LLMs

## 1. Project Overview

This project addresses the problem of **automatic dataset-type classification** using a hybrid intelligence approach. The system combines:

* Traditional machine learning (Random Forest)
* Fuzzy matching over column metadata
* Large Language Model (LLM)-assisted semantic reasoning

The objective is to accurately classify datasets (e.g., healthcare, finance, education, ecommerce, etc.) **purely from metadata**, without inspecting raw data values.

This work is positioned as an **applied data science framework** with strong emphasis on **reproducibility, transparency, and experimental validation**, aligning with modern open-science expectations.

---

## 2. Datasets Used

### 2.1 Dataset Description

The experiments use a **publicly shareable, metadata-only dataset** constructed from multiple open-access datasets. Each dataset instance is represented by:

* A consolidated text of column names
* A dataset-type label

**File:** `data/dataset_type_training_data.csv`

---

### 2.2 Dataset Schema

| Column Name  | Description                           |
| ------------ | ------------------------------------- |
| column_text  | Space-separated list of column names  |
| dataset_type | Ground-truth dataset category (label) |

---

### 2.3 Dataset Statistics

* Total dataset types: **15**
* Total metadata rows: *auto-derived from CSV*

Dataset statistics can be reproduced using:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/dataset_type_training_data.csv'); print('Classes:', df['dataset_type'].nunique()); print('Samples:', len(df))"
```

---

### 2.4 Data Source and License

* Metadata collected from **open-access datasets** (Kaggle, UCI ML Repository, OpenML)
* Only column names are used; no raw values are accessed
* No personal, private, or sensitive data involved
* Fully suitable for academic research and redistribution

---

## 3. Methodology

### 3.1 Hybrid Architecture

The proposed system follows a **three-stage hybrid decision pipeline**:

1. **Statistical Learning Layer**

   * Column metadata vectorized using a learned representation
   * Random Forest classifier produces probabilistic predictions

2. **Fuzzy Matching Layer**

   * Lexical similarity computed using Levenshtein-distance–based matching
   * Improves robustness for overlapping or sparse metadata

3. **LLM Reasoning Layer**

   * Semantic interpretation of column-name sets
   * Invoked only when confidence from earlier stages is insufficient

Final predictions are obtained through **confidence-aware aggregation**, ensuring efficiency while preserving accuracy.

---

## 4. Experiments and Evaluation

### 4.1 Baseline Models

* Random Forest (RF) only
* Fuzzy matching only
* LLM-only semantic classification

### 4.2 Proposed Hybrid Model

* Integrated ensemble of RF + Fuzzy + LLM reasoning

### 4.3 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

Evaluation outputs are stored in:

```
results/hybrid_model_test_results.csv
```

---

## 5. How to Run the Code

> **Validated Environment:** Python 3.10
> All dependencies are explicitly listed in `requirements.txt` for reproducibility.

### 5.1 Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 5.2 Run Experiments

```bash
python experiments/run_experiments.py
```

This command will:

* Load metadata-only dataset
* Train the hybrid model from scratch
* Evaluate all baselines and the proposed method
* Save metrics and predictions

---

### 5.3 Exporting API Key (Required)

⚠️ **Do NOT hard-code API keys.**

Set the Groq API key as an environment variable:

**Linux / macOS**

```bash
export GROQ_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

The application securely accesses the key using:

```python
os.getenv("GROQ_API_KEY")
```

---

## 6. Reproducibility Checklist

* Public, metadata-only datasets
* Deterministic preprocessing pipeline
* Fixed random seeds
* Explicit dependency versions
* Single-command experiment execution

This repository adheres to **open science and reproducible research best practices**.

---

## 7. Repository Structure

```
├── data/
│   └── dataset_type_training_data.csv
├── experiments/
│   └── run_experiments.py
├── results/
│   └── hybrid_model_test_results.csv
├── requirements.txt
├── README.md
```

---

## 8. Citation

If you use this work, please cite:

> Ramesh M., *Hybrid Dataset-Type Classification Using Ensemble Learning and Large Language Models*, 2025.

---

## 9. Contact

For questions or collaboration:

* **Author:** Ramesh M.
* **Email:** ramesh.m.j.2006@gmail
