# Hybrid Dataset-Type Classification Using Ensemble Learning and LLMs

## 1. Project Overview

This project addresses the problem of **automatic dataset-type classification** using a hybrid intelligence approach. The system combines:

* Traditional machine learning (Random Forest)
* Fuzzy matching over column metadata
* Large Language Model (LLM)-assisted semantic reasoning

The objective is to accurately classify datasets (e.g., healthcare, finance, education, ecommerce, etc.) **purely from metadata**, without inspecting raw data values.

This work is positioned as an **applied data science framework** with emphasis on reproducibility, transparency, and experimental validation.

---

## 2. Datasets Used

### 2.1 Dataset Description

The experiments use a **publicly shareable, metadata-only dataset** constructed from multiple open datasets. Each dataset is represented by:

* Column names
* Dataset type label

**File:** `dataset_type_training_data.csv`

### 2.2 Dataset Schema

| Column Name  | Description                           |
| ------------ | ------------------------------------- |
| column_name  | Name of a column in the dataset       |
| dataset_type | Ground-truth dataset category (label) |

### 2.3 Dataset Statistics

* Total dataset types: *N*
* Total datasets: *M*
* Total metadata rows: *K*

*(Exact numbers should be updated based on the CSV file)*

### 2.4 Data Source and License

* Data collected from **open-access datasets** (e.g., Kaggle, UCI ML Repository, OpenML)
* Only metadata is used
* No personal or sensitive data included
* Suitable for academic research and reproducibility

---

## 3. Methodology

### 3.1 Hybrid Architecture

The system consists of three layers:

1. **Statistical Layer**

   * Column names are vectorized
   * Random Forest classifier trained on metadata

2. **Fuzzy Matching Layer**

   * Levenshtein-based similarity matching
   * Captures lexical overlaps between domains

3. **LLM Reasoning Layer**

   * Semantic interpretation of column sets
   * Resolves ambiguous or overlapping cases

The final prediction is obtained through **confidence-weighted aggregation**.

---

## 4. Experiments and Evaluation

### 4.1 Baseline Models

* Random Forest only
* Fuzzy matching only
* LLM-only reasoning

### 4.2 Proposed Hybrid Model

* Ensemble of all three components

### 4.3 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

Results are saved in:

```
hybrid_model_test_results.csv
```

---

## 5. How to Run the Code

### 5.1 Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Run Experiments

```bash
python run_experiments.py
```

This will:

* Load the dataset
* Train the model from scratch
* Evaluate performance
* Save results

### 5.3 Exporting API key
```
export GROQ_API_KEY=your_key_here
---

## 6. Reproducibility Checklist

* Fixed random seeds
* Public dataset metadata
* Deterministic preprocessing
* Single-command execution
* All parameters explicitly defined

This repository follows **open science and reproducible research principles**.

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

* Author: Ramesh M.
* Email: [ramesh.m.j.2006@gmail.com](mailto:ramesh.m.j.2006@gmail.com)
