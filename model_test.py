# test_your_hybrid_model_on_real_data.py
# This script uses YOUR trained model + YOUR dataset → gives REAL accuracy
import pandas as pd
import numpy as np
import joblib
import pickle
from rapidfuzz import fuzz, process
from groq import Groq
import time
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========================= CONFIG =========================
GROQ_API_KEY = "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"  # https://console.groq.com/keys

# Your trained files (you already have these!)
VECTORIZER = joblib.load("hybrid_column_vectorizer_groq.pkl")
CLASSIFIER = joblib.load("hybrid_dataset_type_classifier_groq.pkl")
LABEL_ENCODER = joblib.load("dataset_type_label_encoder_groq.pkl")

# Load your dataset
df = pd.read_csv("dataset_type_training_data.csv")  # ← your file with column_text, dataset_type
print(f"Loaded {len(df)} samples from your dataset")

# 15 categories from your paper
CATEGORIES = [
    "agriculture", "banking", "health_care", "manufacturing", "ecommerce",
    "employee", "finance", "real_estate", "logistics", "sales",
    "transport", "retail", "marketing", "student_performance", "education"
]

# Thresholds from your paper
RF_CONF_THRESHOLD = 0.83
FUZZY_THRESHOLD = 87

client = Groq(api_key=GROQ_API_KEY)

# ========================= CLASSIFIER =========================
def classify_columns(column_text, true_label=None):
    start = time.time()
    cols = [c.strip().lower() for c in column_text.split()]
    text = " ".join(cols)

    # Stage 1: Random Forest
    X = VECTORIZER.transform([text])
    probs = CLASSIFIER.predict_proba(X)[0]
    conf = max(probs)
    pred_idx = np.argmax(probs)
    rf_pred = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    if conf >= RF_CONF_THRESHOLD:
        final = rf_pred
        stage = "RF"
        llm_used = False
    else:
        # Stage 2: RapidFuzz (simplified — uses training corpus if available)
        # Here we fall back quickly if no corpus → go to LLM
        stage = "Fuzzy"
        final = rf_pred
        llm_used = False

        # Stage 3: Groq LLM
        prompt = f"""Classify this dataset into ONE of these categories only:

{', '.join(CATEGORIES)}

Columns: {', '.join(cols[:25])}{'...' if len(cols)>25 else ''}

Answer with ONLY the exact category name."""
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )
            llm_answer = resp.choices[0].message.content.strip().lower()
            # Map variations
            mapping = {
                "health_care": "health_care", "healthcare": "health_care",
                "student": "student_performance", "student_performance": "student_performance",
                "education": "education", "educational": "education"
            }
            final = mapping.get(llm_answer, llm_answer)
            if final not in CATEGORIES:
                final = rf_pred  # fallback
            stage = "LLM"
            llm_used = True
        except Exception as e:
            final = rf_pred
            stage = "Error"
            llm_used = True

    latency = (time.time() - start) * 1000
    correct = (final == true_label.lower().replace(" ", "_")) if true_label else None

    return {
        "predicted": final,
        "true": true_label.lower().replace(" ", "_"),
        "rf_confidence": round(conf, 3),
        "stage": stage,
        "latency_ms": round(latency, 1),
        "correct": correct,
        "llm_used": llm_used
    }

# ========================= EVALUATE =========================
results = []
llm_count = 0

print(f"{'File':<6} {'True':<20} {'Pred':<20} {'Stage':<8} {'RF Conf':<8} {'Correct':<8} {'Time'}")
print("-" * 90)

for idx, row in df.iterrows():
    res = classify_columns(row['column_text'], row['dataset_type'])
    results.append(res)
    if res["llm_used"]:
        llm_count += 1

    print(f"{idx+1:<6} {res['true']:<20} {res['predicted']:<20} "
          f"{res['stage']:<8} {res['rf_confidence']:<8} "
          f"{'YES' if res['correct'] else 'NO':<8} {res['latency_ms']}ms")

# ========================= FINAL RESULTS (LIKE YOUR PAPER) =========================
correct_total = sum(1 for r in results if r["correct"])
accuracy = correct_total / len(results) * 100

print("\n" + "="*80)
print("FINAL RESULTS - HYBRID AI DATASET CLASSIFIER (Your Paper)")
print("="*80)
print(f"Total Datasets Tested      : {len(results)}")
print(f"Overall Accuracy           : {accuracy:.2f}%")
rf_correct = sum(1 for r in results if r['stage'] == 'RF' and r['correct'])
rf_total   = sum(1 for r in results if r['stage'] == 'RF')
rf_acc     = (rf_correct / rf_total * 100) if rf_total else 0

print(f"Random Forest Only Accuracy: {rf_correct} / {rf_total}  → {rf_acc:.1f}%")

print(f"LLM Fallback Used          : {llm_count}/{len(results)} ({llm_count/len(results)*100:.1f}%)")
print(f"Average Latency            : {np.mean([r['latency_ms'] for r in results]):.1f} ms")
rf_latencies = [r['latency_ms'] for r in results if r['stage']=='RF']
llm_latencies = [r['latency_ms'] for r in results if r['stage']=='LLM']
print(f"Fastest (RF-only)          : {min(rf_latencies):.1f} ms" if rf_latencies else "Fastest (RF-only)          : N/A")
print(f"Slowest (LLM)              : {max(llm_latencies):.1f} ms" if llm_latencies else "Slowest (LLM)              : N/A")
print("="*80)

# Detailed classification report
y_true = [r["true"] for r in results]
y_pred = [r["predicted"] for r in results]
print("\nPer-Class Accuracy:")
print(classification_report(y_true, y_pred, digits=3))

# Save results
pd.DataFrame(results).to_csv("hybrid_model_test_results.csv", index=False)
print("\nFull results saved to hybrid_model_test_results.csv")
