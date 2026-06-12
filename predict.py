import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR = "models/deberta_lora"
BASE_MODEL = "distilbert-base-uncased"
SEED = 42

CRISIS_KEYWORDS = [
    "urgent", "critical", "outage", "down", "breach", "broken", "emergency",
    "failure", "not working", "data loss", "cannot access", "system failure",
    "escalate", "immediately", "asap", "severe", "blocked", "production down"
]
LOW_KEYWORDS = [
    "question", "inquiry", "how to", "tutorial", "info", "wondering",
    "when will", "update me", "minor", "small issue", "curious"
]
PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}
PRIORITY_REVERSE = {v: k for k, v in PRIORITY_MAP.items()}


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["ticket_description"] = df["ticket_description"].fillna("")
    df["ticket_subject"] = df["ticket_subject"].fillna("")
    df["ticket_channel"] = df["ticket_channel"].fillna("unknown")
    df["ticket_priority"] = df["ticket_priority"].str.strip().str.lower()
    df["full_text"] = df["ticket_subject"] + " " + df["ticket_description"]
    df["priority_numeric"] = df["ticket_priority"].map(PRIORITY_MAP)

    def parse_resolution_time(val):
        if pd.isna(val):
            return np.nan
        val = str(val).lower()
        numbers = re.findall(r"[\d.]+", val)
        if not numbers:
            return np.nan
        num = float(numbers[0])
        return num * 24 if "day" in val else num

    df["resolution_hours"] = df["resolution_time_hours"].apply(parse_resolution_time)
    df["resolution_hours"] = df["resolution_hours"].fillna(df["resolution_hours"].median())
    return df


# ─── SIGNAL COMPUTATION ───────────────────────────────────────────────────────
def compute_signals(df):
    # Rule-based
    def rule_score(text):
        text = text.lower()
        score = sum(1 for kw in CRISIS_KEYWORDS if kw in text)
        score -= sum(0.5 for kw in LOW_KEYWORDS if kw in text)
        score -= len(re.findall(r"\bnot\b|\bno\b|\bnever\b|\bwithout\b", text)) * 0.3
        return score

    df["rule_score"] = df["full_text"].apply(rule_score)
    df["rule_score_norm"] = df["rule_score"].rank(pct=True) * 3

    # Embedding clustering
    print("Encoding tickets with sentence transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["full_text"].tolist(), batch_size=64, show_progress_bar=True)
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)
    cluster_severity = df.groupby("cluster")["resolution_hours"].mean().sort_values()
    cluster_to_severity = {c: rank for rank, c in enumerate(cluster_severity.index)}
    df["cluster_severity"] = df["cluster"].map(cluster_to_severity)

    # Resolution time
    scaler = MinMaxScaler(feature_range=(0, 3))
    df["resolution_severity"] = scaler.fit_transform(df[["resolution_hours"]])

    # Fuse
    df["inferred_severity_score"] = (
        0.4 * df["rule_score_norm"] +
        0.4 * df["cluster_severity"] +
        0.2 * df["resolution_severity"]
    )

    def score_to_label(score):
        if score < 0.75:   return "low"
        elif score < 1.5:  return "medium"
        elif score < 2.25: return "high"
        else:              return "critical"

    df["inferred_severity"] = df["inferred_severity_score"].apply(score_to_label)
    df["inferred_severity_numeric"] = df["inferred_severity"].map(PRIORITY_MAP)
    df["severity_delta"] = df["inferred_severity_numeric"] - df["priority_numeric"]

    return df


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    return tokenizer, model


# ─── INFERENCE ────────────────────────────────────────────────────────────────
def run_inference(df, tokenizer, model):
    print("Running inference...")
    df["resolution_norm"] = MinMaxScaler().fit_transform(df[["resolution_hours"]])
    df["input_text"] = (
        df["full_text"] +
        " [CHANNEL] " + df["ticket_channel"] +
        " [RESTIME] " + df["resolution_norm"].round(2).astype(str)
    )

    predictions = []
    confidences = []

    for _, row in df.iterrows():
        inputs = tokenizer(
            row["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        predictions.append(pred)
        confidences.append(round(probs[pred].item(), 4))

    df["prediction"] = predictions        # 0=Consistent, 1=Mismatch
    df["model_confidence"] = confidences
    return df


# ─── DOSSIER GENERATION ───────────────────────────────────────────────────────
def generate_dossier(row, median_resolution):
    assigned = row["ticket_priority"]
    inferred = row["inferred_severity"]
    delta = int(row["severity_delta"])

    text = row["full_text"].lower()
    matched_keywords = [kw for kw in CRISIS_KEYWORDS if kw in text]
    keyword_weight = round(min(len(matched_keywords) / len(CRISIS_KEYWORDS), 1.0), 3)

    mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"

    dossier = {
        "ticket_id": str(row.get("ticket_id", row.name)),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mismatch_type,
        "severity_delta": str(delta),
        "feature_evidence": [
            {
                "signal": "keyword",
                "value": ", ".join(matched_keywords[:5]) if matched_keywords else "none",
                "weight": str(keyword_weight)
            },
            {
                "signal": "resolution_time_hours",
                "value": str(round(row["resolution_hours"], 1)) + " hours",
                "interpretation": (
                    "Above median — suggests elevated complexity"
                    if row["resolution_hours"] > median_resolution
                    else "Below median — routine resolution"
                )
            },
            {
                "signal": "embedding_cluster",
                "value": f"Cluster {row['cluster']}",
                "weight": str(round(row["cluster_severity"] / 3, 3))
            }
        ],
        "constraint_analysis": (
            f"The ticket was assigned '{assigned}' but semantic and keyword signals indicate "
            f"'{inferred}' severity. Resolution time of {round(row['resolution_hours'], 1)} hours "
            f"and embedding cluster membership further support this discrepancy, "
            f"yielding a severity delta of {delta}."
        ),
        "confidence": str(round(row["model_confidence"], 3))
    }
    return dossier


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main(input_csv, output_csv, dossier_output):
    # Load and process
    df_raw = pd.read_csv(input_csv)
    df = preprocess(df_raw)
    df = compute_signals(df)

    # Inference
    tokenizer, model = load_model()
    df = run_inference(df, tokenizer, model)

    # Save predictions CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    output_cols = [
        "ticket_priority", "inferred_severity", "severity_delta",
        "prediction", "model_confidence", "ticket_channel", "resolution_hours"
    ]
    df[output_cols].to_csv(output_csv, index=True, index_label="ticket_id")
    print(f"Predictions saved to {output_csv}")

    # Generate dossiers for predicted mismatches
    median_res = df["resolution_hours"].median()
    mismatched = df[df["prediction"] == 1]
    dossiers = [generate_dossier(row, median_res) for _, row in mismatched.iterrows()]

    os.makedirs(os.path.dirname(dossier_output) if os.path.dirname(dossier_output) else ".", exist_ok=True)
    with open(dossier_output, "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"Generated {len(dossiers)} dossiers → {dossier_output}")

    # Summary
    print(f"\nTotal tickets:     {len(df)}")
    print(f"Consistent:        {(df['prediction'] == 0).sum()}")
    print(f"Mismatch flagged:  {(df['prediction'] == 1).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=str, default="data/raw_tickets.csv",        help="Input CSV path")
    parser.add_argument("--output",  type=str, default="outputs/predictions.csv",     help="Output predictions CSV")
    parser.add_argument("--dossier", type=str, default="outputs/dossiers/dossiers.json", help="Output dossier JSON")
    args = parser.parse_args()

    main(args.input, args.output, args.dossier)