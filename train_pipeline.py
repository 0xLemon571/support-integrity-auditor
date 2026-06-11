import pandas as pd
import numpy as np
import re
import os
import json
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from torch.nn import CrossEntropyLoss
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
np.random.seed(SEED)
torch.manual_seed(SEED)

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


# ─── STAGE 1: PREPROCESSING ───────────────────────────────────────────────────
def load_and_preprocess(csv_path):
    print(f"[1/7] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["ticket_description"] = df["ticket_description"].fillna("")
    df["ticket_subject"] = df["ticket_subject"].fillna("")
    df["ticket_channel"] = df["ticket_channel"].fillna("unknown")
    df["ticket_type"] = df["ticket_type"].fillna("unknown")
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

    print(f"    Loaded {len(df)} tickets.")
    return df


# ─── STAGE 1: SIGNAL 1 — RULE-BASED NLP ──────────────────────────────────────
def compute_rule_score(df):
    print("[2/7] Computing rule-based NLP signal...")

    def rule_score(text):
        text = text.lower()
        score = sum(1 for kw in CRISIS_KEYWORDS if kw in text)
        score -= sum(0.5 for kw in LOW_KEYWORDS if kw in text)
        score -= len(re.findall(r"\bnot\b|\bno\b|\bnever\b|\bwithout\b", text)) * 0.3
        return score

    df["rule_score"] = df["full_text"].apply(rule_score)
    df["rule_score_norm"] = df["rule_score"].rank(pct=True) * 3
    return df


# ─── STAGE 1: SIGNAL 2 — EMBEDDING CLUSTERING ────────────────────────────────
def compute_embedding_signal(df):
    print("[3/7] Computing embedding cluster signal...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["full_text"].tolist(), batch_size=64, show_progress_bar=True)

    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    cluster_severity = df.groupby("cluster")["resolution_hours"].mean().sort_values()
    cluster_to_severity = {c: rank for rank, c in enumerate(cluster_severity.index)}
    df["cluster_severity"] = df["cluster"].map(cluster_to_severity)
    return df


# ─── STAGE 1: SIGNAL 3 — RESOLUTION TIME ─────────────────────────────────────
def compute_resolution_signal(df):
    print("[4/7] Computing resolution time signal...")
    scaler = MinMaxScaler(feature_range=(0, 3))
    df["resolution_severity"] = scaler.fit_transform(df[["resolution_hours"]])
    return df


# ─── STAGE 1: FUSE SIGNALS + PSEUDO-LABELS ───────────────────────────────────
def generate_pseudo_labels(df):
    print("[5/7] Fusing signals and generating pseudo-labels...")

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

    # Strict mismatch: delta >= 2 AND rule signal agrees
    df["rule_agrees"] = (
        ((df["severity_delta"] >= 2) & (df["rule_score_norm"] > 1.5)) |
        ((df["severity_delta"] <= -2) & (df["rule_score_norm"] < 1.0))
    )
    df["mismatch_label"] = (
        (df["severity_delta"].abs() >= 2) & df["rule_agrees"]
    ).astype(int)

    def mismatch_type(delta):
        if delta >= 2:  return "Hidden Crisis"
        elif delta <= -2: return "False Alarm"
        return "Consistent"

    df["mismatch_type"] = df["severity_delta"].apply(mismatch_type)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/pseudo_labeled.csv", index=False)
    print(f"    Mismatch distribution:\n{df['mismatch_label'].value_counts()}")
    return df


# ─── STAGE 2: PREPARE DATASET ─────────────────────────────────────────────────
def prepare_datasets(df, tokenizer):
    print("[6/7] Preparing datasets for training...")

    df_model = df[["full_text", "ticket_channel", "resolution_hours", "mismatch_label"]].copy().dropna()
    df_model["resolution_norm"] = MinMaxScaler().fit_transform(df_model[["resolution_hours"]])
    df_model["input_text"] = (
        df_model["full_text"] +
        " [CHANNEL] " + df_model["ticket_channel"] +
        " [RESTIME] " + df_model["resolution_norm"].round(2).astype(str)
    )

    train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=SEED, stratify=df_model["mismatch_label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED, stratify=train_df["mismatch_label"])

    def tokenize(batch):
        return tokenizer(batch["input_text"], truncation=True, padding=True, max_length=256)

    def to_hf_dataset(dataframe):
        ds = Dataset.from_pandas(dataframe[["input_text", "mismatch_label"]].reset_index(drop=True))
        ds = ds.map(tokenize, batched=True)
        ds = ds.rename_column("mismatch_label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    return to_hf_dataset(train_df), to_hf_dataset(val_df), to_hf_dataset(test_df), train_df


# ─── STAGE 2: WEIGHTED TRAINER ────────────────────────────────────────────────
def build_weighted_trainer(model, args, train_ds, val_ds, tokenizer, class_weights):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")
        }

    return WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )


# ─── STAGE 2: TRAIN ───────────────────────────────────────────────────────────
def train(train_ds, val_ds, test_ds, train_df, tokenizer):
    print("[7/7] Training classifier...")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=["q_lin", "v_lin"]
    )
    model = get_peft_model(base_model, lora_config)

    label_counts = train_df["mismatch_label"].value_counts().sort_index()
    total = len(train_df)
    class_weights = torch.tensor([total / (2 * c) for c in label_counts], dtype=torch.float)

    training_args = TrainingArguments(
        output_dir="models/deberta_lora",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        warmup_ratio=0.15,
        weight_decay=0.02,
        learning_rate=3e-5,
        logging_dir="outputs/metrics",
        seed=SEED,
        fp16=False,
        use_cpu=True
    )

    trainer = build_weighted_trainer(model, training_args, train_ds, val_ds, tokenizer, class_weights)
    trainer.train()

    # Evaluate
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    print(classification_report(labels, preds, target_names=["Consistent", "Mismatch"]))
    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")

    os.makedirs("outputs/metrics", exist_ok=True)
    metrics_out = classification_report(labels, preds, target_names=["Consistent", "Mismatch"], output_dict=True)
    with open("outputs/metrics/test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    os.makedirs("models/deberta_lora", exist_ok=True)
    model.save_pretrained("models/deberta_lora")
    tokenizer.save_pretrained("models/deberta_lora")
    print("Model saved to models/deberta_lora")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw_tickets.csv", help="Path to raw CSV")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = load_and_preprocess(args.data)
    df = compute_rule_score(df)
    df = compute_embedding_signal(df)
    df = compute_resolution_signal(df)
    df = generate_pseudo_labels(df)

    train_ds, val_ds, test_ds, train_df = prepare_datasets(df, tokenizer)
    train(train_ds, val_ds, test_ds, train_df, tokenizer)