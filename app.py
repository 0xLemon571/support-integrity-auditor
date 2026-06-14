import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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

st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide"
)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── PREPROCESSING ────────────────────────────────────────────────────────────
def preprocess_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["ticket_description"] = df["ticket_description"].fillna("")
    df["ticket_subject"]     = df["ticket_subject"].fillna("")
    df["ticket_channel"]     = df["ticket_channel"].fillna("unknown")
    df["ticket_priority"]    = df["ticket_priority"].str.strip().str.lower()
    df["full_text"]          = df["ticket_subject"] + " " + df["ticket_description"]
    df["priority_numeric"]   = df["ticket_priority"].map(PRIORITY_MAP)

    def parse_rt(val):
        if pd.isna(val): return np.nan
        val = str(val).lower()
        nums = re.findall(r"[\d.]+", val)
        if not nums: return np.nan
        return float(nums[0]) * 24 if "day" in val else float(nums[0])

    df["resolution_hours"] = df["resolution_time"].apply(parse_rt)
    df["resolution_hours"] = df["resolution_hours"].fillna(df["resolution_hours"].median())
    return df


# ─── SIGNAL COMPUTATION ───────────────────────────────────────────────────────
def compute_signals(df, embedder):
    is_single = len(df) == 1

    # Rule-based
    def rule_score(text):
        text = text.lower()
        score  = sum(1 for kw in CRISIS_KEYWORDS if kw in text)
        score -= sum(0.5 for kw in LOW_KEYWORDS if kw in text)
        score -= len(re.findall(r"\bnot\b|\bno\b|\bnever\b|\bwithout\b", text)) * 0.3
        return score

    df["rule_score"]      = df["full_text"].apply(rule_score)
    df["rule_score_norm"] = df["rule_score"].rank(pct=True) * 3

    # Embedding clustering — skip KMeans for single ticket
    embeddings = embedder.encode(df["full_text"].tolist(), batch_size=64, show_progress_bar=False)

    if is_single:
        # For single ticket assign middle cluster severity directly from rule score
        df["cluster"] = 0
        rule = df["rule_score"].iloc[0]
        if rule >= 2:
            df["cluster_severity"] = 3
        elif rule >= 1:
            df["cluster_severity"] = 2
        elif rule >= 0:
            df["cluster_severity"] = 1
        else:
            df["cluster_severity"] = 0
    else:
        kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
        df["cluster"] = kmeans.fit_predict(embeddings)
        cluster_sev = df.groupby("cluster")["resolution_hours"].mean().sort_values()
        df["cluster_severity"] = df["cluster"].map({c: r for r, c in enumerate(cluster_sev.index)})

    # Resolution time
    df["resolution_severity"] = MinMaxScaler(feature_range=(0, 3)).fit_transform(df[["resolution_hours"]])

    # Z-score fusion — skip zscore for single ticket (zscore of 1 value = 0)
    if is_single:
        df["rule_z"]       = df["rule_score_norm"] - 1.5   # center around midpoint
        df["cluster_z"]    = df["cluster_severity"].astype(float) - 1.5
        df["resolution_z"] = df["resolution_severity"] - 1.5
    else:
        from scipy.stats import zscore
        df["rule_z"]       = zscore(df["rule_score_norm"])
        df["cluster_z"]    = zscore(df["cluster_severity"].astype(float))
        df["resolution_z"] = zscore(df["resolution_severity"])

    df["inferred_severity_score"] = (
        0.4 * df["rule_z"] +
        0.4 * df["cluster_z"] +
        0.2 * df["resolution_z"]
    )

    if is_single:
        # Fixed thresholds for single ticket
        score = df["inferred_severity_score"].iloc[0]
        def score_to_label_single(s):
            if s < -0.5:   return "low"
            elif s < 0.5:  return "medium"
            elif s < 1.5:  return "high"
            else:          return "critical"
        df["inferred_severity"] = df["inferred_severity_score"].apply(score_to_label_single)
    else:
        p25 = df["inferred_severity_score"].quantile(0.25)
        p50 = df["inferred_severity_score"].quantile(0.50)
        p75 = df["inferred_severity_score"].quantile(0.75)

        def score_to_label(s):
            if s <= p25:   return "low"
            elif s <= p50: return "medium"
            elif s <= p75: return "high"
            else:          return "critical"

        df["inferred_severity"] = df["inferred_severity_score"].apply(score_to_label)

    df["inferred_severity_numeric"] = df["inferred_severity"].map(PRIORITY_MAP)
    df["severity_delta"]            = df["inferred_severity_numeric"] - df["priority_numeric"]
    return df


# ─── INFERENCE ────────────────────────────────────────────────────────────────
def run_inference(df, tokenizer, model):
    df["resolution_norm"] = MinMaxScaler().fit_transform(df[["resolution_hours"]])
    df["input_text"] = (
        df["full_text"] +
        " [CHANNEL] " + df["ticket_channel"] +
        " [RESTIME] " + df["resolution_norm"].round(2).astype(str)
    )

    preds, confs = [], []
    for _, row in df.iterrows():
        inputs = tokenizer(
            row["input_text"], return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred  = torch.argmax(probs).item()
        preds.append(pred)
        confs.append(round(probs[pred].item(), 4))

    df["prediction"]       = preds
    df["model_confidence"] = confs
    return df


# ─── DOSSIER ──────────────────────────────────────────────────────────────────
def generate_dossier(row, median_res):
    assigned = row["ticket_priority"]
    inferred = row["inferred_severity"]
    delta    = int(row["severity_delta"])
    text     = row["full_text"].lower()

    matched  = [kw for kw in CRISIS_KEYWORDS if kw in text]
    kw_weight = round(min(len(matched) / len(CRISIS_KEYWORDS), 1.0), 3)

    return {
        "ticket_id":        str(row.get("ticket_id", row.name)),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type":    "Hidden Crisis" if delta > 0 else "False Alarm",
        "severity_delta":   str(delta),
        "feature_evidence": [
            {
                "signal": "keyword",
                "value":  ", ".join(matched[:5]) if matched else "none",
                "weight": str(kw_weight)
            },
            {
                "signal":         "resolution_time",
                "value":          str(round(row["resolution_hours"], 1)) + " hours",
                "interpretation": "Above median — elevated complexity" if row["resolution_hours"] > median_res else "Below median — routine"
            },
            {
                "signal": "embedding_cluster",
                "value":  f"Cluster {row['cluster']}",
                "weight": str(round(row["cluster_severity"] / 3, 3))
            }
        ],
        "constraint_analysis": (
            f"Ticket assigned '{assigned}' but signals indicate '{inferred}' severity. "
            f"Resolution time of {round(row['resolution_hours'], 1)}h and cluster membership "
            f"support this discrepancy. Severity delta: {delta}."
        ),
        "confidence": str(round(row["model_confidence"], 3))
    }


# ─── SINGLE TICKET FORM ───────────────────────────────────────────────────────
def single_ticket_ui(tokenizer, model, embedder):
    st.subheader("🎫 Single Ticket Audit")

    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            subject     = st.text_input("Ticket Subject", placeholder="e.g. System outage affecting all users")
            description = st.text_area("Ticket Description", height=150, placeholder="Full description of the issue...")
            priority    = st.selectbox("Assigned Priority", ["low", "medium", "high", "critical"])
        with col2:
            channel         = st.selectbox("Ticket Channel", ["email", "chat", "phone", "social media"])
            resolution_time = st.text_input("Resolution Time", placeholder="e.g. 2 days or 48 hours")
            ticket_type     = st.text_input("Ticket Type", placeholder="e.g. Technical Issue")

        submitted = st.form_submit_button("🔍 Audit Ticket", use_container_width=True)

    if submitted:
        row = {
            "ticket_subject":    subject,
            "ticket_description": description,
            "ticket_priority":   priority,
            "ticket_channel":    channel,
            "resolution_time":   resolution_time,
            "ticket_type":       ticket_type
        }
        df_single = pd.DataFrame([row])
        df_single = preprocess_df(df_single)

        with st.spinner("Analyzing ticket..."):
            df_single = compute_signals(df_single, embedder)
            df_single = run_inference(df_single, tokenizer, model)

        result = df_single.iloc[0]
        pred   = result["prediction"]
        median_res = df_single["resolution_hours"].median()

        st.divider()
        if pred == 0:
            st.success("✅ **CONSISTENT** — Priority assignment looks correct.")
        else:
            mtype = "Hidden Crisis 🚨" if result["severity_delta"] > 0 else "False Alarm ⚠️"
            st.error(f"❌ **MISMATCH DETECTED** — {mtype}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assigned Priority",  result["ticket_priority"].upper())
        col2.metric("Inferred Severity",  result["inferred_severity"].upper())
        col3.metric("Severity Delta",     result["severity_delta"])
        col4.metric("Model Confidence",   f"{result['model_confidence']:.1%}")

        if pred == 1:
            st.subheader("📋 Evidence Dossier")
            dossier = generate_dossier(result, median_res)
            st.json(dossier)


# ─── BATCH CSV UI ─────────────────────────────────────────────────────────────
def batch_ui(tokenizer, model, embedder):
    st.subheader("📂 Batch CSV Audit")
    uploaded = st.file_uploader("Upload tickets CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_raw)}** tickets.")

        if st.button("🚀 Run Batch Audit", use_container_width=True):
            with st.spinner("Processing all tickets... this may take a few minutes."):
                df = preprocess_df(df_raw)
                df = compute_signals(df, embedder)
                df = run_inference(df, tokenizer, model)

            st.success("Audit complete!")
            st.session_state["batch_results"] = df

    if "batch_results" in st.session_state:
        df = st.session_state["batch_results"]
        median_res = df["resolution_hours"].median()
        mismatched = df[df["prediction"] == 1]

        # ── Dashboard ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Priority Mismatch Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tickets",    len(df))
        c2.metric("Mismatches",       len(mismatched))
        c3.metric("Mismatch Rate",    f"{len(mismatched)/len(df):.1%}")
        c4.metric("Hidden Crises",    len(mismatched[mismatched["severity_delta"] > 0]))

        col1, col2 = st.columns(2)

        with col1:
            # Mismatch type distribution
            type_counts = df["mismatch_type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig1 = px.pie(type_counts, names="Type", values="Count",
                          title="Mismatch Type Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Assigned vs inferred priority counts
            assigned_counts = df["ticket_priority"].value_counts().reset_index()
            assigned_counts.columns = ["Priority", "Count"]
            assigned_counts["Type"] = "Assigned"
            inferred_counts = df["inferred_severity"].value_counts().reset_index()
            inferred_counts.columns = ["Priority", "Count"]
            inferred_counts["Type"] = "Inferred"
            combined = pd.concat([assigned_counts, inferred_counts])
            fig2 = px.bar(combined, x="Priority", y="Count", color="Type",
                          barmode="group", title="Assigned vs Inferred Priority",
                          color_discrete_sequence=["#636EFA", "#EF553B"])
            st.plotly_chart(fig2, use_container_width=True)

        # ── Severity delta heatmap ────────────────────────────────────────────
        st.subheader("🌡️ Severity Delta Heatmap")
        if "ticket_type" in df.columns and "ticket_channel" in df.columns:
            heatmap_data = df.groupby(["ticket_type", "ticket_channel"])["severity_delta"].mean().unstack(fill_value=0)
            fig3 = px.imshow(
                heatmap_data,
                color_continuous_scale="RdYlGn_r",
                title="Mean Severity Delta by Ticket Type × Channel",
                labels={"color": "Avg Delta"}
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Top contributing signals ──────────────────────────────────────────
        st.subheader("🔑 Top Contributing Signals")
        col1, col2 = st.columns(2)

        with col1:
            sig_data = pd.DataFrame({
                "Signal": ["Rule-Based NLP", "Embedding Cluster", "Resolution Time"],
                "Weight": [0.4, 0.4, 0.2]
            })
            fig4 = px.bar(sig_data, x="Signal", y="Weight",
                          title="Signal Fusion Weights",
                          color="Weight", color_continuous_scale="Blues")
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            conf_fig = px.histogram(
                mismatched, x="model_confidence", nbins=20,
                title="Confidence Distribution (Mismatches)",
                color_discrete_sequence=["#EF553B"]
            )
            st.plotly_chart(conf_fig, use_container_width=True)

        # ── Flagged tickets table ─────────────────────────────────────────────
        st.subheader("🚩 Flagged Tickets")
        display_cols = ["ticket_priority", "inferred_severity", "severity_delta",
                        "mismatch_type", "model_confidence", "ticket_channel"]
        available = [c for c in display_cols if c in mismatched.columns]
        st.dataframe(mismatched[available].reset_index(drop=True), use_container_width=True)

        # ── Dossiers ─────────────────────────────────────────────────────────
        st.subheader("📋 Evidence Dossiers")
        dossiers = [generate_dossier(row, median_res) for _, row in mismatched.iterrows()]

        dossier_json = json.dumps(dossiers, indent=2)
        st.download_button(
            label="⬇️ Download All Dossiers (JSON)",
            data=dossier_json,
            file_name="dossiers.json",
            mime="application/json"
        )

        with st.expander("Preview first 3 dossiers"):
            for d in dossiers[:3]:
                st.json(d)
                st.divider()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.title("🔍 Support Integrity Auditor (SIA)")
    st.caption("Semantics-driven priority mismatch detection for CRM support tickets.")

    with st.spinner("Loading model..."):
        tokenizer, model = load_model()
        embedder = load_embedder()

    tab1, tab2 = st.tabs(["🎫 Single Ticket", "📂 Batch CSV"])

    with tab1:
        single_ticket_ui(tokenizer, model, embedder)

    with tab2:
        batch_ui(tokenizer, model, embedder)


if __name__ == "__main__":
    main()