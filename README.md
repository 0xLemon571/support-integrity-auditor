# 🔍 Support Integrity Auditor (SIA)

A semantics-driven, evidence-grounded auditor that detects **Priority Mismatch** in CRM support tickets — cases where a ticket's true severity (inferred from text, channel, and resolution behavior) conflicts with its human-assigned priority label.

Unlike keyword-matching triage tools, SIA bootstraps its own supervision signal from raw, unlabeled ticket data, trains a fine-tuned classifier on the resulting pseudo-labels, and produces a hallucination-free **Evidence Dossier** for every flagged ticket.

---

## 1. Problem

Manual ticket triage is vulnerable to agent fatigue, favoritism, and keyword anchoring. A ticket described as "Low" priority may actually be a production outage; a "Critical" ticket may be a routine question. SIA identifies these mismatches **without any pre-existing mismatch labels**, by fusing multiple independent severity signals into a self-supervised pseudo-label, then training a classifier on top of that signal.

---

## 2. Architecture

```
Raw Tickets (CSV)
       │
       ▼
┌─────────────────────────────┐
│  STAGE 1 — Pseudo-Labeling  │
│  (self-supervised)          │
│                              │
│  • Rule-based NLP signal     │
│  • Embedding cluster signal  │
│  • Resolution-time signal    │
│         │                    │
│   z-score fusion             │
│         │                    │
│  inferred_severity_score     │
│         │                    │
│  severity_delta = inferred − assigned
│         │                    │
│  mismatch_label (0/1)        │
└─────────────┬────────────────┘
              ▼
┌─────────────────────────────┐
│  STAGE 2 — Classifier        │
│  DistilBERT + LoRA adapter   │
│  (text + channel + res_time) │
│  weighted CrossEntropy loss  │
│  (handles class imbalance)   │
└─────────────┬────────────────┘
              ▼
┌─────────────────────────────┐
│  STAGE 3 — Evidence Dossier  │
│  Structured, field-grounded  │
│  JSON per flagged ticket     │
└─────────────┬────────────────┘
              ▼
     predict.py / app.py
```

---

## 3. Pipeline Stages

### Stage 1 — Self-Supervised Pseudo-Label Generation

Three independent, complementary signals are computed per ticket:

| Signal | Method | Rationale |
|---|---|---|
| **Rule-based NLP** | Keyword density over crisis vs. low-urgency lexicons, minus negation penalty (`not`, `no`, `never`, `without`), percentile-ranked to [0, 3] | Captures explicit lexical urgency cues |
| **Embedding clustering** | `all-MiniLM-L6-v2` sentence embeddings → KMeans (k=4) → clusters ranked by mean resolution time | Captures semantic/topical urgency independent of exact keywords |
| **Resolution-time** | Parsed resolution time (hours), Min-Max scaled to [0, 3] | Longer resolution often correlates with real complexity/severity |

**Fusion strategy:** each signal is **z-score normalized** before fusing, so no single signal's raw scale dominates:

```
inferred_severity_score = 0.4 · rule_z + 0.4 · cluster_z + 0.2 · resolution_z
```

Text-derived signals (rule-based + embedding) are weighted equally and higher than resolution time, since resolution time is a noisier, indirect proxy (affected by staffing, SLA queue position, etc.) rather than a direct severity indicator.

The fused score is bucketed into `low / medium / high / critical` using **percentile thresholds** (25th/50th/75th) relative to the batch, compared against the human-assigned `ticket_priority`, and the signed difference becomes `severity_delta`.

A binary `mismatch_label` is only set to 1 when `|severity_delta| ≥ 2` **and** the rule-based signal independently agrees with the direction of the mismatch — this cross-signal agreement requirement reduces false pseudo-labels from any single noisy signal.

```
mismatch_type = "Hidden Crisis"  if severity_delta ≥ 2   (under-prioritized)
              = "False Alarm"    if severity_delta ≤ -2  (over-prioritized)
              = "Consistent"     otherwise
```

**Pseudo-label distribution** (n = 20,000 tickets, `data/pseudo_labeled.csv`):

| | Count | % |
|---|---|---|
| Consistent (`mismatch_label = 0`) | 11,701 | 58.5% |
| Mismatch (`mismatch_label = 1`) | 8,299 | 41.5% |
| — Hidden Crisis (`severity_delta ≥ 2`) | 6,693 | 33.5% |
| — False Alarm (`severity_delta ≤ -2`) | 2,444 | 12.2% |

> Note: Hidden Crisis + False Alarm (9,137) exceeds the strict `mismatch_label = 1` count (8,299) because `mismatch_type` is derived from `severity_delta` alone, while `mismatch_label` additionally requires rule-signal agreement — some large-delta tickets don't clear that second gate.

**Ablation — individual signal contribution** (recomputed by rebuilding the pseudo-label from each signal in isolation and measuring agreement against the full-fusion `mismatch_label`; see `compute_ablation.py`):

| Configuration | Mismatch Rate | Agreement w/ Full Fusion |
|---|---|---|
| Rule-based only | 23.2% | 60.6% |
| Embedding cluster only | 44.4% | 86.0% |
| Resolution-time only | 45.9% | 66.1% |
| Rule + Embedding (no resolution) | 41.2% | 90.7% |
| **Full fusion (0.4 / 0.4 / 0.2)** | 45.7% | 95.8%* |

*Measured against a `|delta| ≥ 2` reconstruction of the fused score alone, without the additional rule-agreement gate applied to the true `mismatch_label` — hence < 100%. This isolates how much the *severity scoring* (as opposed to the extra agreement gate) is driven by each signal.

**Pairwise signal agreement** (how often two signals' independent severity buckets agree with each other):

| Pair | Agreement |
|---|---|
| Rule vs. Embedding Cluster | 27.1% |
| Rule vs. Resolution-Time | 25.4% |
| Embedding Cluster vs. Resolution-Time | 27.1% |

Low pairwise agreement across all signal pairs supports the design choice to fuse three independent signals rather than rely on any single one — each captures a different, largely non-redundant slice of "true severity," and the embedding-cluster signal contributes the most standalone signal toward the final fused label (86.0% agreement alone vs. 60.6% for rule-based alone).

### Stage 2 — Classifier Training

- **Base model:** `distilbert-base-uncased`, adapted via **LoRA** (`r=8`, `alpha=16`, dropout `0.1`, targeting `q_lin`/`v_lin`) — not a frozen zero-shot pipeline.
- **Inputs:** ticket text (`subject + description`) concatenated with structured metadata tags — `[CHANNEL] <channel>` and `[RESTIME] <normalized resolution time>`.
- **Labels:** binary `mismatch_label` from Stage 1.
- **Imbalance handling:** inverse-frequency **class-weighted CrossEntropyLoss** via a custom `WeightedTrainer`. Base class balance is 58.5% / 41.5%, a moderate imbalance the weighted loss corrects for.
- **Training config:** 8 epochs, batch size 8 (train) / 32 (eval), LR `3e-5`, warmup ratio `0.15`, weight decay `0.02`, best checkpoint selected by macro F1, CPU training.
- **Split:** 72% train / 8% val / 20% test, stratified on `mismatch_label`.

### Stage 3 — Evidence Dossier Generation

Every ticket predicted as a mismatch (`prediction == 1`) gets a structured JSON dossier, with every `feature_evidence` item traceable to a real input field (no fabricated claims):

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "...",
  "feature_evidence": [
    { "signal": "keyword", "value": "matched crisis keywords", "weight": "..." },
    { "signal": "resolution_time_hours", "value": "...", "interpretation": "..." },
    { "signal": "embedding_cluster", "value": "Cluster N", "weight": "..." }
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": "..."
}
```

---

## 4. Repository Structure

```
.
├── data/
│   ├── customer_support_data.csv   # raw input
│   └── pseudo_labeled.csv          # Stage 1 output (20,000 tickets)
├── models/deberta_lora/            # saved tokenizer + LoRA adapter weights
├── outputs/
│   ├── predictions.csv
│   ├── ablation_table.csv
│   ├── dossiers/dossiers.json
│   └── metrics/test_metrics.json
├── notebook.ipynb                  # full reproducible pipeline (pseudo-label → train → infer)
├── train_pipeline.py                # standalone training script
├── compute_ablation.py              # recomputes the Stage 1 ablation table
├── predict.py                       # CSV in → predictions + dossiers out
├── app.py                           # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 5. Setup

```bash
git clone <repo-url>
cd support-integrity-auditor
pip install -r requirements.txt
```

Download the dataset from [Customer Support Tickets — CRM Dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data) and place it at `data/customer_support_data.csv`.

---

## 6. Usage

### Train

```bash
python train_pipeline.py --data data/customer_support_data.csv
```

Runs the full 7-step pipeline: load/preprocess → rule signal → embedding signal → resolution signal → pseudo-label fusion → dataset prep → LoRA fine-tuning. Saves the adapter to `models/deberta_lora/`, the pseudo-labeled dataset to `data/pseudo_labeled.csv`, and test metrics to `outputs/metrics/test_metrics.json`.

### Recompute the ablation table

```bash
python compute_ablation.py --data data/pseudo_labeled.csv
```

### Predict / Audit a CSV

```bash
python predict.py --input data/customer_support_data.csv \
                   --output outputs/predictions.csv \
                   --dossier outputs/dossiers/dossiers.json
```

### Run the Web App

```bash
streamlit run app.py
```

Provides two modes:
- **Single Ticket Audit** — form input, instant judgment + dossier.
- **Batch CSV Audit** — upload a CSV, get a full **Priority Mismatch Dashboard** (mismatch type distribution, assigned-vs-inferred priority chart, severity-delta heatmap by ticket type × channel, signal fusion weights, confidence distribution, flagged ticket table, and downloadable dossiers).

---

## 7. Evaluation

| Metric | Threshold | Result | Pass? |
|---|---|---|---|
| Binary Classification Accuracy | ≥ 83% | 72.3% | ❌ |
| Macro F1 Score | ≥ 0.82 | 0.705 | ❌ |
| Per-Class Recall (Consistent) | ≥ 0.78 | 0.669 | ❌ |
| Per-Class Recall (Mismatch) | ≥ 0.78 | 0.864 | ✅ |
| Pseudo-Label Signal Agreement | — | 27.1% (rule↔cluster), 25.4% (rule↔resolution), 27.1% (cluster↔resolution) | — |
| Adversarial Robustness (10 held-out) | ≥ 7/10 for bonus | TBD | — |

**Full test-set breakdown** (n = 4,000, from `outputs/metrics/test_metrics.json`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Consistent | 0.928 | 0.669 | 0.777 | 2,891 |
| Mismatch | 0.500 | 0.864 | 0.634 | 1,109 |

**Current status: not verified.** 3 of 4 required thresholds fail (§6 of the spec). The pattern is informative: recall on `Mismatch` is strong (0.864) and its precision is weak (0.500), while `Consistent` is the mirror image (0.928 precision, 0.669 recall) — the model is over-predicting the minority `Mismatch` class. This is a known side effect of aggressive inverse-frequency class weighting on a moderate (58.5/41.5) imbalance; the weight is likely overcorrecting.

**Suggested next steps to clear the thresholds:**
- Reduce the class-weight ratio (e.g., weight by `sqrt(total / (2·count))` instead of the full inverse frequency, or cap the weight ratio) so `Mismatch` isn't over-predicted at the expense of `Consistent` precision/recall.
- Try a lower learning rate or fewer epochs — recall this skewed can indicate the model latched onto a shortcut rather than the intended signal.
- Re-check whether `load_best_model_at_end` with `metric_for_best_model="f1_macro"` is actually selecting the best checkpoint, or whether an earlier/later epoch balances the two classes better.

Adversarial robustness still requires manually scoring 10 hand-crafted tickets designed to fool keyword-only systems.

---

## 8. Notes / Limitations

- Pseudo-labels are a proxy for ground truth; classifier performance is bounded by fusion signal quality.
- Resolution-time signal is noisy (queue/staffing effects), so it's intentionally down-weighted (0.2) in the fusion, and shows the weakest pairwise agreement with the other two signals (25-27%).
- The embedding-cluster signal is the strongest standalone predictor of the fused label (86.0% agreement alone), suggesting semantic/topical urgency carries more information than raw keyword density for this dataset.
- Single-ticket inference in `app.py` uses fixed heuristic thresholds instead of percentile bucketing, since percentiles are undefined for `n=1`.
