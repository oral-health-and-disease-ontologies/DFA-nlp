# Topic Modeling — Dental Fear & Anxiety

A reproducible pipeline to ingest three academic PDFs on dental fear/anxiety, preprocess text, build features, train three topic models (**NMF**, **Anchored CorEx**, **BERTopic**), evaluate them (coherence, seed-overlap, diversity), and generate human-readable topic cards.

---

## 1) What we’re trying to achieve

**Research goal:** Distill recurring **themes** in a small, expert-written corpus and compare modeling approaches for **interpretability**, **domain alignment**, and **non‑redundancy**.

Why three models?
- **NMF (TF–IDF):** transparent, lexical topics; good baseline.
- **Anchored CorEx (seed‑guided):** steer topics toward known constructs (e.g., pain, claustrophobia).
- **BERTopic (embeddings):** clusters semantically similar chunks even when words differ; adapts `k` automatically.

**Primary outputs**
- Topic word lists (per model) → `models/*/*.json`
- Topic cards (Markdown) → `reports/topic_cards/<model>/`
- Metrics & comparison → `evaluation/*.csv`

---

## 2) Quickstart (end-to-end)

```bash
# Create and activate a virtual env (Linux/Mac)
python -m venv .venv && source .venv/bin/activate

# Install (full stack: pdfminer, CorEx, BERTopic, embeddings, umap, hdbscan)
pip install -U pip
pip install -r requirements-full.txt

# 1) Ingest PDFs → chunks
bash scripts/01_ingest.sh

# 2) Preprocess chunks → cleaned tokens
bash scripts/02_preprocess.sh

# 3) Train NMF, CorEx, BERTopic
bash scripts/03_run_models.sh

# 4) Evaluate (coherence, seed-overlap, diversity + comparison table)
bash scripts/04_evaluate.sh

# 5) Generate topic cards (Markdown)
bash scripts/05_make_report.sh
```

Key artifacts to show your professor:
- `evaluation/model_comparison.csv` — **one line per model** with seed-overlap total, mean coherence (c_v), and term diversity.
- `evaluation/coherence.csv` — **per-topic** coherence scores.
- `reports/topic_cards/<model>/topic_*.md` — **read these to interpret topics**.

---

## 3) Repository structure (what each folder/file does)

```
topic-modeling-dental-fear/
├─ configs/
│  ├─ base.yaml            # paths, chunk sizes, tf-idf, stopwords, random seed
│  ├─ nmf.yaml             # NMF hyperparams (k-range, sparsity, init)
│  ├─ corex.yaml           # CorEx hyperparams (n_topics, anchors, strength)
│  ├─ bertopic.yaml        # Embedding model, UMAP/HDBSCAN, top_n_words
│  └─ seeds.yaml           # Domain seed sets (pain, claustrophobia, etc.)
├─ data/
│  ├─ raw/                 # original PDFs (read-only)
│  ├─ interim/             # chunks.jsonl (post-ingest)
│  └─ processed/           # chunks_tokens.jsonl (preprocessed tokens)
├─ models/
│  ├─ nmf/                 # best_terms.json, grid_results.json
│  ├─ corex/               # topics.json
│  └─ bertopic/            # topics.json, bertopic_model.pkl
├─ evaluation/
│  ├─ extrinsic_overlap.csv    # seed-overlap per model
│  ├─ coherence.csv            # per-topic c_v
│  └─ model_comparison.csv     # aggregate summary
├─ reports/
│  ├─ figures/             # (optional) plots
│  └─ topic_cards/
│     ├─ nmf/              # topic_00.md, topic_01.md, ...
│     ├─ corex/            # topic_00.md, ...
│     └─ bertopic/         # topic_00.md, ...
├─ src/
│  ├─ ingest/
│  │  ├─ pdf_to_text.py    # pdfminer → text per PDF
│  │  └─ chunker.py        # split into ~320-token chunks with overlap
│  ├─ preprocess/
│  │  └─ clean.py          # normalize, lemmatize, stopwords, n-grams
│  ├─ features/
│  │  ├─ tfidf.py          # build TF–IDF + vocab
│  │  └─ seeds.py          # load seed sets for CorEx/extrinsic eval
│  ├─ models/
│  │  ├─ nmf_runner.py     # grid over k, save best terms
│  │  ├─ corex_runner.py   # anchored CorEx with binary features
│  │  └─ bertopic_runner.py# embeddings → UMAP → HDBSCAN → c-TF-IDF labels
│  ├─ eval/
│  │  ├─ extrinsic.py      # seed-overlap
│  │  ├─ coherence.py      # c_v using gensim
│  │  └─ compare.py        # aggregates comparison table
│  ├─ labeling/
│  │  └─ topic_cards.py    # write Markdown cards per topic
│  └─ utils/
│     ├─ io.py             # YAML/JSONL helpers
│     └─ logging.py        # (if present) simple logging config
├─ scripts/
│  ├─ 01_ingest.sh
│  ├─ 02_preprocess.sh
│  ├─ 03_run_models.sh
│  ├─ 04_evaluate.sh
│  └─ 05_make_report.sh
├─ README.md               # (this file)
└─ LICENSE
```

---

## 4) Flow — what runs first, what’s next

1) **Ingest** (`scripts/01_ingest.sh`)  
   - `src/ingest/pdf_to_text.py`: pdfminer → raw text  
   - `src/ingest/chunker.py`: chunk ~320 tokens w/ 50-token overlap  
   - **Writes:** `data/interim/chunks.jsonl`

2) **Preprocess** (`scripts/02_preprocess.sh`)  
   - `src/preprocess/clean.py`: normalize, lemmatize, stopwords (NLTK + domain), handle n‑grams  
   - **Writes:** `data/processed/chunks_tokens.jsonl`

3) **Modeling** (`scripts/03_run_models.sh`)  
   - **NMF:** `src/models/nmf_runner.py` → TF–IDF grid over `k`, save best terms  
   - **CorEx:** `src/models/corex_runner.py` → binary CSR + anchors from `configs/seeds.yaml`  
   - **BERTopic:** `src/models/bertopic_runner.py` → embeddings + UMAP/HDBSCAN + c‑TF‑IDF labels  
   - **Writes:** `models/*/*.json`, (BERTopic `.pkl`)

4) **Evaluation** (`scripts/04_evaluate.sh`)  
   - `src/eval/extrinsic.py` → `evaluation/extrinsic_overlap.csv`  
   - `src/eval/coherence.py` → `evaluation/coherence.csv`  
   - `src/eval/compare.py` → `evaluation/model_comparison.csv`

5) **Reporting** (`scripts/05_make_report.sh`)  
   - `src/labeling/topic_cards.py` → `reports/topic_cards/<model>/topic_*.md`

---

## 5) Parameters you might tweak

### Preprocess (in `configs/base.yaml`)
- `tfidf.ngram_min/ngram_max`: 1–3 (phrases); 2–3 improves labels but increases sparsity.
- `tfidf.max_df`: 0.85–0.95 filters boilerplate; raise to drop “author/publisher” noise.
- `extra_stopwords`: add domain names/URLs, ubiquitous author surnames.

### NMF (`configs/nmf.yaml`)
- `k_min/k_max`: topic count scan; try 3–10 for small corpora.
- `alpha_W/alpha_H, l1_ratio`: sparsity/interpretability trade-off (higher L1 yields sparser topics).
- `init`: `nndsvda` is stable; `max_iter`: 500–2000.

### CorEx (`configs/corex.yaml`)
- `n_topics`: try 3–8; small corpora prefer fewer topics.
- `use_anchors`: true/false; `anchor_strength`: 1.5–3.0 typical.
- `seeds.yaml`: refine seed lists to match constructs you care about.

### BERTopic (`configs/bertopic.yaml`)
- `embedding_model`: e.g., `all-MiniLM-L6-v2` (384‑d).
- `umap.n_neighbors`: (5–30); lower → finer clusters; higher → broader themes.
- `hdbscan.min_cluster_size/min_samples`: granularity vs. stability.
- `nr_topics`: `"auto"` or an integer; `top_n_words`: 10–20.

---


## 6) Metrics & how to read them

- **Coherence (c_v)**: ~0–1; >0.5 good; >0.6 solid; >0.7 strong (corpus‑dependent).  
  → See `evaluation/coherence.csv` (per-topic) and `model_comparison.csv` (mean).

- **Seed-overlap**: extrinsic alignment with your curated domain seeds; higher = closer to intended constructs.  
  → See `evaluation/extrinsic_overlap.csv` and totals in `model_comparison.csv`.

- **Term diversity**: fraction of unique words across topics; closer to 1.0 = less redundancy.  
  → Reported in `model_comparison.csv`.

**Explain to a reviewer:** *“We compared three complementary models. BERTopic had the strongest domain alignment and coherence; CorEx offered excellent topic separation with seeds; NMF was most transparent but weaker on this small corpus. We validated with both intrinsic (coherence/diversity) and extrinsic (seed overlap) measures, plus qualitative topic cards.”*


## Results (from this corpus & run)

We trained **three models** on the 3 PDFs (20 chunks after preprocessing):

- **NMF** (TF–IDF, grid search; best k = 3)
- **Anchored CorEx** (binary TF; domain seeds; n_topics from config)
- **BERTopic** (all-MiniLM-L6-v2 embeddings → UMAP → HDBSCAN → c-TF-IDF)

### 6.1 Model-level comparison

| Model     | Seed Overlap (↑) | Mean Coherence c_v (↑) | Term Diversity (↑) |
|-----------|-------------------|-------------------------|--------------------|
| BERTopic  | **6**             | **0.6917**              | **0.9333**         |
| CorEx     | 4                 | 0.5621                  | **1.0000**         |
| NMF       | 0                 | 0.3956                  | 0.3333             |

**Interpretation.** On this small academic corpus:
- **BERTopic** shows the best overall balance (highest mean coherence + strong seed alignment + high diversity).
- **CorEx** is a close second with **perfect diversity** and solid coherence, especially when seeds are aligned to constructs of interest.
- **NMF** is transparent but underperforms here (likely due to small data and author/publisher surface forms). It typically improves with more documents and tighter stoplists.

> Files: see `evaluation/model_comparison.csv` for these aggregates.

---

### 6.2 Per-topic coherence (c_v)

Coherence (c_v) is roughly on **[0, 1]**, where **>0.5 is decent**, **>0.6 good**, **>0.7 strong** (corpus-dependent).

**NMF (k = 3)**
- Topic 0: **0.3956**
- Topic 1: **0.3956**
- Topic 2: **0.3956**

**CorEx (n_topics per config; seed-guided)**
- Topic 0: **0.5093**
- Topic 1: **0.8044**  ← strongest coherent factor
- Topic 2: **0.4653**
- Topic 3: **0.4461**
- Topic 4: **0.5853**

**BERTopic (HDBSCAN inferred k = 3)**
- Topic 0: **0.7528**
- Topic 1: **0.5644**
- Topic 2: **0.7579**

> Files: full list in `evaluation/coherence.csv`.

---

### 6.3 What the metrics mean (quick recap)

- **Seed Overlap (↑)** — extrinsic alignment to curated domain seeds (e.g., *fear/anxiety/SNS, pain, claustrophobia, mutilation, assessment/exposure*). Higher = closer to intended constructs. Source: `evaluation/extrinsic_overlap.csv`.
- **Coherence (c_v) (↑)** — intrinsic interpretability proxy based on word co-occurrence and similarity. Higher = more self-consistent topics. Source: `evaluation/coherence.csv`.
- **Term Diversity (↑)** — fraction of unique top terms across a model’s topics. Higher = less redundancy. Source: `evaluation/model_comparison.csv`.

---

### 6.4 Reading topics (where to look)

For human interpretation and labeling with your professor, open the topic cards:
- `reports/topic_cards/nmf/topic_*.md`
- `reports/topic_cards/corex/topic_*.md`
- `reports/topic_cards/bertopic/topic_*.md`

Each card lists the **top words** (and optionally seed/anchor notes for CorEx). Use these cards to merge/split/rename topics as needed.

---

### 6.5 Suggested next steps

- Add 5–10 more relevant PDFs → improves NMF/CorEx robustness and BERTopic stability.
- Extend **extra_stopwords** in `configs/base.yaml` (author names, publisher boilerplate).
- Tune **CorEx** anchors and `anchor_strength` for constructs you care about most.
- Adjust **BERTopic** granularity via `umap.n_neighbors`, `hdbscan.min_cluster_size`, and `min_topic_size`.

---

## 7) Troubleshooting (common issues)

- **`pdfminer` not found** → `pip install -r requirements-full.txt`; rerun `01_ingest.sh`.
- **CorEx `.A1` attribute error** → ensure binary CSR is passed (handled in runner); upgrade `corextopic` if needed.
- **BERTopic `IsADirectoryError` on save** → save to a file path (e.g., `bertopic_model.pkl`), not a directory.
- **Coherence “unable to interpret topic”** → we join multi-words with underscores and filter to dictionary; falling back to unigrams fixes it.
- **Empty vocabulary** → relax stopwording; reduce `ngram_max` to 2; raise `max_df` slightly; confirm tokens exist.

---

## 8) Minimal “how to read outputs”

```python
# Compare models
import pandas as pd
print(pd.read_csv("evaluation/model_comparison.csv"))

# NMF topics
import json
print(json.load(open("models/nmf/best_terms.json"))["terms"][0])

# CorEx topics
print(json.load(open("models/corex/topics.json"))[0])

# BERTopic topics
print(json.load(open("models/bertopic/topics.json"))[0])
```

---

## 9) Reproducibility & environment

- Pin Python version in `.python-version` if you use pyenv.  
- Use `requirements-full.txt` for full dependencies.  
- Random seeds live in `configs/base.yaml`. Reproducibility across environments requires the **same** versions of Python, sklearn, UMAP/HDBSCAN, sentence-transformers, and BERTopic.

---

## 10) License & citation

- See `LICENSE`.  
- If you use this in a paper, cite the underlying libraries (scikit-learn, corextopic, BERTopic, sentence-transformers, UMAP, HDBSCAN) and your data sources (SAGE Encyclopedia, McNeil & Berryman, McNeil & Randall).

---

## 11) One‑sentence summary for your professor

> “We ingest, clean, and chunk the PDFs; train three complementary topic models (NMF, Anchored CorEx, BERTopic); and compare them with coherence, seed‑overlap, and term diversity, then read topic cards to interpret what the models found about dental fear/anxiety.”
