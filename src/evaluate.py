import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from bert_score import score as bert_score
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "corpus_sample_150_manual.csv"
KEYWORD_PATH = BASE_DIR / "data" / "keyword_dictionary.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

EVALUATION_SET_PATH = OUTPUT_DIR / "evaluation_set.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1 — Load data and get topic assignments
# ============================================================

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

# Parse CSV: each column = one category, rows = keywords
kw_df = pd.read_csv(KEYWORD_PATH)
keyword_categories = {}
for col in kw_df.columns:
    keywords = [str(k).strip() for k in kw_df[col].dropna() if str(k).strip()]
    keyword_categories[col] = keywords
CATEGORIES = list(keyword_categories.keys())
print(f"Loaded {len(CATEGORIES)} categories from {KEYWORD_PATH.name}")

# Label normalizer: lower-case, strip whitespace, remove punctuation, collapse spaces
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9 ]", " ", s)).strip().lower()

_label_map = {_norm_key(cat): cat for cat in CATEGORIES}

def normalize_label(label: str) -> str:
    return _label_map.get(_norm_key(label), label)

print("Loading BERTopic model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")
topic_model = BERTopic.load(str(MODEL_PATH), embedding_model=embedding_model)

docs = df["text"].tolist()
print("Running model prediction")
topics, probs = topic_model.transform(docs)

# Normalize probs to a single float per document
probs_arr = np.array(probs) if not isinstance(probs, np.ndarray) else probs
if probs_arr.ndim == 2:
    topic_probs = probs_arr.max(axis=1).tolist()
else:
    topic_probs = probs_arr.tolist()

df["predicted_topic_id"] = topics
df["topic_probability"] = topic_probs

outlier_count = sum(1 for t in topics if t == -1)
print(f"\nOutlier chunks (topic_id == -1): {outlier_count} / {len(df)} "
      f"({outlier_count / len(df) * 100:.1f}%)")

# ============================================================
# STEP 2 — Split multi-label rows into single-label rows
# ============================================================

original_count = len(df)
multi_label_count = df["manual_label"].str.contains(";", na=False).sum()

df_exploded = df.copy()
df_exploded["manual_label"] = df_exploded["manual_label"].str.split(";")
df_exploded = df_exploded.explode("manual_label")
df_exploded["manual_label"] = df_exploded["manual_label"].str.strip().apply(normalize_label)
df_exploded = df_exploded[df_exploded["manual_label"] != ""].reset_index(drop=True)
exploded_count = len(df_exploded)

# Report any labels that didn't match a known category 
unknown_labels = set(df_exploded["manual_label"]) - set(CATEGORIES)
if unknown_labels:
    print(f"  WARNING: unrecognized labels after normalization: {unknown_labels}")

df_exploded.to_csv(EVALUATION_SET_PATH, index=False)

print(f"Original chunk count:   {original_count}")
print(f"Exploded row count:     {exploded_count}")
print(f"Multi-label chunks:     {multi_label_count}")
print(f"Saved to: {EVALUATION_SET_PATH}")

# ============================================================
# STEP 3 — BERTScore: topic keywords vs manual_label keywords
# ============================================================

# Build topic keyword strings
unique_topics = [t for t in df_exploded["predicted_topic_id"].unique() if t != -1]
topic_keyword_strings = {}
for tid in unique_topics:
    words_scores = topic_model.get_topic(tid)
    topic_keyword_strings[tid] = " ".join(w for w, _ in words_scores)

# Build category reference strings from dictionary
category_ref_strings = {
    cat: " ".join(keyword_categories[cat])
    for cat in CATEGORIES
}

# For each row in evaluation_set: 
# candidate = topic keywords, reference = manual_label category keywords
cands, refs = [], []
for _, row in df_exploded.iterrows():
    tid = row["predicted_topic_id"]
    label = row["manual_label"]
    
    if tid == -1:
        cands.append("unassigned")
    else:
        cands.append(topic_keyword_strings[tid])
    
    refs.append(category_ref_strings.get(label, label))

print(f"Scoring {len(cands)} chunk-level pairs...")
_, _, F1_all = bert_score(cands, refs, lang="en", verbose=True)

df_exploded["bertscore_f1"] = [round(float(f), 4) for f in F1_all]

print(f"\nOverall BERTScore F1:")
print(f"  Mean:   {df_exploded['bertscore_f1'].mean():.4f}")
print(f"  Std:    {df_exploded['bertscore_f1'].std():.4f}")
print(f"  Min:    {df_exploded['bertscore_f1'].min():.4f}")
print(f"  Max:    {df_exploded['bertscore_f1'].max():.4f}")

print(f"\nBERTScore F1 per manual_label:")
print(df_exploded.groupby("manual_label")["bertscore_f1"].mean().sort_values(ascending=False).to_string())

df_exploded.to_csv(EVALUATION_SET_PATH, index=False)
print(f"\nUpdated evaluation_set.csv with bertscore_f1: {EVALUATION_SET_PATH}")

# ============================================================
# STEP 4 — Build evaluation_results.csv 
# ============================================================

# --- BERTScore F1 per topic (average across all chunks in that topic) ---
bertscore_per_topic = (
    df_exploded[df_exploded["predicted_topic_id"] != -1]
    .groupby("predicted_topic_id")["bertscore_f1"]
    .mean()
    .round(4)
)

# --- n_docs per topic  ---
topic_info = topic_model.get_topic_info()
# topic_info has columns: Topic, Count, Name, Representation, ...
topic_info = topic_info[topic_info["Topic"] != -1].copy()

# --- Topic label ---
def clean_topic_name(name: str) -> str:
    parts = name.split("_")
    # drop the leading topic_id number
    parts = [p for p in parts if not p.isdigit()]
    return " ".join(parts).title()

topic_info["topic_label"] = topic_info["Name"].apply(clean_topic_name)

# --- Coherence per topic  ---
print("Computing coherence scores")
full_docs = df["text"].tolist()
tokenized = [doc.lower().split() for doc in full_docs]
n_docs = len(tokenized)

coherence_scores = {}
for tid in topic_info["Topic"].tolist():
    words_scores = topic_model.get_topic(tid)
    if not words_scores:
        coherence_scores[tid] = None
        continue
    topic_words = [w for w, _ in words_scores]

    pairs_score = []
    for i in range(len(topic_words)):
        for j in range(i + 1, len(topic_words)):
            w1, w2 = topic_words[i], topic_words[j]
            d_w1 = sum(1 for d in tokenized if w1 in d)
            d_w2 = sum(1 for d in tokenized if w2 in d)
            d_both = sum(1 for d in tokenized if w1 in d and w2 in d)
            if d_w1 > 0 and d_w2 > 0 and d_both > 0:
                pmi = np.log((d_both * n_docs) / (d_w1 * d_w2))
                pairs_score.append(pmi)
    coherence_scores[tid] = round(float(np.mean(pairs_score)), 4) if pairs_score else None


# --- Topic Diversity ---
all_topic_words = []
for tid in topic_info["Topic"].tolist():
    words_scores = topic_model.get_topic(tid)
    if words_scores:
        all_topic_words.extend([w for w, _ in words_scores])

total_words = len(all_topic_words)
unique_words = len(set(all_topic_words))
overall_diversity = round(unique_words / total_words, 4) if total_words > 0 else None
print(f"Overall topic diversity: {overall_diversity}")

# --- Assemble final results table ---
rows = []
for _, trow in topic_info.iterrows():
    tid = trow["Topic"]
    rows.append({
        "topic_id": tid,
        "topic_label": trow["topic_label"],
        "top_keywords": " ".join(w for w, _ in topic_model.get_topic(tid)),
        "bertscore_f1": bertscore_per_topic.get(tid, None),
        "coherence": coherence_scores.get(tid, None),
        "diversity": overall_diversity,
        "n_docs": int(trow["Count"]),
    })

results_df = pd.DataFrame(rows)
RESULTS_PATH = OUTPUT_DIR / "evaluation_results.csv"
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\nSaved evaluation_results.csv ({len(results_df)} rows): {RESULTS_PATH}")
print("\nSample:")
print(results_df[["topic_label", "bertscore_f1", "coherence", "diversity", "n_docs"]].head(5).to_string())

print(f"\n=== Summary ===")
print(f"Topics evaluated:         {len(results_df)}")
print(f"Mean BERTScore F1:        {results_df['bertscore_f1'].dropna().mean():.4f}")
print(f"Mean Coherence:      {results_df['coherence'].dropna().mean():.4f}")
print(f"Overall Topic Diversity:  {overall_diversity}")