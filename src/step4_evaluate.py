import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from bert_score import score as bert_score
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "step4_corpus_sample_150_manual.csv"
KEYWORD_PATH = BASE_DIR / "data" / "step2_keyword_dictionary.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

EVALUATION_SET_PATH = OUTPUT_DIR / "step4_evaluation_set.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load data and get topic assignments
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

# Topic label map: topic_id -> cleaned topic name
def clean_topic_name(name: str) -> str:
    parts = name.split("_")
    # drop the leading topic_id number
    parts = [p for p in parts if not p.isdigit()]
    return " ".join(parts).title()

topic_info_full = topic_model.get_topic_info()
topic_label_map = {
    row["Topic"]: clean_topic_name(row["Name"])
    for _, row in topic_info_full.iterrows()
    if row["Topic"] != -1
}
df["predicted_topic_label"] = df["predicted_topic_id"].map(topic_label_map).fillna("Noise / Unassigned")

# Step 2: Split multi-label rows into single-label rows
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

# Step 3: BERTScore: topic keywords vs manual_label keywords
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
_, _, F1_all = bert_score(cands, refs, lang="en", verbose=True) #we do not calculate precision and recall

df_exploded["bertscore_f1"] = [round(float(f), 4) for f in F1_all]

print(f"\nOverall BERTScore F1:")
print(f"  Mean:   {df_exploded['bertscore_f1'].mean():.4f}")
print(f"  Std:    {df_exploded['bertscore_f1'].std():.4f}")
print(f"  Min:    {df_exploded['bertscore_f1'].min():.4f}")
print(f"  Max:    {df_exploded['bertscore_f1'].max():.4f}")

print(f"\nBERTScore F1 per manual_label:")
print(df_exploded.groupby("manual_label")["bertscore_f1"].mean().sort_values(ascending=False).to_string())

df_exploded.to_csv(EVALUATION_SET_PATH, index=False)
print(f"\nUpdated step4_evaluation_set.csv with bertscore_f1: {EVALUATION_SET_PATH}")

# Step 4: Build step4_evaluation_results.csv 
# BERTScore F1 per topic
bertscore_grouped = (
    df_exploded[df_exploded["predicted_topic_id"] != -1]
    .groupby("predicted_topic_id")["bertscore_f1"]
)
bertscore_per_topic = bertscore_grouped.mean().round(4)
n_eval_chunks_per_topic = bertscore_grouped.count()

# topic_info_full has columns: Topic, Count, Name, Representation, ...
topic_info = topic_info_full[topic_info_full["Topic"] != -1].copy()
topic_info["topic_label"] = topic_info["Topic"].map(topic_label_map)

# Assemble final results table
rows = []
for _, trow in topic_info.iterrows():
    tid = trow["Topic"]
    rows.append({
        "topic_id": tid,
        "topic_label": trow["topic_label"],
        "top_keywords": " ".join(w for w, _ in topic_model.get_topic(tid)),
        "bertscore_f1": bertscore_per_topic.get(tid, None),
        "n_eval_chunks": int(n_eval_chunks_per_topic.get(tid, 0)),
    })

results_df = pd.DataFrame(rows)
RESULTS_PATH = OUTPUT_DIR / "step4_evaluation_results.csv"
results_df.to_csv(RESULTS_PATH, index=False)

print(f"\nSaved step4_evaluation_results.csv ({len(results_df)} rows): {RESULTS_PATH}")
print("\nSample:")
print(results_df[["topic_label", "bertscore_f1", "n_eval_chunks"]].head(5).to_string())

print(f"\n=== Summary ===")
print(f"Topics evaluated:         {len(results_df)}")
print(f"Mean BERTScore F1:        {results_df['bertscore_f1'].dropna().mean():.4f}")