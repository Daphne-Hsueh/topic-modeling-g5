import pandas as pd
import numpy as np
from bert_score import score as bert_score
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "corpus_sample_150.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models"
RESULTS_PATH = BASE_DIR / "outputs" / "evaluation_results.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find CSV at {CSV_PATH}")
    exit()

TEXT_COLUMN = 'text'
LABEL_COLUMN = 'manual_label'

df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("")

# Parse semicolon-separated multi-labels into a list per chunk
df['labels_list'] = df[LABEL_COLUMN].apply(
    lambda row: [l.strip() for l in str(row).split(';') if l.strip()]
)

print(f"Loaded {len(df)} chunks")
print(f"  Single-label chunks : {df['labels_list'].apply(len).eq(1).sum()}")
print(f"  Multi-label chunks  : {df['labels_list'].apply(len).gt(1).sum()}")

# ==========================================
# 2. LOAD MODEL AND ASSIGN TOPICS
# ==========================================
print("\nLoading BERTopic model...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')

try:
    model = BERTopic.load(str(MODEL_PATH), embedding_model=embedding_model)
except Exception as e:
    print(f"Model load error: {e}")
    exit()

docs = df[TEXT_COLUMN].tolist()
print("Assigning topics to chunks...")
topics, probs = model.transform(docs)
df['assigned_topic'] = topics

noise_count = sum(1 for t in topics if t == -1)
print(f"Done. Noise chunks (topic -1): {noise_count}/{len(df)} ({noise_count/len(df)*100:.1f}%)")

# ==========================================
# 3. BUILD TOPIC KEYWORD LOOKUP
# ==========================================
print("\nBuilding topic keyword lookup...")
topic_keyword_map = {}
for topic_id in set(topics):
    if topic_id == -1:
        topic_keyword_map[-1] = "unclassified noise"
    else:
        words = [word for word, _ in model.get_topic(topic_id)[:5]]
        topic_keyword_map[topic_id] = " ".join(words)

df['topic_keywords'] = df['assigned_topic'].map(topic_keyword_map)

# ==========================================
# 4. BUILD FLAT PAIRS FOR BERTSCORE
#
# Logic: for each chunk, pair its assigned topic keywords
# against EACH of its manual labels. BERTScore runs once
# on all pairs, then we take the best-matching label per chunk.
#
# This handles multi-label chunks fairly — the model only needs
# to semantically match ONE of the correct labels to score well.
# ==========================================
print("\nBuilding evaluation pairs...")
candidates  = []   # topic keywords (repeated per label)
references  = []   # manual label names
chunk_idxs  = []   # which chunk each pair belongs to

for idx, row in df.iterrows():
    kw     = row['topic_keywords']
    labels = row['labels_list']
    for label in labels:
        candidates.append(kw)
        references.append(label)
        chunk_idxs.append(idx)

print(f"Total (chunk, label) pairs to evaluate: {len(candidates)}")

# ==========================================
# 5. RUN BERTSCORE ONCE ON ALL PAIRS
# ==========================================
print("\nRunning BERTScore (this runs once for all pairs)...")
P_all, R_all, F1_all = bert_score(
    candidates,
    references,
    lang="en",
    verbose=True
)
print("BERTScore done.")

# ==========================================
# 6. AGGREGATE — BEST LABEL MATCH PER CHUNK
# ==========================================
# For multi-label chunks, take whichever label gave the highest F1
chunk_best_f1      = {}
chunk_best_label   = {}
chunk_best_p       = {}
chunk_best_r       = {}

for i, chunk_idx in enumerate(chunk_idxs):
    f1_val = F1_all[i].item()
    if chunk_idx not in chunk_best_f1 or f1_val > chunk_best_f1[chunk_idx]:
        chunk_best_f1[chunk_idx]    = f1_val
        chunk_best_label[chunk_idx] = references[i]
        chunk_best_p[chunk_idx]     = P_all[i].item()
        chunk_best_r[chunk_idx]     = R_all[i].item()

# ==========================================
# 7. PRINT CHUNK-LEVEL RESULTS
# ==========================================
print("\n" + "=" * 70)
print("--- CHUNK-LEVEL EVALUATION RESULTS ---")
print("=" * 70)
print(f"{'#':<4} | {'Topic':>5} | {'F1':>6} | {'Best Matched Label':<35} | Keywords")
print("-" * 70)

all_f1        = []
non_noise_f1  = []

for idx, row in df.iterrows():
    f1         = chunk_best_f1.get(idx, 0.0)
    best_label = chunk_best_label.get(idx, "N/A")
    topic_id   = row['assigned_topic']
    kw         = row['topic_keywords']

    all_f1.append(f1)
    if topic_id != -1:
        non_noise_f1.append(f1)

    # Flag noise chunks
    noise_flag = " ⚠️" if topic_id == -1 else ""
    print(f"{idx:<4} | {topic_id:>5}{noise_flag} | {f1:.4f} | {best_label:<35} | {kw}")

# ==========================================
# 8. OVERALL SUMMARY
# ==========================================
print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print(f"  Mean BERTScore F1 — all 150 chunks   : {np.mean(all_f1):.4f}")
print(f"  Mean BERTScore F1 — non-noise only   : {np.mean(non_noise_f1):.4f}")
print(f"  Noise chunks (topic = -1)             : {noise_count} ({noise_count/len(df)*100:.1f}%)")
print(f"  Chunks successfully clustered         : {len(df) - noise_count} ({(len(df)-noise_count)/len(df)*100:.1f}%)")

# ==========================================
# 9. PER-CATEGORY BREAKDOWN
#
# For each category, collect the BERTScore F1 from every
# (chunk, label) pair where that label appears, then average.
# This shows which risk categories the model captures best.
# ==========================================
print("\n" + "=" * 70)
print("PER-CATEGORY BREAKDOWN")
print("=" * 70)
print(f"{'Category':<35} | {'Mean F1':<8} | {'Chunks'}")
print("-" * 70)

category_scores = {}
for i, chunk_idx in enumerate(chunk_idxs):
    label  = references[i]
    f1_val = F1_all[i].item()
    if label not in category_scores:
        category_scores[label] = []
    category_scores[label].append(f1_val)

for label in sorted(category_scores.keys()):
    scores = category_scores[label]
    print(f"{label:<35} | {np.mean(scores):.4f}   | {len(scores)}")

# ==========================================
# 10. SAVE RESULTS TO CSV
# ==========================================
results_rows = []
for idx, row in df.iterrows():
    results_rows.append({
        "chunk_id"        : row.get("chunk_id", idx),
        "ticker"          : row.get("ticker", ""),
        "year"            : row.get("year", ""),
        "assigned_topic"  : row['assigned_topic'],
        "topic_keywords"  : row['topic_keywords'],
        "manual_labels"   : "; ".join(row['labels_list']),
        "best_matched_label" : chunk_best_label.get(idx, "N/A"),
        "precision"       : round(chunk_best_p.get(idx, 0.0), 4),
        "recall"          : round(chunk_best_r.get(idx, 0.0), 4),
        "f1"              : round(chunk_best_f1.get(idx, 0.0), 4),
    })

results_df = pd.DataFrame(results_rows)
results_df.to_csv(RESULTS_PATH, index=False)
print(f"\n✅ Full results saved to: {RESULTS_PATH}")
print("=" * 70)