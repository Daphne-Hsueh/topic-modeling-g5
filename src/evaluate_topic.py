import numpy as np
import pandas as pd
import warnings
import math
warnings.filterwarnings('ignore')

from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from sklearn.feature_extraction.text import CountVectorizer

# ==========================================
# 0. PATHS
# ==========================================
BASE_DIR    = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "processed" / "filtered_corpus.csv"
GOLDEN_PATH = BASE_DIR / "data" / "corpus_sample_150.csv"
MODEL_PATH  = BASE_DIR / "outputs" / "models"
OUTPUT_PATH = BASE_DIR / "outputs" / "evaluation_results.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading full corpus...")
corpus_df = pd.read_csv(CORPUS_PATH)
corpus_df = corpus_df.drop_duplicates(subset=['text', 'ticker', 'year'])
all_docs  = corpus_df["text"].tolist()
print(f"Corpus: {len(all_docs)} chunks")

print("Loading golden set (150 chunks)...")
golden_df = pd.read_csv(GOLDEN_PATH)
golden_df['labels_list'] = golden_df['manual_label'].apply(
    lambda row: [l.strip() for l in str(row).split(';') if l.strip()]
)

# All 14 manual categories from the golden set
all_categories = sorted(set(
    label
    for labels in golden_df['labels_list']
    for label in labels
))
print(f"Manual categories ({len(all_categories)}): {all_categories}\n")

# ==========================================
# 2. LOAD BERTOPIC MODEL
# ==========================================
print("Loading BERTopic model...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
try:
    model = BERTopic.load(str(MODEL_PATH), embedding_model=embedding_model)
except Exception as e:
    print(f"Model load error: {e}")
    exit()

topic_info   = model.get_topic_info()
valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
print(f"Model has {len(valid_topics)} valid topics.\n")

# ==========================================
# 3. GET TOPIC KEYWORDS
# ==========================================
topic_keywords_str  = {}   # topic_id -> "word1 word2 word3 word4 word5"
topic_keywords_list = {}   # topic_id -> [word1, ..., word10]

for topic_id in valid_topics:
    words = [w for w, _ in model.get_topic(topic_id)[:10]]
    topic_keywords_list[topic_id] = words
    topic_keywords_str[topic_id]  = " ".join(words[:5])

# ==========================================
# 4. BERTSCORE — BEST MATCHING MANUAL LABEL
# ==========================================
print("Running BERTScore: topics vs manual categories...")

candidates = []
references = []
pair_map   = []

for topic_id in valid_topics:
    kw = topic_keywords_str[topic_id]
    for category in all_categories:
        candidates.append(kw)
        references.append(category)
        pair_map.append((topic_id, category))

_, _, F1_all = bert_score(candidates, references, lang="en", verbose=True)

topic_best_label = {}
topic_best_f1    = {}

for i, (topic_id, category) in enumerate(pair_map):
    f1_val = F1_all[i].item()
    if topic_id not in topic_best_f1 or f1_val > topic_best_f1[topic_id]:
        topic_best_f1[topic_id]    = f1_val
        topic_best_label[topic_id] = category

print("BERTScore done.\n")

# ==========================================
# 5. NPMI COHERENCE (replaces gensim Cv)
# ==========================================
# Reused from tune_hyperparameters.py
# Measures how often a topic's top words co-occur in the same chunk.
# Range: -1 (never co-occur) to +1 (always co-occur)
print("Computing NPMI coherence per topic...")

def compute_npmi_for_topic(docs, words, top_n=10):
    words = words[:top_n]
    vocab = list(set(words))
    if len(vocab) < 2:
        return 0.0

    vectorizer = CountVectorizer(vocabulary=vocab, binary=True, stop_words='english')
    try:
        X = vectorizer.fit_transform(docs).toarray()
    except Exception:
        return 0.0

    word_to_idx = vectorizer.vocabulary_
    N           = X.shape[0]
    p_word      = X.sum(axis=0) / N

    topic_score = 0
    pairs_count = 0

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            if w1 not in word_to_idx or w2 not in word_to_idx:
                continue
            idx1, idx2  = word_to_idx[w1], word_to_idx[w2]
            joint_count = np.sum((X[:, idx1] == 1) & (X[:, idx2] == 1))

            if joint_count == 0:
                npmi = -1.0
            else:
                p_joint = joint_count / N
                p1, p2  = p_word[idx1], p_word[idx2]
                pmi     = math.log(p_joint / (p1 * p2))
                npmi    = pmi / (-math.log(p_joint))

            topic_score += npmi
            pairs_count += 1

    return round(topic_score / pairs_count, 4) if pairs_count > 0 else 0.0

topic_coherence = {}
for topic_id in valid_topics:
    words = topic_keywords_list[topic_id]
    topic_coherence[topic_id] = compute_npmi_for_topic(all_docs, words)

print("Coherence done.\n")

# ==========================================
# 6. TOPIC DIVERSITY (per topic)
# ==========================================
# Fraction of this topic's top-10 words that do NOT
# appear in any other topic's top-10 words.
# 1.0 = fully unique (very distinct)
# 0.0 = all words shared with other topics (generic)
print("Computing topic diversity...")

all_words_per_topic = {tid: set(words) for tid, words in topic_keywords_list.items()}

topic_diversity = {}
for topic_id in valid_topics:
    this_words  = all_words_per_topic[topic_id]
    other_words = set(
        w for tid, words in all_words_per_topic.items()
        if tid != topic_id
        for w in words
    )
    unique_words              = this_words - other_words
    topic_diversity[topic_id] = round(len(unique_words) / len(this_words), 4) if this_words else 0.0

print("Diversity done.\n")

# ==========================================
# 7. N_DOCS PER TOPIC
# ==========================================
topic_counts = topic_info.set_index('Topic')['Count'].to_dict()

# ==========================================
# 8. ASSEMBLE RESULTS TABLE
# ==========================================
rows = []
for topic_id in valid_topics:
    rows.append({
        "topic_id"       : topic_id,
        "topic_label"    : topic_best_label.get(topic_id, "Unknown"),
        "topic_keywords" : topic_keywords_str.get(topic_id, ""),
        "bertscore_f1"   : round(topic_best_f1.get(topic_id, 0.0), 4),
        "coherence_npmi" : topic_coherence.get(topic_id, 0.0),
        "diversity"      : topic_diversity.get(topic_id, 0.0),
        "n_docs"         : topic_counts.get(topic_id, 0),
    })

results_df = pd.DataFrame(rows).sort_values("n_docs", ascending=False)

# ==========================================
# 9. PRINT RESULTS
# ==========================================
print("=" * 90)
print("TOPIC-LEVEL EVALUATION RESULTS")
print("=" * 90)
print(f"{'topic_label':<35} | {'bertscore_f1':<14} | {'coherence_npmi':<16} | {'diversity':<10} | n_docs")
print("-" * 90)
for _, row in results_df.iterrows():
    print(
        f"{row['topic_label']:<35} | "
        f"{row['bertscore_f1']:<14.4f} | "
        f"{row['coherence_npmi']:<16.4f} | "
        f"{row['diversity']:<10.4f} | "
        f"{int(row['n_docs'])}"
    )

print("=" * 90)
print(f"\n OVERALL SUMMARY")
print(f"  Mean BERTScore F1    : {results_df['bertscore_f1'].mean():.4f}")
print(f"  Mean Coherence NPMI  : {results_df['coherence_npmi'].mean():.4f}")
print(f"  Mean Diversity       : {results_df['diversity'].mean():.4f}")
print(f"  Total valid topics   : {len(results_df)}")
print("=" * 90)

# ==========================================
# 10. SAVE
# ==========================================
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Results saved to: {OUTPUT_PATH}")