import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import itertools
from pathlib import Path
import warnings
import math
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
CSV_PATH = BASE_DIR / "data" / "processed" / "filtered_corpus.csv"

# 1. Load embeddings first (with guard)
if not EMBEDDINGS_PATH.exists():
    print("ERROR: embeddings.npy not found! Run model.py once to cache embeddings first.")
    exit()

print("Loading cached embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
print(f"Loaded {len(embeddings)} embeddings.")

# 2. Then load chunks
print("Loading chunks from CSV...")
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(subset=['text', 'ticker', 'year'])
chunks = df["text"].tolist()
print(f"Loaded {len(chunks)} chunks.")

# 3. Now both exist, safe to compare
assert len(chunks) == len(embeddings), (
    f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
)

print("Data aligned. Starting Grid Search...\n")

# ==========================================
# 2. CUSTOM NPMI COHERENCE FUNCTION (Pure Python)
# ==========================================
def calculate_corpus_npmi(chunks, topic_words):
    """
    Calculates Normalized Pointwise Mutual Information (NPMI) for topic words.
    Scores range from -1 (never co-occur) to +1 (always co-occur).
    """
    # Flatten all top words to build a localized, fast vocabulary
    vocab = list(set([word for topic in topic_words for word in topic]))
    if not vocab:
        return -1.0
    
    # Create a binary document-term matrix (1 if word is in chunk, 0 if not)
    vectorizer = CountVectorizer(vocabulary=vocab, binary=True, stop_words='english')
    X = vectorizer.fit_transform(chunks).toarray()
    word_to_idx = vectorizer.vocabulary_
    N = X.shape[0] # Total chunks (14,000)
    
    # Calculate independent probabilities for every word
    p_word = X.sum(axis=0) / N
    
    total_npmi = 0
    valid_topics_count = 0
    
    for words in topic_words:
        topic_score = 0
        pairs_count = 0
        
        # Check every unique pair of words in the top words list
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                if w1 not in word_to_idx or w2 not in word_to_idx:
                    continue
                    
                idx1, idx2 = word_to_idx[w1], word_to_idx[w2]
                
                # Calculate joint occurrence (how many chunks contain both words)
                joint_count = np.sum((X[:, idx1] == 1) & (X[:, idx2] == 1))
                
                if joint_count == 0:
                    npmi = -1.0 # Minimum possible coherence score
                else:
                    p_joint = joint_count / N
                    p1 = p_word[idx1]
                    p2 = p_word[idx2]
                    
                    # Classic NPMI Formula
                    pmi = math.log(p_joint / (p1 * p2))
                    npmi = pmi / (-math.log(p_joint))
                    
                topic_score += npmi
                pairs_count += 1
                
        if pairs_count > 0:
            total_npmi += (topic_score / pairs_count)
            valid_topics_count += 1
            
    return total_npmi / valid_topics_count if valid_topics_count > 0 else -1.0

def get_topic_words(chunks, labels, top_n=10):
    """Extracts top N words for each HDBSCAN cluster using c-TF-IDF principles."""
    df = pd.DataFrame({"text": chunks, "topic": labels})
    topic_words = []
    
    for topic in sorted(df['topic'].unique()):
        if topic == -1: 
            continue # Skip noise
            
        topic_docs = df[df['topic'] == topic]['text'].values
        vectorizer = CountVectorizer(stop_words='english')
        try:
            X = vectorizer.fit_transform(topic_docs)
            words = vectorizer.get_feature_names_out()
            freqs = X.sum(axis=0).A1
            top_indices = freqs.argsort()[-top_n:][::-1]
            topic_words.append([words[i] for i in top_indices])
        except ValueError:
            pass # Handle empty text edge cases
            
    return topic_words

# ==========================================
# 3. RUN THE GRID SEARCH
# ==========================================
n_neighbors_list = [25, 30, 35]
min_cluster_size_list = [40, 50, 60, 70]
min_samples_list = [5, 10, None]

results = []

print(f"\n{'n_neighbors':<12} | {'min_cluster':<12} | {'min_samples':<12} | {'Topics':<8} | {'Noise %':<8} | {'NPMI Coherence':<15}")
print("-" * 80)

for nn, mcs, ms in itertools.product(n_neighbors_list, min_cluster_size_list, min_samples_list):

    reduced_emb = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom').fit(reduced_emb)
    labels = clusterer.labels_

    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = list(labels).count(-1) / len(labels)

    npmi_score = -1.0

    if n_topics > 1:
        topic_words = get_topic_words(chunks, labels, top_n=10)
        if topic_words:
            npmi_score = calculate_corpus_npmi(chunks, topic_words)

    results.append({
        "n_neighbors": nn,
        "min_cluster_size": mcs,
        "min_samples": str(ms),  # store as string so None displays cleanly
        "topics": n_topics,
        "noise_pct": noise_ratio * 100,
        "coherence": npmi_score
    })

    print(f"{nn:<12} | {mcs:<12} | {str(ms):<12} | {n_topics:<8} | {noise_ratio*100:<7.1f}% | {npmi_score:<15.4f}")

# ==========================================
# 4. HIGHLIGHT THE OPTIMAL COMBINATION
# ==========================================
df_results = pd.DataFrame(results)

valid_setups = df_results[
    (df_results['noise_pct'] <= 50.0) &
    (df_results['topics'] >= 10) &
    (df_results['topics'] <= 80)
]

if not valid_setups.empty:
    best_setup = valid_setups.sort_values(by="coherence", ascending=False).iloc[0]
    print("\n" + "="*50)
    print("🏆 OPTIMAL HYPERPARAMETERS FOUND 🏆")
    print("="*50)
    print(f"UMAP n_neighbors        : {int(best_setup['n_neighbors'])}")
    print(f"HDBSCAN min_cluster_size: {int(best_setup['min_cluster_size'])}")
    print(f"HDBSCAN min_samples     : {best_setup['min_samples']}")
    print(f"Resulting Topics        : {int(best_setup['topics'])}")
    print(f"Resulting Noise         : {best_setup['noise_pct']:.1f}%")
    print(f"NPMI Coherence Score    : {best_setup['coherence']:.4f}")
else:
    print("\n⚠️ No setup perfectly matched the target constraints. Review raw logs.")

df_results.to_csv(BASE_DIR / "outputs" / "grid_search_results.csv", index=False)
print("\nFull results saved to outputs/grid_search_results.csv")