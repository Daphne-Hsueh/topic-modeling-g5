import numpy as np
import pandas as pd
import umap
import hdbscan
import itertools
from pathlib import Path
import warnings
import math
import json
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "step3_embeddings.npy"
CSV_PATH = BASE_DIR / "data" / "processed" / "step2_filtered_corpus.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not EMBEDDINGS_PATH.exists():
    print("ERROR: step3_embeddings.npy not found! Run step3_model.py once to cache embeddings first.")
    exit()

print("Loading cached embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading chunks from CSV...")
df = pd.read_csv(CSV_PATH)

# Drop duplicates to prevent repetitive boilerplate from skewing the topic distributions
df = df.drop_duplicates(subset=['text', 'ticker', 'year'])
chunks = df["text"].tolist()

assert len(chunks) == len(embeddings), f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
print("Data aligned. Starting Grid Search...\n")

# Dictionary to store UMAP projections so we don't recalculate them for every HDBSCAN tweak
umap_cache = {}

def calculate_corpus_npmi(chunks, topic_words):
    """
    Calculates the Normalized Pointwise Mutual Information (NPMI).
    This measures if the top words defining a topic actually appear 
    together in real sentences within our 10-K text chunks.
    Scores range from -1.0 (never occur together) to 1.0 (always occur together).
    """
    # Create a unique list of all top words across all discovered topics
    vocab = list(set([word for topic in topic_words for word in topic]))
    if not vocab:
        return -1.0
    
    # Create a binary matrix: 1 if the word is in the chunk, 0 if it isn't
    vectorizer = CountVectorizer(vocabulary=vocab, binary=True, stop_words='english')
    X = vectorizer.fit_transform(chunks).toarray()
    word_to_idx = vectorizer.vocabulary_
    N = X.shape[0] # Total number of documents
    
    # Calculate the baseline probability of each word appearing in the corpus
    p_word = X.sum(axis=0) / N
    total_npmi = 0
    valid_topics_count = 0
    
    # Evaluate each topic one by one
    for words in topic_words:
        topic_score = 0
        pairs_count = 0
        
        # Compare every word in the topic to every other word in that same topic
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                
                # Skip words that got filtered out
                if w1 not in word_to_idx or w2 not in word_to_idx:
                    continue
                    
                idx1, idx2 = word_to_idx[w1], word_to_idx[w2]
                
                # Count how many chunks contain BOTH words simultaneously
                joint_count = np.sum((X[:, idx1] == 1) & (X[:, idx2] == 1))
                
                # Calculate the NPMI score for this specific word pair
                if joint_count == 0:
                    npmi = -1.0  
                elif joint_count >= N:
                    npmi = 1.0   
                else:
                    p_joint = joint_count / N
                    p1 = p_word[idx1]
                    p2 = p_word[idx2]
                    pmi = math.log(p_joint / (p1 * p2))
                    npmi = pmi / (-math.log(p_joint))
                    
                topic_score += npmi
                pairs_count += 1
                
        # Average the scores for this topic
        if pairs_count > 0:
            total_npmi += (topic_score / pairs_count)
            valid_topics_count += 1
            
    # Return the mean NPMI across all valid topics
    return total_npmi / valid_topics_count if valid_topics_count > 0 else -1.0

def get_topic_words(chunks, labels, top_n=10):
    """
    Extracts the top N most frequent words for each discovered topic.
    This acts as a simplified c-TF-IDF just for the sake of the grid search evaluation.
    """
    df = pd.DataFrame({"text": chunks, "topic": labels})
    topic_words = []
    
    for topic in sorted(df['topic'].unique()):
        if topic == -1: # Ignore the noise cluster
            continue 
            
        # Isolate documents belonging only to this topic
        topic_docs = df[df['topic'] == topic]['text'].values
        vectorizer = CountVectorizer(stop_words='english')
        
        try:
            # Count word frequencies within this specific topic
            X = vectorizer.fit_transform(topic_docs)
            words = vectorizer.get_feature_names_out()
            freqs = X.sum(axis=0).A1
            
            # Grab the indices of the top_n most frequent words
            top_indices = freqs.argsort()[-top_n:][::-1]
            topic_words.append([words[i] for i in top_indices])
        except ValueError:
            pass 
            
    return topic_words

n_neighbors_list = [25, 30, 35]
min_cluster_size_list = [40, 50, 60, 70]
min_samples_list = [5, 10, None]

results = []
print(f"\n{'n_neighbors':<12} | {'min_cluster':<12} | {'min_samples':<12} | {'Topics':<8} | {'Noise %':<8} | {'NPMI Coherence':<15}")
print("-" * 80)

# Iterate through every possible combination of the parameters above
for nn, mcs, ms in itertools.product(n_neighbors_list, min_cluster_size_list, min_samples_list):
    
    # Step A: Dimensionality Reduction
    # Cache the UMAP projection. UMAP is slow, so we only run it once per n_neighbor value.
    if nn not in umap_cache:
        umap_cache[nn] = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    reduced_emb = umap_cache[nn]
    
    # Step B: Clustering
    # Test different clustering strictness levels on the reduced data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom').fit(reduced_emb)
    labels = clusterer.labels_

    # Calculate basic metrics: How many topics were found, and how much data was thrown out?
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = list(labels).count(-1) / len(labels)

    npmi_score = -1.0

    # Step C: Evaluation
    # Only calculate the expensive NPMI score if the model actually found valid topics
    if n_topics > 1:
        topic_words = get_topic_words(chunks, labels, top_n=10)
        if topic_words:
            npmi_score = calculate_corpus_npmi(chunks, topic_words)

    # Store the results for this run
    results.append({
        "n_neighbors": nn,
        "min_cluster_size": mcs,
        "min_samples": ms,
        "topics": n_topics,
        "noise_pct": noise_ratio * 100,
        "coherence": npmi_score
    })

    print(f"{nn:<12} | {mcs:<12} | {str(ms):<12} | {n_topics:<8} | {noise_ratio*100:<7.1f}% | {npmi_score:<15.4f}")
df_results = pd.DataFrame(results)
valid_setups = df_results[
    (df_results['noise_pct'] <= 50.0) &
    (df_results['topics'] >= 10) &
    (df_results['topics'] <= 80)
]

if not valid_setups.empty:
    best_setup = valid_setups.sort_values(by="coherence", ascending=False).iloc[0]
    
    # Save the best parameters to a JSON file to prevent hardcoding
    best_params = {
        "n_neighbors": int(best_setup['n_neighbors']),
        "min_cluster_size": int(best_setup['min_cluster_size']),
        # Handle the None value safely for JSON
        "min_samples": int(best_setup['min_samples']) if pd.notna(best_setup['min_samples']) else None
    }
    
    with open(OUTPUT_DIR / "step3_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    print("\n" + "="*50)
    print("OPTIMAL HYPERPARAMETERS FOUND AND SAVED")
    print("="*50)
    print(f"UMAP n_neighbors        : {best_params['n_neighbors']}")
    print(f"HDBSCAN min_cluster_size: {best_params['min_cluster_size']}")
    print(f"HDBSCAN min_samples     : {best_params['min_samples']}")
    print(f"Resulting Topics        : {int(best_setup['topics'])}")
    print(f"Resulting Noise         : {best_setup['noise_pct']:.1f}%")
else:
    print("\nWARNING: No setup perfectly matched the target constraints.")

df_results.to_csv(OUTPUT_DIR / "step3_grid_search_results.csv", index=False)