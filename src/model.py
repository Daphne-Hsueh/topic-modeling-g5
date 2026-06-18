import ast
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan

# 2. Dynamically build paths
CSV_PATH = BASE_DIR / "data" / "processed" / "filtered_corpus.csv"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "corpus_with_ai_topics.csv"

print(f"Loading data from: {CSV_PATH}")

# 3. Load the dataframe
df = pd.read_csv(CSV_PATH)
df["matched_categories"] = df["matched_categories"].apply(ast.literal_eval)

docs = df["text"].tolist()
timestamps = df["year"].tolist()

# 4. Configure Models
embedding_model = SentenceTransformer('all-mpnet-base-v2')
umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Initialize the vectorizer to remove standard English stop words
vectorizer_model = CountVectorizer(stop_words="english")

# 5. Embedding Caching Logic
if EMBEDDINGS_PATH.exists():
    print("\nLoading pre-calculated embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    print("\nCalculating embeddings for the first time...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Embeddings saved to cache.")

# 6. Initialize and Fit BERTopic
print("\nFitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
df['bertopic_id'] = topics

topic_info = topic_model.get_topic_info()
print("\nTop 10 Discovered Topics (Cleaned):")
print(topic_info.head(10))

# 7. Dynamic Topic Modeling (Temporal Coherence)
print("\nCalculating topics over time...")
topics_over_time = topic_model.topics_over_time(docs, timestamps)

# Generate the interactive timeline chart
fig = topic_model.visualize_topics_over_time(topics_over_time)
fig.show()

# 8. Sanity Check
print("\n" + "="*40)
print("DATA PIPELINE SANITY CHECK")
print("="*40)

print(f"Total chunks loaded: {len(df)}")
print(f"Unique companies (tickers) present: {df['ticker'].nunique()}")

if 'bertopic_id' in df.columns:
    noise_chunks = len(df[df['bertopic_id'] == -1])
    clustered_chunks = len(df[df['bertopic_id'] != -1])
    total_valid_topics = df['bertopic_id'].nunique() - 1 
    
    print("\n--- CLUSTERING RESULTS ---")
    print(f"Total valid topics discovered: {total_valid_topics}")
    print(f"Successfully clustered chunks: {clustered_chunks}")
    print(f"Noise chunks (Topic -1): {noise_chunks} ({(noise_chunks/len(df))*100:.1f}%)")
    
    print("\n--- TOP LARGEST TOPICS (Including Noise) ---")
    print(df['bertopic_id'].value_counts().head(6).to_string())
print("="*40 + "\n")

# 9. Save final output
df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")