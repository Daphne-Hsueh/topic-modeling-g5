import ast
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "step2_filtered_corpus.csv"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "step3_embeddings.npy"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "step3_corpus_with_bert_topics.csv"

MODEL_DIR  = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
PARAMS_PATH = OUTPUT_DIR / "step3_best_params.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Dynamic Parameters
if PARAMS_PATH.exists():
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"Loaded tuned parameters: {params}")
else:
    print("WARNING: step3_best_params.json not found. Falling back to default parameters.")
    params = {"n_neighbors": 15, "min_cluster_size": 80, "min_samples": 5}

# 2. Load Data
df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates(subset=['text', 'ticker', 'year'])
df["matched_categories"] = df["matched_categories"].apply(ast.literal_eval)

docs = df["text"].tolist()
timestamps = df["year"].tolist()

# 3. Configure Models Dynamically
embedding_model = SentenceTransformer('all-mpnet-base-v2')
umap_model = umap.UMAP(n_neighbors=params["n_neighbors"], n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], metric='euclidean', cluster_selection_method='eom', prediction_data=True)

vectorizer_model = CountVectorizer(stop_words="english")

# 4. Load or Calculate Embeddings
if EMBEDDINGS_PATH.exists():
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)

# 5. Fit BERTopic
print("\nFitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
df['bertopic_id'] = topics

print("\nCalculating topics over time...")
topics_over_time = topic_model.topics_over_time(docs, timestamps)
topics_over_time.to_csv(OUTPUT_DIR / "step3_topics_over_time.csv", index=False)

# 6. Save Data Output
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset saved to {OUTPUT_PATH}")

# 7. Save the Model Engine for Visualization Step
topic_model.save(str(MODEL_DIR), serialization="safetensors", save_ctfidf=True)
print("Model engine successfully saved to outputs/models/")