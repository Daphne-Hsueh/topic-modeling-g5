import pandas as pd
import ast

df = pd.read_csv("data/processed/filtered_corpus.csv")

df["primary_cat"] = df["matched_categories"].apply(
    lambda x: ast.literal_eval(x)[0] if pd.notna(x) else "Unknown"
)
df["size_bin"] = pd.cut(df["n_words"], bins=3, labels=["small", "medium", "large"])
df["stratum"] = df["primary_cat"] + "_" + df["size_bin"].astype(str)

sampled = (
    df.groupby("stratum", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), max(1, int(150 * len(g) / len(df)))), random_state=42))
)
if len(sampled) < 150:
    remaining = df[~df.index.isin(sampled.index)]
    topup = remaining.sample(150 - len(sampled), random_state=42)
    sampled = pd.concat([sampled, topup])

sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

cols = ["cik", "ticker", "company", "sector", "year", "chunk_id", "n_words", "text"]
sampled[cols].to_csv("data/processed/corpus_sample_150.csv", index=False)
print(f"Saved {len(sampled)} rows")