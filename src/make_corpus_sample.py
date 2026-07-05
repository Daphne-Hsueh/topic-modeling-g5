"""Draw the stratified 150-chunk evaluation sample.

Writes an UNLABELED sample to data/processed/step4_corpus_sample_150.csv.
The labels live in data/step4_corpus_sample_150_manual.csv, which was labeled
by hand from an earlier run of this script — never overwrite that file.
"""
import pandas as pd
import ast

df = pd.read_csv("data/processed/step2_filtered_corpus.csv")

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
sampled[cols].to_csv("data/processed/step4_corpus_sample_150.csv", index=False)
print(f"Saved {len(sampled)} rows")