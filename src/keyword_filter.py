"""Keyword filtering step.

Reads the cleaned chunk corpus, matches each chunk's raw `text` against the
multi-word keyword lists in keyword_categories.json, and keeps only the chunks
that match at least one reputation-risk category. A `matched_categories` column
records which categories each surviving chunk matched.

Output: data/processed/filtered_corpus.csv (chunks_clean.csv is never modified).
"""

import json
import re
from pathlib import Path

import pandas as pd

# Resolve paths relative to the repo root (this file lives in src/).
ROOT = Path(__file__).resolve().parents[1]
KEYWORDS_PATH = ROOT / "keyword_categories.json"
INPUT_PATH = ROOT / "data" / "processed" / "chunks_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "filtered_corpus.csv"


def load_categories(path: Path) -> dict[str, list[str]]:
    """Load category -> keyword list mapping from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compile_category_patterns(
    categories: dict[str, list[str]]
) -> dict[str, list[re.Pattern]]:
    """Compile each keyword into a case-insensitive, word-boundary-aware regex.

    Word boundaries (\\b) ensure that e.g. "bribe" does not match inside
    "bribed"... actually \\b allows trailing word chars only at the boundary;
    we anchor both ends so "bribe" matches "bribe" but not "bribery" as a
    substring of a longer token. Phrases are escaped so hyphens and other
    characters are treated literally rather than as regex metacharacters.
    """
    compiled: dict[str, list[re.Pattern]] = {}
    for category, keywords in categories.items():
        compiled[category] = [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in keywords
        ]
    return compiled


def match_categories(text: str, patterns: dict[str, list[re.Pattern]]) -> list[str]:
    """Return the list of category names whose keywords appear in `text`."""
    if not isinstance(text, str):
        return []
    matched = []
    for category, pats in patterns.items():
        if any(p.search(text) for p in pats):
            matched.append(category)
    return matched


def main() -> None:
    categories = load_categories(KEYWORDS_PATH)
    patterns = compile_category_patterns(categories)

    df = pd.read_csv(INPUT_PATH)
    total_in = len(df)

    # Match keyword phrases against the natural-language `text` column.
    df["matched_categories"] = df["text"].apply(lambda t: match_categories(t, patterns))

    filtered = df[df["matched_categories"].map(len) > 0].copy()
    total_out = len(filtered)

    filtered.to_csv(OUTPUT_PATH, index=False)

    # --- Verification output ---
    pct = (total_out / total_in * 100) if total_in else 0.0
    print(f"Chunks in:  {total_in}")
    print(f"Chunks out: {total_out}")
    print(f"Retained:   {pct:.1f}%")

    print("\nMatches per category:")
    for category in categories:
        count = filtered["matched_categories"].apply(lambda cats: category in cats).sum()
        flag = "  <-- ZERO HITS" if count == 0 else ""
        print(f"  {count:6d}  {category}{flag}")

    print("\nExample rows:")
    for _, row in filtered.head(2).iterrows():
        snippet = row["text"][:300].replace("\n", " ")
        print(f"\n  matched_categories: {row['matched_categories']}")
        print(f"  text: {snippet}...")

    in_cols = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()
    out_cols = filtered.columns.tolist()
    print(
        f"\nColumn check: input has {len(in_cols)} columns, "
        f"output has {len(out_cols)} columns "
        f"(diff = {len(out_cols) - len(in_cols)})"
    )
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
