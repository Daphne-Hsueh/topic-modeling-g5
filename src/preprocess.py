"""Preprocessing: turn the document corpus into model-ready chunks.

Input : data/corpus.csv             (one row per filing: cik, ticker, company, year, text)
        data/selected_companies.csv  (for the GICS sector label)
Output: data/processed/chunks_clean.csv  (one row per chunk, with company metadata)

Why chunk? An Item 1A section covers 20–40 distinct risks. Embedding a whole
section as one vector blurs them together, so we split each section into
paragraph-sized chunks — roughly one risk factor each — which is the right unit
for BERTopic.

Two text columns are produced:
  * `text`       — the natural-language chunk. FEED THIS TO BERTopic; its
                   sentence-transformer needs real grammar and context.
  * `text_clean` — lower-cased, de-punctuated, stop-words removed. Only for a
                   bag-of-words baseline (LDA/NMF) or a custom CountVectorizer.
                   Do NOT embed it with BERTopic.

Run as a script (`python -m src.preprocess`) or import `build_chunks` from a notebook.
"""
from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
CORPUS_CSV  = DATA_DIR / "corpus.csv"
COMPANY_CSV = DATA_DIR / "selected_companies.csv"
OUTPUT_CSV  = DATA_DIR / "processed" / "chunks_clean.csv"

# ── tunables ─────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS = 400   # split paragraphs longer than this on sentence boundaries
MIN_CHUNK_WORDS = 20    # drop fragments shorter than this (headers, stray lines)

STOP_WORDS = set(stopwords.words("english"))

# Generic risk-disclaimer phrasing that appears in nearly every filing and carries
# no topical signal — removed from the bag-of-words column so it can't dominate.
BOILERPLATE = [
    "there can be no assurance", "forward-looking statements", "material adverse effect",
    "we undertake no obligation to update", "actual results may differ materially",
    "within the meaning of the private securities litigation reform act",
    "risk factors that could cause actual results to differ", "you should carefully consider",
    "the following risks could materially", "in addition to the other information set forth",
    "these risks are not the only risks we face", "among other things", "including but not limited to",
]
_BOILERPLATE_RE = re.compile("|".join(re.escape(p) for p in BOILERPLATE), re.IGNORECASE)
_PAGE_NUM_RE    = re.compile(r"^\s*(?:page\s+\d+\s+of\s+\d+|\d+)\s*$", re.IGNORECASE)
_PUNCT_RE       = re.compile(r"[^\w\s]")


# ── Step 1: light clean (keep natural language) ───────────────────────────────
def light_clean(raw: str) -> str:
    """Remove page numbers and short ALL-CAPS header lines; normalise blank lines."""
    kept = []
    for line in raw.splitlines():
        if _PAGE_NUM_RE.match(line):
            continue
        stripped = line.strip()
        if stripped and stripped == stripped.upper() and len(stripped.split()) < 5:
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


# ── Step 2: split into chunks ─────────────────────────────────────────────────
def split_chunks(text: str) -> list[str]:
    """Split into paragraph-sized chunks; over-long paragraphs are packed into
    sub-chunks on sentence boundaries so no chunk exceeds MAX_CHUNK_WORDS."""
    chunks: list[str] = []
    for para in re.split(r"\n\n+", text):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= MAX_CHUNK_WORDS:
            if len(words) >= MIN_CHUNK_WORDS:
                chunks.append(para)
            continue
        current, count = [], 0
        for sent in sent_tokenize(para):
            sw = len(sent.split())
            if count + sw > MAX_CHUNK_WORDS and current:
                if count >= MIN_CHUNK_WORDS:
                    chunks.append(" ".join(current))
                current, count = [sent], sw
            else:
                current.append(sent)
                count += sw
        if current and count >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(current))
    return chunks


# ── Step 3: deep clean (bag-of-words baseline only) ───────────────────────────
def deep_clean(text: str) -> str:
    text = _BOILERPLATE_RE.sub(" ", text.lower())
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(t for t in text.split() if t not in STOP_WORDS)


# ── orchestration ─────────────────────────────────────────────────────────────
def build_chunks(corpus: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    """corpus (cik, ticker, company, year, text) -> one row per chunk with metadata."""
    sector_map = sector_map or {}
    rows = []
    for _, doc in corpus.iterrows():
        for idx, chunk in enumerate(split_chunks(light_clean(str(doc["text"])))):
            rows.append({
                "cik":        doc["cik"],
                "ticker":     doc.get("ticker"),
                "company":    doc.get("company"),
                "sector":     sector_map.get(doc["cik"]),
                "year":       int(doc["year"]),
                "chunk_id":   idx,
                "n_words":    len(chunk.split()),
                "text":       chunk,             # -> BERTopic
                "text_clean": deep_clean(chunk),  # -> LDA/NMF baseline
            })
    return pd.DataFrame(rows, columns=[
        "cik", "ticker", "company", "sector", "year", "chunk_id", "n_words", "text", "text_clean"
    ])


def load_inputs() -> tuple[pd.DataFrame, dict[str, str]]:
    if not CORPUS_CSV.exists():
        raise FileNotFoundError(f"{CORPUS_CSV} not found — run 02_extraction_pipeline.ipynb first.")
    corpus = pd.read_csv(CORPUS_CSV, dtype={"cik": str})
    corpus["cik"] = corpus["cik"].str.zfill(10)
    companies = pd.read_csv(COMPANY_CSV, dtype={"cik": str})
    companies["cik"] = companies["cik"].str.zfill(10)
    return corpus, dict(zip(companies["cik"], companies["sector"]))


def main() -> None:
    corpus, sector_map = load_inputs()
    chunks = build_chunks(corpus, sector_map)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(OUTPUT_CSV, index=False)

    print(f"Filings in     : {len(corpus)}")
    print(f"Chunks out     : {len(chunks)}")
    print(f"Companies      : {chunks['cik'].nunique()}")
    print(f"Years          : {chunks['year'].min()}-{chunks['year'].max()}")
    print(f"Mean chunks/doc: {len(chunks) / max(len(corpus), 1):.1f}")
    print(f"Saved          : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
