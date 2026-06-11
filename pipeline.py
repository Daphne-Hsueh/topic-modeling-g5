import re
import csv
import string
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── paths ──────────────────────────────────────────────────────────────────────
ITEM1A_DIR = Path("data/item1a")
OUTPUT_CSV = Path("data/chunks_clean.csv")

# ── constants ──────────────────────────────────────────────────────────────────
MAX_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 20

STOP_WORDS = set(stopwords.words("english"))

BOILERPLATE = [
    "there can be no assurance",
    "forward-looking statements",
    "material adverse effect",
    "we undertake no obligation to update",
    "actual results may differ materially",
    "within the meaning of the private securities litigation reform act",
    "risk factors that could cause actual results to differ",
    "you should carefully consider",
    "the following risks could materially",
    "in addition to the other information set forth",
    "these risks are not the only risks we face",
    "among other things",
    "including but not limited to",
]

# pre-compile for speed
_BOILERPLATE_RE = re.compile(
    "|".join(re.escape(p) for p in BOILERPLATE), re.IGNORECASE
)
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+\d+\s+of\s+\d+|\d+)\s*$", re.IGNORECASE)
_PUNCT_KEEP_PERIOD = str.maketrans("", "", string.punctuation.replace(".", ""))


# ── Step 1: light clean ────────────────────────────────────────────────────────
def light_clean(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        # drop page numbers
        if _PAGE_NUM_RE.match(line):
            continue
        # drop all-caps lines shorter than 5 words
        stripped = line.strip()
        if stripped and stripped == stripped.upper() and len(stripped.split()) < 5:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # collapse 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── Step 2: split into paragraphs / sentence chunks ───────────────────────────
def split_chunks(text: str) -> list[str]:
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= MAX_CHUNK_WORDS:
            if len(words) >= MIN_CHUNK_WORDS:
                chunks.append(para)
        else:
            # split large paragraph into sentences, then group into sub-chunks
            sentences = sent_tokenize(para)
            current, current_words = [], 0
            for sent in sentences:
                sw = len(sent.split())
                if current_words + sw > MAX_CHUNK_WORDS and current:
                    candidate = " ".join(current)
                    if len(candidate.split()) >= MIN_CHUNK_WORDS:
                        chunks.append(candidate)
                    current, current_words = [sent], sw
                else:
                    current.append(sent)
                    current_words += sw
            if current:
                candidate = " ".join(current)
                if len(candidate.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(candidate)
    return chunks


# ── Step 3: deep clean ─────────────────────────────────────────────────────────
def deep_clean(text: str) -> str:
    text = text.lower()
    text = _BOILERPLATE_RE.sub(" ", text)
    text = text.translate(_PUNCT_KEEP_PERIOD)
    tokens = [t for t in text.split() if t not in STOP_WORDS and t != "."]
    return " ".join(tokens)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    files = sorted(ITEM1A_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {ITEM1A_DIR}")
        return

    rows = []
    files_processed = 0

    for path in files:
        # parse cik and year from filename: <cik>_<year>.txt
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  Skipping unrecognised filename: {path.name}")
            continue
        cik, year = parts[0], parts[1]

        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = light_clean(raw)
        chunks = split_chunks(cleaned)

        for idx, chunk in enumerate(chunks):
            rows.append({
                "company_cik": cik,
                "year":        year,
                "chunk_id":    idx,
                "text":        chunk,
                "text_clean":  deep_clean(chunk),
            })

        files_processed += 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company_cik", "year", "chunk_id", "text", "text_clean"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Files processed : {files_processed}")
    print(f"Total chunks    : {len(rows)}")
    print(f"Output saved to : {OUTPUT_CSV}")

    # print 3 random rows
    import random
    sample = random.sample(rows, min(3, len(rows)))
    print("\n── 3 random rows ──────────────────────────────────────────")
    for r in sample:
        print(f"\n  company_cik : {r['company_cik']}")
        print(f"  year        : {r['year']}")
        print(f"  chunk_id    : {r['chunk_id']}")
        print(f"  text        : {r['text'][:200]}...")
        print(f"  text_clean  : {r['text_clean'][:200]}...")


if __name__ == "__main__":
    main()
