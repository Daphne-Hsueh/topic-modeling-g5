"""
Analyze paragraph-level word count distribution in corpus.csv.

Paragraphs are defined by splitting `text` on double newlines (\n\n),
which mirrors the chunking logic in pipeline.py before the 400-word cap kicks in.

Outputs:
  - Terminal summary (stats + percentiles + % exceeding 400 words)
  - data/processed/paragraph_length_distribution.png
  - data/processed/chunk_size_analysis.md
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PNG_PATH = os.path.join(OUT_DIR, "paragraph_length_distribution.png")
MD_PATH = os.path.join(OUT_DIR, "chunk_size_analysis.md")

CHUNK_CAP = 400
MIN_WORDS = 20       # pipeline discards paragraphs shorter than this
LONG_THRESHOLD = 3000
LONG_EXAMPLES = 3
PREVIEW_CHARS = 300
THRESHOLDS = [300, 400, 500]


def load_paragraphs(path: str) -> tuple[np.ndarray, list[str]]:
    """Return (word_count_array, list_of_texts_exceeding_LONG_THRESHOLD)."""
    df = pd.read_csv(path, usecols=["text"], dtype=str)
    df["text"] = df["text"].fillna("")

    word_counts: list[int] = []
    long_texts: list[str] = []

    for text in df["text"]:
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            wc = len(para.split())
            word_counts.append(wc)
            if wc > LONG_THRESHOLD:
                long_texts.append(para)

    return np.array(word_counts, dtype=np.int32), long_texts


def compute_stats(wc: np.ndarray) -> dict:
    pcts = np.percentile(wc, [50, 75, 90, 95, 99])
    return {
        "n_paragraphs": len(wc),
        "mean": float(np.mean(wc)),
        "median": float(np.median(wc)),
        "std": float(np.std(wc)),
        "min": int(np.min(wc)),
        "max": int(np.max(wc)),
        "p50": float(pcts[0]),
        "p75": float(pcts[1]),
        "p90": float(pcts[2]),
        "p95": float(pcts[3]),
        "p99": float(pcts[4]),
        "pct_over_cap": float(np.mean(wc > CHUNK_CAP) * 100),
    }


def compute_sensitivity(wc_filtered: np.ndarray, thresholds: list[int]) -> list[dict]:
    rows = []
    for t in thresholds:
        over = wc_filtered > t
        pct_over = float(np.mean(over) * 100)
        # paragraphs at or under cap → 1 chunk each; over cap → ceil(wc / t) chunks
        chunks = int(np.where(over, np.ceil(wc_filtered / t), 1).sum())
        rows.append({"threshold": t, "pct_over": pct_over, "total_chunks": chunks})
    return rows


def print_sensitivity(rows: list[dict]) -> None:
    print("\n  Threshold Sensitivity (filtered population, >= 20 words)")
    print(f"  {'Threshold':>10}  {'% exceeding':>12}  {'Total chunks':>13}")
    print("  " + "-" * 40)
    for r in rows:
        print(
            f"  {r['threshold']:>10,}  {r['pct_over']:>11.2f}%  {r['total_chunks']:>13,}"
        )
    print()


def print_stats_block(label: str, s: dict) -> None:
    print(f"\n  {label}")
    print(f"    Paragraphs      : {s['n_paragraphs']:>12,}")
    print(f"    Mean            : {s['mean']:>12.1f}")
    print(f"    Median          : {s['median']:>12.1f}")
    print(f"    Std dev         : {s['std']:>12.1f}")
    print(f"    Min             : {s['min']:>12,}")
    print(f"    Max             : {s['max']:>12,}")
    print(f"    50th pct        : {s['p50']:>12.1f}")
    print(f"    75th pct        : {s['p75']:>12.1f}")
    print(f"    90th pct        : {s['p90']:>12.1f}")
    print(f"    95th pct        : {s['p95']:>12.1f}")
    print(f"    99th pct        : {s['p99']:>12.1f}")
    print(f"    > {CHUNK_CAP} words       : {s['pct_over_cap']:>11.2f}%")


def print_summary(s_all: dict, s_filtered: dict, long_texts: list[str]) -> None:
    print("=" * 52)
    print("  Paragraph Word-Count Analysis")
    print(f"  (chunk cap = {CHUNK_CAP} words)")
    print("=" * 52)

    print_stats_block("All paragraphs (raw splits)", s_all)
    print()
    print_stats_block(
        f"Filtered: paragraphs >= {MIN_WORDS} words (pipeline-relevant)", s_filtered
    )

    # Long paragraph breakdown
    n_long = len(long_texts)
    print()
    print(f"  Paragraphs > {LONG_THRESHOLD:,} words")
    print(f"    Count           : {n_long:>12,}")
    print(f"    % of all        : {n_long / s_all['n_paragraphs'] * 100:>11.2f}%")
    print()
    print(f"  Examples (first {PREVIEW_CHARS} chars each):")
    for i, text in enumerate(long_texts[:LONG_EXAMPLES], 1):
        preview = text[:PREVIEW_CHARS].replace("\n", " ")
        print(f"\n  [{i}] ({len(text.split()):,} words)")
        print(f"  {preview} …")

    print("\n" + "=" * 52)


def save_histogram(wc_filtered: np.ndarray, path: str) -> None:
    display_cap = 800
    clipped = wc_filtered[wc_filtered <= display_cap]
    n_clipped = len(wc_filtered) - len(clipped)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(clipped, bins=100, color="#4C72B0", edgecolor="none", alpha=0.85)

    ax.axvline(300, color="#E07B00", linewidth=1.5, linestyle="--", label="300-word threshold")
    ax.axvline(400, color="#DD3333", linewidth=2.5, linestyle="--", label="400-word threshold (chosen)")
    ax.axvline(500, color="#1A6DB5", linewidth=1.5, linestyle="--", label="500-word threshold")

    ax.set_xlabel("Paragraph word count", fontsize=12)
    ax.set_ylabel("Number of paragraphs", fontsize=12)
    ax.set_title(
        f"Paragraph length distribution — filtered population (≥ 20 words, n = {len(wc_filtered):,})\n"
        f"x-axis capped at {display_cap} words; {n_clipped:,} paragraphs beyond cap not shown",
        fontsize=12,
    )
    ax.set_xlim(0, display_cap)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Histogram saved → {path}")


def _stats_table(s: dict) -> list[str]:
    return [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total paragraphs | {s['n_paragraphs']:,} |",
        f"| Mean | {s['mean']:.1f} |",
        f"| Median | {s['median']:.1f} |",
        f"| Std dev | {s['std']:.1f} |",
        f"| Min | {s['min']:,} |",
        f"| Max | {s['max']:,} |",
        f"| 50th percentile | {s['p50']:.1f} |",
        f"| 75th percentile | {s['p75']:.1f} |",
        f"| 90th percentile | {s['p90']:.1f} |",
        f"| 95th percentile | {s['p95']:.1f} |",
        f"| 99th percentile | {s['p99']:.1f} |",
        f"| % exceeding {CHUNK_CAP} words | {s['pct_over_cap']:.2f}% |",
    ]


def save_markdown(
    s_all: dict,
    s_filtered: dict,
    long_texts: list[str],
    sensitivity: list[dict],
    png_path: str,
    md_path: str,
) -> None:
    rel_png = os.path.relpath(png_path, os.path.dirname(md_path))
    n_long = len(long_texts)

    lines = [
        "# Chunk Size Analysis",
        "",
        f"Paragraph boundaries defined by `\\n\\n` splits (mirrors pipeline.py).  ",
        f"Chunk cap under analysis: **{CHUNK_CAP} words**.",
        "",
        "## 1. All Paragraphs (Raw Splits)",
        "",
        *_stats_table(s_all),
        "",
        "## 2. Pipeline-Relevant Paragraphs (≥ 20 Words)",
        "",
        f"The pipeline discards any paragraph shorter than {MIN_WORDS} words downstream, "
        f"so this is the population that actually matters for justifying the {CHUNK_CAP}-word cap.",
        "",
        *_stats_table(s_filtered),
        "",
        "## 3. Very Long Paragraphs (> 3,000 Words)",
        "",
        f"**{n_long:,}** paragraphs exceed {LONG_THRESHOLD:,} words "
        f"({n_long / s_all['n_paragraphs'] * 100:.2f}% of all paragraphs).  ",
        "These are likely parsing artifacts — SEC filings that lack double-newline breaks "
        "causing multiple paragraphs to merge into a single blob.",
        "",
        "### Examples",
        "",
    ]

    for i, text in enumerate(long_texts[:LONG_EXAMPLES], 1):
        preview = text[:PREVIEW_CHARS].replace("\n", " ")
        lines += [
            f"**Example {i}** ({len(text.split()):,} words)",
            "",
            f"> {preview} …",
            "",
        ]

    lines += [
        "## 4. Threshold Sensitivity (300 vs 400 vs 500)",
        "",
        f"Applied to the filtered population ({s_filtered['n_paragraphs']:,} paragraphs ≥ {MIN_WORDS} words).  ",
        "Total chunks = paragraphs at or under threshold (1 chunk each) "
        "+ paragraphs over threshold split into ⌈word_count / threshold⌉ chunks.",
        "",
        "| Threshold | % exceeding | Total chunks |",
        "| --- | --- | --- |",
        *[
            f"| {r['threshold']:,} words | {r['pct_over']:.2f}% | {r['total_chunks']:,} |"
            for r in sensitivity
        ],
        "",
        "## Distribution",
        "",
        "Dashed lines mark the three candidate thresholds: orange = 300 words, "
        f"red (bold) = {CHUNK_CAP} words (chosen), blue = 500 words.  ",
        "Display clipped at 800 words for readability; the long tail (including "
        "the artifacts discussed above) is excluded from the plot but included in all statistics.",
        "",
        f"![Paragraph length distribution]({rel_png})",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Markdown saved   → {md_path}\n")


def main() -> None:
    print(f"\nLoading corpus from {CORPUS_PATH} …")
    wc_all, long_texts = load_paragraphs(CORPUS_PATH)
    print(f"Extracted {len(wc_all):,} paragraphs. Computing stats …")

    s_all = compute_stats(wc_all)

    wc_filtered = wc_all[wc_all >= MIN_WORDS]
    s_filtered = compute_stats(wc_filtered)

    sensitivity = compute_sensitivity(wc_filtered, THRESHOLDS)

    print_summary(s_all, s_filtered, long_texts)
    print_sensitivity(sensitivity)

    os.makedirs(OUT_DIR, exist_ok=True)
    save_histogram(wc_filtered, PNG_PATH)
    save_markdown(s_all, s_filtered, long_texts, sensitivity, PNG_PATH, MD_PATH)


if __name__ == "__main__":
    main()
