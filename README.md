# topic-modeling-G5

Dynamic topic modelling of **reputational risk** in SEC EDGAR 10-K filings
(Item 1A: Risk Factors), using BERTopic over an extended time period.

## Pipeline: 5 steps

Every file is prefixed with the step that produces (or belongs to) it:

| Step | Name | Run | Produces |
|---|---|---|---|
| 1 | Data extraction | `notebooks/step1_explore_companies.ipynb`, then `notebooks/step1_extraction_pipeline.ipynb` (both use `src/step1_edgar.py`) | `data/step1_availability.csv`, `data/step1_selected_companies.csv`, `data/item1a/`, `data/step1_filing_index.csv`, `data/step1_corpus.csv` |
| 2 | Preprocessing | `python -m src.step2_preprocess`, then `python -m src.step2_keyword_filter` | `data/processed/step2_chunks_clean.csv`, `data/processed/step2_filtered_corpus.csv` |
| 3 | Model | `python src/step3_model.py` (first run caches embeddings) → `python src/step3_tune_hyperparameters.py` → `python src/step3_model.py` again (picks up tuned params) → `python src/step3_visualize.py` | `data/processed/step3_embeddings.npy`, `outputs/models/`, `data/processed/step3_corpus_with_bert_topics.csv`, `outputs/step3_best_params.json`, `outputs/step3_topics_over_time.csv`, `outputs/visualizations/` |
| 4 | Evaluation | `python src/step4_evaluate.py` (reads the hand-labeled `data/step4_corpus_sample_150_manual.csv`, drawn by `src/make_corpus_sample.py`) | `outputs/step4_evaluation_results.csv`, `outputs/step4_evaluation_set.csv` |
| 5 | Streamlit | `streamlit run app/app.py` | interactive dashboard reading the outputs of steps 3–4 |

## Project Structure

```
topic-modeling-G5/
├── notebooks/
│   ├── step1_explore_companies.ipynb    # Select a sector-balanced S&P 500 sample
│   └── step1_extraction_pipeline.ipynb  # Fetch 10-Ks, extract Item 1A, build corpus
├── src/
│   ├── step1_edgar.py                   # EDGAR client: listing, download, Item 1A extraction
│   ├── step2_preprocess.py              # Cleaning + chunking (step1_corpus.csv -> chunks)
│   ├── step2_keyword_filter.py          # Keep only chunks matching the keyword dictionary
│   ├── step2_analyze_chunk_size.py      # QA: justifies the chunking thresholds
│   ├── step3_model.py                   # BERTopic training (uses tuned params if present)
│   ├── step3_tune_hyperparameters.py    # Grid search (NPMI coherence) -> step3_best_params.json
│   ├── step3_visualize.py               # Distance map, topic barcharts, topics over time
│   ├── step4_evaluate.py                # Evaluation incl. BERTScore
│   ├── check_data.py                    # Utility: inspect the filtered corpus
│   ├── verify_pipeline.py               # Utility: check pipeline files exist
│   ├── make_corpus_sample.py            # Utility: draw the 150-chunk evaluation sample
│   ├── make_keyword_csv.py              # Utility: export keyword_categories.json -> CSV
│   └── make_keyword_excel.py            # Utility: same export as Excel
├── app/
│   └── app.py                           # Step 5: Streamlit dashboard
├── data/
│   ├── item1a/                          # Extracted Item 1A text, one file per company-year
│   ├── step1_availability.csv           # Cached 10-K coverage matrix
│   ├── step1_selected_companies.csv     # Final company selection
│   ├── step1_filing_index.csv           # Per-filing extraction status
│   ├── step1_corpus.csv                 # Document corpus, one row per filing
│   ├── step2_keyword_dictionary.csv     # CSV export of the keyword dictionary
│   ├── step4_corpus_sample_150_manual.csv  # Hand-labeled evaluation sample
│   └── processed/
│       ├── step2_chunks_clean.csv       # Model-ready chunks
│       ├── step2_filtered_corpus.csv    # Chunks matching >=1 keyword category (training corpus)
│       ├── step2_chunk_size_analysis.md # Why chunks are 20-400 words (+ .png chart)
│       ├── step3_embeddings.npy         # Cached sentence-transformer embeddings
│       ├── step3_corpus_with_bert_topics.csv  # Chunks + assigned BERTopic topic
│       └── step4_corpus_sample_150.csv  # Raw (unlabeled) evaluation sample
├── outputs/
│   ├── step3_best_params.json           # Grid-search winner
│   ├── step3_grid_search_results.csv    # All grid-search combos
│   ├── step3_topics_over_time.csv       # Topic frequency per year
│   ├── step4_evaluation_results.csv     # Per-topic BERTScore F1 + eval-chunk count
│   ├── step4_evaluation_set.csv         # Labeled evaluation rows
│   ├── models/                          # Saved BERTopic engine
│   └── visualizations/                  # Interactive HTML charts
├── .streamlit/config.toml               # Dashboard theme settings
├── keyword_categories.json              # Hand-curated reputation-risk keyword dictionary
├── dictionary-documentation.md          # How the dictionary was built
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv ~/.venvs/repurisk-env
source ~/.venvs/repurisk-env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

### Notes on the data

- **Full filing history.** `src/step1_edgar.py` paginates the SEC submissions API (and uses the
  browse-edgar feed for selection), so long histories from active filers — banks especially —
  are recovered rather than silently capped at the most recent ~1,000 filings.
- **`step2_chunks_clean.csv` has one text column, `text`:** the natural-language chunk that BERTopic
  embeds. No stop-word removal or lower-casing is done in preprocessing — BERTopic tokenises
  internally (configure stop-words in its `CountVectorizer` if you want cleaner topic words).
- **Complete panel.** Every selected company filed a 10-K in **every year 2010–2025**, so the
  dataset is balanced and gap-free across the full study period. (2026 is excluded because its
  filings are still arriving, which would leave the final year incomplete.)
- **`data/step4_corpus_sample_150_manual.csv` contains 150 hand-assigned labels.** Rerunning
  `make_corpus_sample.py` writes a fresh *unlabeled* sample to `data/processed/` — never
  overwrite the manual file with it.
