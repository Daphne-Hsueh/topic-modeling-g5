# topic-modeling-G5

Dynamic topic modelling of **reputational risk** in SEC EDGAR 10-K filings
(Item 1A: Risk Factors), using BERTopic over an extended time period.

## Project Structure

```
topic-modeling-G5/
├── notebooks/
│   ├── 01_explore_companies.ipynb    # Select a sector-balanced S&P 500 sample
│   ├── 02_extraction_pipeline.ipynb  # Fetch 10-Ks, extract Item 1A, build corpus
│   └── 03_preprocessing.ipynb        # Clean + chunk the corpus, with QA checks
├── src/
│   ├── edgar.py                      # EDGAR client: listing, download, Item 1A extraction
│   ├── preprocess.py                 # Cleaning + chunking (corpus.csv -> chunks)
│   ├── model.py                      # BERTopic training 
│   └── evaluate.py                   # Evaluation incl. BERTScore 
├── app/
│   └── app.py                        # Streamlit dashboard 
├── data/
│   ├── item1a/                       # Extracted Item 1A text, one file per company-year
│   ├── processed/
│   │   └── chunks_clean.csv          # Model-ready chunks (output of notebook 03)
│   ├── results/                      # BERTopic + evaluation outputs 
│   ├── availability.csv              # Cached 10-K coverage matrix (built by nb 01)
│   ├── selected_companies.csv        # Final company selection
│   ├── filing_index.csv              # Per-filing extraction status (built by nb 02)
│   └── corpus.csv                    # Document corpus, one row per filing (built by nb 02)
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

## Pipeline

Run the notebooks in order; each writes the input for the next.

| Step | Notebook | What it does | Output |
|---|---|---|---|
| 1 | `01_explore_companies.ipynb` | Keep S&P 500 firms with a 10-K **every year 2010–2025**, then draw a **sector-balanced** random sample (8 × 11 GICS sectors = 88). | `selected_companies.csv` |
| 2 | `02_extraction_pipeline.ipynb` | Download each firm's full 10-K history and extract Item 1A. | `item1a/`, `filing_index.csv`, `corpus.csv` |
| 3 | `03_preprocessing.ipynb` | Clean and split sections into chunks (~one risk factor each), with QA. | `processed/chunks_clean.csv` |

### Notes on the data

- **Full filing history.** `src/edgar.py` paginates the SEC submissions API (and uses the
  browse-edgar feed for selection), so long histories from active filers — banks especially —
  are recovered rather than silently capped at the most recent ~1,000 filings.
- **`chunks_clean.csv` has two text columns:** use `text` (natural language) for BERTopic;
  `text_clean` (lower-cased, de-punctuated, stop-words removed) is only for a bag-of-words
  baseline such as LDA/NMF.
- **Complete panel.** Every selected company filed a 10-K in **every year 2010–2025**, so the
  dataset is balanced and gap-free across the full study period. (2026 is excluded because its
  filings are still arriving, which would leave the final year incomplete.)
