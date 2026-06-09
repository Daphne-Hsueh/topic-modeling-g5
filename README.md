# topic-modeling-G5

NLP analysis of SEC EDGAR 10-K filings (Item 1A: Risk Factors).

## Project Structure

```
topic-modeling-G5/
├── notebooks/
│   ├── 01_explore_companies.ipynb    # Browse EDGAR companies, select S&P 500 sample
│   └── 02_extraction_pipeline.ipynb  # Fetch 10-K filings, extract Item 1A, build corpus
├── src/
│   ├── preprocess.py                 # Text cleaning and tokenisation
│   ├── model.py                      # BERTopic training and inference
│   └── evaluate.py                   # Topic quality metrics
├── app/
│   └── app.py                        # Streamlit dashboard
├── data/
│   ├── item1a/                       # Extracted Item 1A text per company/year
│   ├── raw_filings/                  # Raw HTML downloads (not committed)
│   ├── processed/                    # Cleaned documents ready for modelling
│   ├── results/                      # BERTopic outputs and evaluation results
│   ├── corpus.csv                    # Full document corpus with metadata
│   ├── filing_index.csv              # Index of fetched filings
│   └── selected_companies.csv        # Final company selection for the pipeline
├── requirements.txt
└── README.md
```

## Setup

**1. Create and activate a virtual environment** (outside the project folder):

```bash
python3 -m venv ~/.venvs/repurisk-env
source ~/.venvs/repurisk-env/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Launch Jupyter:**

```bash
jupyter notebook
```

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_explore_companies.ipynb` | Browse EDGAR companies, filter S&P 500, select final sample |
| `notebooks/02_extraction_pipeline.ipynb` | Fetch 10-K filings, extract Item 1A, build corpus |
