# topic-modeling-G5

NLP analysis of SEC EDGAR 10-K filings (Item 1A: Risk Factors).

## Project Structure

```
topic-modeling-G5/
├── data/
│   ├── filing_index.csv      # Index of fetched filings
│   ├── item1a/               # Extracted Item 1A text per company/year
│   └── raw_filings/          # Raw HTML downloads (not committed)
├── week1_edgar_pipeline.ipynb
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
| `week1_edgar_pipeline.ipynb` | EDGAR data fetching, Item 1A extraction, corpus construction |
