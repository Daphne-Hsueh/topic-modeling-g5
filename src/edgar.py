"""EDGAR client for the reputational-risk project.

Single source of truth for talking to SEC EDGAR, shared by the selection and
extraction notebooks:

    get_sp500()                      -> S&P 500 constituents (ticker, company, sector, cik)
    list_10k_filings(cik, lo, hi)    -> every 10-K in [lo, hi] (full history, paginated)
    download_filing(...)             -> raw HTML of a filing's primary document
    extract_item_1a(html)            -> the Item 1A "Risk Factors" body, or None

Why pagination matters: the `submissions/CIK*.json` "recent" block only holds the
most recent ~1,000 filings. For active filers (banks especially) that block does
not reach back to 2010, so naively reading only `recent` silently drops a decade
of 10-Ks. `list_10k_filings` also walks the historical `filings.files` shards, so
the full 10-K history is recovered.
"""
from __future__ import annotations

import re
import time
import warnings
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# A descriptive User-Agent with contact info is required by SEC fair-access rules.
HEADERS = {
    "User-Agent": "daphne s_hsueh25@stud.hwr-berlin.de",
    "Accept-Encoding": "gzip, deflate",
}
REQUEST_DELAY = 0.12  # be polite: SEC asks for <= ~10 requests/second

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BROWSE_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
              "&type=10-K&dateb=&owner=include&count=40&output=atom")


def _get(url: str, headers: dict | None = None) -> requests.Response:
    resp = requests.get(url, headers=headers or HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp


# ── company universe ───────────────────────────────────────────────────────────
def get_sp500() -> pd.DataFrame:
    """Current S&P 500 constituents from Wikipedia."""
    resp = _get(SP500_WIKI, headers={"User-Agent": "Mozilla/5.0 (research project)"})
    df = pd.read_html(StringIO(resp.text))[0][["Symbol", "Security", "GICS Sector", "CIK"]]
    df.columns = ["ticker", "company", "sector", "cik"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    return df


# ── filing history (full, paginated) ───────────────────────────────────────────
def _filings_frame(block: dict) -> pd.DataFrame:
    """Turn a submissions 'recent' block or a historical shard into a tidy frame."""
    if not block or not block.get("form"):
        return pd.DataFrame(columns=["form", "filing_date", "accession_number", "primary_document"])
    return pd.DataFrame({
        "form":             block["form"],
        "filing_date":      block["filingDate"],
        "accession_number": block["accessionNumber"],
        "primary_document": block["primaryDocument"],
    })


def list_10k_filings(cik: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Return every 10-K filed by `cik` with filing-year in [start_year, end_year].

    Walks both the 'recent' block and any historical shards whose date range
    overlaps the requested window, so long filing histories are not truncated.
    """
    cik = str(cik).zfill(10)
    data = _get(SUBMISSIONS_URL.format(name=f"CIK{cik}.json")).json()
    filings = data.get("filings", {})

    frames = [_filings_frame(filings.get("recent", {}))]
    for shard in filings.get("files", []):
        # only fetch a shard if its date range can contain a filing in our window
        if int(shard["filingTo"][:4]) >= start_year and int(shard["filingFrom"][:4]) <= end_year:
            shard_block = _get(SUBMISSIONS_URL.format(name=shard["name"])).json()
            frames.append(_filings_frame(shard_block))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"] == "10-K"].copy()
    df["year"] = pd.to_datetime(df["filing_date"]).dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    # one filing per year (occasionally amendments duplicate a year); keep the earliest
    df = df.sort_values("filing_date").drop_duplicates("year", keep="first")
    return df.sort_values("year").reset_index(drop=True)


def list_10k_years(cik: str) -> list[int]:
    """All years in which `cik` filed a 10-K, via the browse-edgar feed.

    One lightweight request per company that returns the *full* 10-K history —
    ideal for building a coverage matrix during company selection (use
    `list_10k_filings` when you also need accession numbers / document names).
    """
    cik = str(cik).zfill(10)
    text = _get(BROWSE_URL.format(cik=cik)).text
    types = re.findall(r"<filing-type>([^<]+)</filing-type>", text)
    dates = re.findall(r"<filing-date>([^<]+)</filing-date>", text)
    return sorted({int(d[:4]) for t, d in zip(types, dates) if t == "10-K"})


def download_filing(cik: str, accession_number: str, primary_document: str) -> str:
    """Raw HTML of a filing's primary document."""
    url = FILING_URL.format(
        cik=str(int(cik)),
        accession=accession_number.replace("-", ""),
        document=primary_document,
    )
    return _get(url).text


# ── Item 1A extraction ──────────────────────────────────────────────────────────
# Header anchors. `^` plus collapsed horizontal whitespace means these match only a
# header at the START of a line, not an inline cross-reference such as
# "...see Item 2, Properties..." mid-sentence (which previously truncated whole
# sections down to a single disclaimer paragraph).
_ITEM_1A_HDR  = re.compile(r"(?im)^[ \t]*item[ \t]*1a\b")
_ITEM_END_HDR = re.compile(r"(?im)^[ \t]*item[ \t]*(?:1b|2)\b")
_LEADING_TITLE = re.compile(r"^[ \t]*(?:[.\-–—:\"'”’]*\s*)?(?:ri\s*)?sk?\s*factors[.:]?\s*", re.IGNORECASE)
_COORD_CONJ    = re.compile(r"^(?:and|or|but)\b", re.IGNORECASE)

MIN_SECTION_CHARS = 1000  # a real Item 1A body is far longer than a table-of-contents line


def _trim_leading_noise(text: str) -> str:
    """Drop a leftover 'Risk Factors' title and any cross-reference fragment that
    precedes the section's first real sentence."""
    text = _LEADING_TITLE.sub("", text, count=1).lstrip()
    head = text.lstrip()
    if (not head) or (not head[0].isupper()) or _COORD_CONJ.match(head):
        m = re.search(r"[.!?][\"'”’\s]+([A-Z]\w[^.!?\n]{40,})", text)
        if m:
            text = text[m.start(1):]
    return text.strip()


def extract_item_1a(html: str) -> str | None:
    """Extract the Item 1A (Risk Factors) body from a 10-K's HTML.

    Robust to the common failure modes:
      * the table-of-contents 'Item 1A' entry (very short segment),
      * cross-references to 'Item 1A. Risk Factors' in later sections (e.g. MD&A),
        which also land at line-start after text extraction, and
      * truncation at an inline 'Item 2' cross-reference.

    Strategy: collect every line-anchored 'Item 1A' header and every 'Item 1B'/
    'Item 2' header. For each 'Item 1A' take the slice up to the first end-header
    that follows it, then choose the **longest such slice** — the real section body
    dwarfs TOC lines and cross-references. Prefer slices that are actually bounded
    by an end-header so a trailing cross-reference can't run to end-of-document.
    """
    soup = BeautifulSoup(html, "lxml")
    # Strip scripts/styles only. Do NOT strip <table> — many 10-Ks lay out the
    # Item 1A header and risk-factor body text inside tables, so removing tables
    # would delete the section we want.
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse every run of horizontal whitespace — including the non-breaking
    # spaces (\xa0) that 10-K HTML uses inside headers like "Item\xa01A" — to a
    # single regular space, so the line-anchored header regexes below can match.
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    starts = list(_ITEM_1A_HDR.finditer(text))
    ends   = list(_ITEM_END_HDR.finditer(text))
    if not starts:
        return None

    # candidate = (segment_text, is_bounded_by_end_header)
    candidates = []
    for s in starts:
        end = next((e for e in ends if e.start() > s.end()), None)
        segment = text[s.end(): end.start() if end else len(text)]
        candidates.append((segment, end is not None))

    bounded = [seg for seg, is_bounded in candidates if is_bounded]
    pool = bounded or [seg for seg, _ in candidates]   # fall back to unbounded only if needed
    best = max(pool, key=len, default="")

    if len(best) < MIN_SECTION_CHARS:
        return None
    return _trim_leading_noise(best.strip())
