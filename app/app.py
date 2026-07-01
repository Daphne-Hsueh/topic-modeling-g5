import ast
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH       = BASE_DIR / "data" / "processed" / "corpus_with_ai_topics.csv"
EVAL_PATH         = BASE_DIR / "outputs" / "evaluation_results.csv"
TOPICS_TIME_PATH  = BASE_DIR / "outputs" / "topics_over_time.csv"
DISTANCE_MAP_PATH = BASE_DIR / "outputs" / "intertopic_distance_map.html"
WORDCLOUD_DIR     = BASE_DIR
SELECTED_COMPANIES_PATH = BASE_DIR / "data" / "selected_companies.csv"
KEYWORD_CATEGORIES_PATH = BASE_DIR / "keyword_categories.json"
EVAL_SET_PATH = BASE_DIR / "outputs" / "evaluation_set.csv"

# Muted, professional palette for multi-line charts (many series)
MUTED_PALETTE = px.colors.qualitative.Prism

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reputational Event Categorisation", layout="wide")

# ── Custom styling: tabs, KPI cards, dividers ───────────────────────────────────
st.markdown("""
<style>
/* Top-level tab bar */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F0F2F5;
    padding: 6px;
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 20px;
    border-radius: 8px;
    color: #5A6572;
    font-weight: 600;
    font-size: 15px;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #4A90D9 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Nested tabs (Trends / Explorer / Map): smaller and lighter than top-level */
.stTabs .stTabs [data-baseweb="tab-list"] {
    background-color: #E8ECF0;
    padding: 4px;
}
.stTabs .stTabs [data-baseweb="tab"] {
    height: 36px;
    padding: 0 14px;
    font-size: 13px;
    font-weight: 500;
}

/* KPI / metric cards */
[data-testid="stMetric"] {
    background-color: #F5F7FA;
    border: 1px solid #E1E6EB;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
    color: #6B7684;
}
[data-testid="stMetricValue"] {
    color: #1F2937;
    font-weight: 400;
}

/* Lighter dividers */
hr {
    border-color: #E1E6EB !important;
}
</style>
""", unsafe_allow_html=True)
# Note: the stMetric/stMetricLabel/stMetricValue selectors above match current
# Streamlit versions. If metric styling doesn't apply after an update, inspect
# the metric element in your browser dev tools and adjust the selector names.

st.title("Topic Modelling for Reputational Event Categorisation")
st.caption("Dynamic topic modeling of reputational risk in SEC 10-K filings, S&P 500 companies, 2010 to 2025")
st.write(
    "This dashboard applies Dynamic Topic Modeling (BERTopic) to the Item 1A Risk Factors "
    "sections of SEC 10-K annual reports to identify and track emerging reputational risk "
    "categories over time. It combines two complementary views of the same corpus: a "
    "predefined, literature-derived keyword dictionary, and topics discovered automatically "
    "by the BERTopic model. Model quality is assessed against a manually labeled gold standard."
)
st.markdown(
    "- **Data**: overview of the underlying corpus\n"
    "- **Keyword Dictionary**: chunks explored through predefined risk categories and keywords\n"
    "- **Model Results**: topics discovered automatically by the BERTopic model\n"
    "- **Evaluation**: how well the model's topics align with manual labels"
)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CORPUS_PATH, dtype={"cik": str})
    df["matched_categories"] = df["matched_categories"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


@st.cache_data
def load_eval() -> pd.DataFrame:
    return pd.read_csv(EVAL_PATH)


@st.cache_data
def load_topics_over_time(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(TOPICS_TIME_PATH)
    df = df.rename(columns={"Topic": "topic_id", "Frequency": "frequency", "Timestamp": "year"})
    df = df[df["topic_id"] != -1]
    label_map = eval_df.set_index("topic_id")["topic_label"].to_dict()
    df["topic_label"] = df["topic_id"].map(label_map).fillna(df["topic_id"].astype(str))
    return df


@st.cache_data
def load_distance_map() -> str:
    return DISTANCE_MAP_PATH.read_text(encoding="utf-8")


@st.cache_data
def build_labeled_distance_map(html_str: str, eval_df: pd.DataFrame) -> str:
    for _, row in eval_df.iterrows():
        old = f'"Topic {int(row["topic_id"])}"'
        new = f'"Topic {int(row["topic_id"])}: {row["topic_label"]}"'
        html_str = html_str.replace(old, new)
    return html_str


@st.cache_data
def load_selected_companies() -> pd.DataFrame:
    return pd.read_csv(SELECTED_COMPANIES_PATH)


@st.cache_data
def load_keyword_categories() -> dict:
    with open(KEYWORD_CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eval_set() -> pd.DataFrame:
    return pd.read_csv(EVAL_SET_PATH)


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading corpus..."):
    corpus_df = load_corpus()
    eval_df   = load_eval()
    tot_df    = load_topics_over_time(eval_df)

id_to_label = eval_df.set_index("topic_id")["topic_label"].to_dict()
corpus_df["topic_label"] = corpus_df["bertopic_id"].map(id_to_label).fillna("Noise / Unassigned")

# ── Helper for "select all / clear all" buttons ─────────────────────────────
def _set_all(keys, value):
    for k in keys:
        st.session_state[k] = value

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

year_min = int(corpus_df["year"].min())
year_max = int(corpus_df["year"].max())
selected_years = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
year_mask = corpus_df["year"].between(selected_years[0], selected_years[1])

st.sidebar.divider()
st.sidebar.caption("Sector and Company filters below apply only to the **Keyword Dictionary** tab.")

all_sectors = sorted(corpus_df["sector"].dropna().unique().tolist())
sector_keys = [f"sector_{s}" for s in all_sectors]

with st.sidebar.expander("Sector", expanded=True):
    col1, col2 = st.columns(2)
    col1.button("Select all", key="sector_select_all", on_click=_set_all, args=(sector_keys, True))
    col2.button("Clear all", key="sector_clear_all", on_click=_set_all, args=(sector_keys, False))

    selected_sectors = []
    for sector, key in zip(all_sectors, sector_keys):
        if st.checkbox(sector, value=True, key=key):
            selected_sectors.append(sector)

sector_mask = corpus_df["sector"].isin(selected_sectors)

company_lookup = (
    corpus_df.loc[sector_mask, ["ticker", "company"]]
    .drop_duplicates()
    .sort_values("company")
)
company_keys = [f"ticker_{t}" for t in company_lookup["ticker"]]

with st.sidebar.expander(f"Company ({len(company_lookup)})", expanded=False):
    col1, col2 = st.columns(2)
    col1.button("Select all", key="company_select_all", on_click=_set_all, args=(company_keys, True))
    col2.button("Clear all", key="company_clear_all", on_click=_set_all, args=(company_keys, False))

    selected_tickers = []
    for _, row in company_lookup.iterrows():
        key = f"ticker_{row['ticker']}"
        if st.checkbox(row["company"], value=True, key=key):
            selected_tickers.append(row["ticker"])

ticker_mask = corpus_df["ticker"].isin(selected_tickers)

filtered_df = corpus_df[year_mask].copy()
kw_filtered_df = corpus_df[year_mask & sector_mask & ticker_mask].copy()

st.divider()

# ── Top-level thematic sections ─────────────────────────────────────────────
section_data, section_keywords, section_model, section_eval = st.tabs([
    "① Data",
    "② Keyword Dictionary",
    "③ Model Results",
    "④ Evaluation",
])

# ══════════════════════════════════════════════════════════════════════════
# ① DATA
# ══════════════════════════════════════════════════════════════════════════
with section_data:
    st.subheader("Data Explorer")
    st.caption(
        "Overview of the dataset used across this dashboard: the full corpus of Item 1A "
        "Risk Factor text chunks extracted from SEC 10-K filings. Unfiltered, showing the "
        "entire corpus regardless of sidebar selections."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total chunks", f"{len(corpus_df):,}")
    k2.metric("Companies", corpus_df["ticker"].nunique())
    k3.metric("Sectors", corpus_df["sector"].nunique())
    k4.metric("Year range", f"{int(corpus_df['year'].min())}–{int(corpus_df['year'].max())}")

    st.divider()

    with st.container(border=True):
        st.markdown("#### Chunks per sector")
        st.caption(
            "Number of text chunks per GICS sector. Source: `corpus_with_ai_topics.csv`, "
            "chunk counts grouped by sector."
        )
        sector_counts = corpus_df["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "chunks"]
        fig_sector = px.bar(
            sector_counts, x="chunks", y="sector", orientation="h",
            color_discrete_sequence=["#4A90D9"],
            labels={"chunks": "Chunk count", "sector": "Sector"},
        )
        fig_sector.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sector, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Chunks per year")
        st.caption(
            "Number of chunks per filing year. Source: `corpus_with_ai_topics.csv`, "
            "chunk counts grouped by year."
        )
        year_counts = corpus_df["year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["year", "chunks"]
        fig_year = px.bar(
            year_counts, x="year", y="chunks",
            color_discrete_sequence=["#4A90D9"],
            labels={"chunks": "Chunk count", "year": "Year"},
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Company coverage")
        st.caption(
            "Every company included in the corpus, its sector, filing year range, and number "
            "of 10-K filings processed. Source: `selected_companies.csv`."
        )
        companies_df = load_selected_companies()
        st.dataframe(
            companies_df[["ticker", "company", "sector", "first_year", "last_year", "n_10k"]]
            .sort_values("company")
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# ② KEYWORD DICTIONARY
# ══════════════════════════════════════════════════════════════════════════
with section_keywords:
    st.subheader("Keyword Dictionary")
    st.caption(
        "This tab explores the corpus through a predefined keyword dictionary: 14 "
        "reputational risk categories, each defined by keywords and phrases derived from the "
        "academic literature (Hanay et al. 2024; Eckert 2017; Perera et al. 2022, among "
        "others). A chunk is matched to a category whenever it contains one or more of that "
        "category's keywords. This is a rule-based, literature-grounded taxonomy, distinct "
        "from the topics discovered automatically by the BERTopic model in Model Results. "
        "Filtered by the Sector, Company, and Year controls in the sidebar."
    )

    exploded = kw_filtered_df.explode("matched_categories")
    exploded = exploded[exploded["matched_categories"].notna() & (exploded["matched_categories"] != "")]

    if exploded.empty:
        st.info("No category data for the current sidebar filters.")
    else:
        with st.container(border=True):
            st.markdown("#### Chunks per category")
            st.caption(
                "How many chunks matched each of the 14 risk categories. Built from the "
                "`matched_categories` field, generated by scanning each chunk for keywords "
                "from `keyword_categories.json`. A chunk can match more than one category."
            )
            cat_counts = (
                exploded["matched_categories"].value_counts().reset_index()
                .rename(columns={"matched_categories": "category", "count": "chunks"})
            )
            fig_cat = px.bar(
                cat_counts, x="chunks", y="category", orientation="h",
                labels={"chunks": "Chunk count", "category": "Category"},
                color_discrete_sequence=["#4A90D9"],
            )
            fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_cat, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Category activity by sector")
            st.caption(
                "Which sectors mention each risk category most. Chunk counts grouped by "
                "category and sector, from the same matched-category data as above."
            )
            sector_cat = (
                exploded.groupby(["matched_categories", "sector"]).size().reset_index(name="chunks")
                .pivot(index="matched_categories", columns="sector", values="chunks")
                .fillna(0)
            )
            fig_sector_cat = px.imshow(
                sector_cat,
                labels={"x": "Sector", "y": "Category", "color": "Chunks"},
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_sector_cat.update_layout(height=max(400, len(sector_cat) * 30))
            fig_sector_cat.update_yaxes(tickmode="linear")
            st.plotly_chart(fig_sector_cat, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Category mentions over time")
            st.caption(
                "How often each category is mentioned per year. Chunk counts grouped by "
                "category and filing year."
            )
            cat_time = exploded.groupby(["matched_categories", "year"]).size().reset_index(name="chunks")
            fig_time = px.line(
                cat_time, x="year", y="chunks", color="matched_categories", markers=True,
                color_discrete_sequence=MUTED_PALETTE,
                labels={"chunks": "Chunk count", "year": "Year", "matched_categories": "Category"},
            )
            fig_time.update_layout(legend_title_text="Category", hovermode="x unified")
            st.plotly_chart(fig_time, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Keyword dictionary reference")
        st.caption(
            "The exact keywords and phrases used to match chunks to a selected category. "
            "Source: `keyword_categories.json`."
        )
        keyword_categories = load_keyword_categories()
        selected_cat = st.selectbox("Select a category", options=sorted(keyword_categories.keys()))
        st.write(", ".join(keyword_categories[selected_cat]))

# ══════════════════════════════════════════════════════════════════════════
# ③ MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════
with section_model:
    st.subheader("Model Results")
    st.caption(
        "This section presents the results of the BERTopic model: an unsupervised, "
        "data-driven method that discovered 39 distinct topics directly from the corpus "
        "text, without using the keyword dictionary. Unlike the predefined categories in "
        "Keyword Dictionary, these topics and their labels emerged purely from patterns in "
        "the language itself."
    )

    sub_trends, sub_explorer, sub_map = st.tabs(["Trends", "Explorer", "Map"])

    # -- Trends --------------------------------------------------------------
    with sub_trends:
        st.subheader("Trends")
        st.caption(
            "Shows how the frequency of each BERTopic-discovered topic changes over time, "
            "revealing which reputational risk themes are emerging or declining."
        )

        with st.container(border=True):
            st.markdown("#### Topics over time")
            st.caption(
                "Chunk frequency per topic, per year, across the full corpus. Not affected "
                "by sidebar sector or company filters, since topic modeling was run once on "
                "the entire dataset. Source: `outputs/topics_over_time.csv`."
            )
            time_filtered = tot_df[tot_df["year"].between(selected_years[0], selected_years[1])]

            if time_filtered.empty:
                st.info("No data for the selected year range.")
            else:
                fig = px.line(
                    time_filtered, x="year", y="frequency", color="topic_label", markers=True,
                    color_discrete_sequence=MUTED_PALETTE,
                    labels={"frequency": "Chunk frequency", "year": "Year", "topic_label": "Topic"},
                )
                fig.update_layout(legend_title_text="Topic", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Topics over time by sector")
            st.caption(
                "Narrowed to one sector, only topics that actually appear within that "
                "sector's filings. Computed from `corpus_with_ai_topics.csv`. Note: topics "
                "themselves are not sector-specific; this view only shows where a topic "
                "appears within the chosen sector. This selector is separate from the "
                "sidebar Sector filter."
            )

            sectors_available = sorted(filtered_df["sector"].dropna().unique().tolist())

            if not sectors_available:
                st.info("No data for the current year filter.")
            else:
                chosen_sector = st.selectbox("View topics for sector", options=sectors_available, key="trends_sector_select")

                sector_df = filtered_df[
                    (filtered_df["sector"] == chosen_sector) &
                    (filtered_df["bertopic_id"] != -1)
                ]
                sector_trend = (
                    sector_df.groupby(["bertopic_id", "year"]).size().reset_index(name="frequency")
                )
                sector_trend["topic_label"] = sector_trend["bertopic_id"].map(id_to_label)

                if sector_trend.empty:
                    st.info(f"No topic data for {chosen_sector} in the current year filter.")
                else:
                    fig_sector = px.line(
                        sector_trend, x="year", y="frequency", color="topic_label", markers=True,
                        color_discrete_sequence=MUTED_PALETTE,
                        labels={"frequency": "Chunk frequency", "year": "Year", "topic_label": "Topic"},
                    )
                    fig_sector.update_layout(legend_title_text="Topic", hovermode="x unified")
                    st.plotly_chart(fig_sector, use_container_width=True)

    # -- Explorer --------------------------------------------------------------
    with sub_explorer:
        st.subheader("Explorer")
        st.caption(
            "Browse the topics discovered by the model individually: their top keywords, "
            "how many chunks were assigned to each, and example text."
        )

        with st.container(border=True):
            st.markdown("#### Discovered topics")
            st.caption(
                "All 39 topics with their top keywords, number of assigned chunks, and "
                "whether a word cloud is available. Source: `outputs/evaluation_results.csv`."
            )
            explorer_df = eval_df[["topic_id", "topic_label", "top_keywords", "n_docs"]].copy()
            explorer_df["top_keywords"] = explorer_df["top_keywords"].str.replace(" ", "; ")
            explorer_df["wordcloud_available"] = explorer_df["topic_id"].apply(
                lambda tid: "Yes" if (WORDCLOUD_DIR / f"topic_{tid}_wordcloud.png").exists() else "No"
            )

            st.dataframe(
                explorer_df[["topic_label", "top_keywords", "n_docs", "wordcloud_available"]]
                .rename(columns={"wordcloud_available": "Wordcloud"})
                .sort_values("n_docs", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        with st.container(border=True):
            st.markdown("#### Topic detail")
            st.caption(
                "Select a topic to view its word cloud (where available) and a sample of "
                "real text chunks assigned to it. Word clouds are generated from the trained "
                "BERTopic model; sample chunks come from `corpus_with_ai_topics.csv`."
            )

            picker_df = explorer_df.sort_values("n_docs", ascending=False).copy()
            picker_df["display_label"] = picker_df.apply(
                lambda row: f"{row['topic_label']} (includes wordcloud)"
                if row["wordcloud_available"] == "Yes" else row["topic_label"],
                axis=1,
            )

            selected_display_label = st.selectbox("Select a topic to explore", options=picker_df["display_label"].tolist())
            topic_row = picker_df[picker_df["display_label"] == selected_display_label].iloc[0]
            topic_id  = int(topic_row["topic_id"])

            wc_path = WORDCLOUD_DIR / f"topic_{topic_id}_wordcloud.png"
            if wc_path.exists():
                st.image(str(wc_path), use_container_width=True)
            else:
                st.info(f"Word cloud not available for topic {topic_id}.")

            topic_chunks = filtered_df[filtered_df["bertopic_id"] == topic_id]
            sample = topic_chunks.sample(min(5, len(topic_chunks)), random_state=42) if len(topic_chunks) else topic_chunks

            if sample.empty:
                st.info("No chunks for this topic in the current year filter.")
            else:
                st.markdown(f"**Sample chunks** ({len(topic_chunks):,} total in current year filter):")
                for _, row in sample.iterrows():
                    label = f"{row['ticker']} · {int(row['year'])}"
                    with st.expander(label):
                        st.write(row["text"])
                        st.caption(f"Categories: {', '.join(row['matched_categories']) if row['matched_categories'] else '—'}")

        with st.container(border=True):
            st.markdown("#### Topic activity heatmap")
            st.caption(
                "Chunk frequency per topic, per year, as a heatmap. Source: "
                "`outputs/topics_over_time.csv`, pivoted into a topic by year matrix."
            )
            heatmap_data = tot_df.pivot_table(
                index="topic_label", columns="year", values="frequency", fill_value=0
            )
            fig_heat = px.imshow(
                heatmap_data,
                labels={"x": "Year", "y": "Topic", "color": "Chunks"},
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_heat.update_layout(height=max(500, len(heatmap_data) * 22))
            fig_heat.update_yaxes(tickmode="linear")
            st.plotly_chart(fig_heat, use_container_width=True)

    # -- Map --------------------------------------------------------------
    with sub_map:
        st.subheader("Map")
        with st.container(border=True):
            st.markdown("#### Intertopic distance map")
            st.caption(
                "Each point represents one topic; topics placed closer together share more "
                "similar language. Use the dropdown inside the map to highlight a specific "
                "topic, or the reference table to look up a label by topic number. Generated "
                "by BERTopic's built-in visualization function; labels come from "
                "`outputs/evaluation_results.csv`."
            )

            if not DISTANCE_MAP_PATH.exists():
                st.warning("Distance map file not found. Run model.py to generate it.")
            else:
                html_str = build_labeled_distance_map(load_distance_map(), eval_df)

                col_map, col_table = st.columns([3, 2])

                with col_map:
                    components.html(html_str, height=650, scrolling=True)

                with col_table:
                    st.markdown("**Topic reference**")
                    lookup_df = eval_df[["topic_id", "topic_label", "n_docs"]].sort_values("topic_id")
                    st.dataframe(
                        lookup_df.rename(columns={"topic_id": "ID", "topic_label": "Label", "n_docs": "Chunks"}),
                        use_container_width=True,
                        hide_index=True,
                        height=600,
                    )

# ══════════════════════════════════════════════════════════════════════════
# ④ EVALUATION
# ══════════════════════════════════════════════════════════════════════════
with section_eval:
    st.subheader("Model Evaluation")
    st.caption(
        "This section evaluates how well the BERTopic model's results align with "
        "reputational risk categories, comparing the model's output against a manually "
        "labeled gold standard. A sample of 150 text chunks was manually labeled with one "
        "of the 14 keyword-dictionary categories."
    )

    n_topics = len(eval_df)
    n_covered = eval_df["bertscore_f1"].notna().sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Topics evaluated", n_topics)
    k2.metric("With manual-label coverage", f"{n_covered}/{n_topics}")
    k3.metric("Mean BERTScore F1", f"{eval_df['bertscore_f1'].mean():.3f}")
    k4.metric("Topic diversity", f"{eval_df['diversity'].iloc[0]:.3f}")

    st.divider()

    with st.container(border=True):
        st.markdown("#### Per-topic metrics")
        st.caption(
            "Computed in `evaluate.py` using the `bert-score` library (Zhang et al., 2020), "
            "comparing each topic's keywords against its matched manual label's keyword "
            "dictionary text. Topics without manually labeled chunks show no BERTScore, a "
            "gap in evaluation-set coverage, not an error."
        )
        with st.expander("What do these metrics mean?"):
            st.markdown(
                "- **BERTScore F1**: semantic similarity between a topic's own keywords and "
                "the keyword-dictionary category it is compared against.\n"
                "- **Coherence (PMI)**: how often a topic's top keywords co-occur together "
                "within the corpus.\n"
                "- **Diversity**: share of unique keywords across all topics combined, a "
                "single value repeated on every row."
            )

        eval_display = eval_df[["topic_label", "bertscore_f1", "coherence", "diversity", "n_docs"]].copy()
        styled = (
            eval_display.style
            .background_gradient(subset=["bertscore_f1", "coherence"], cmap="Blues")
            .format({"bertscore_f1": "{:.4f}", "coherence": "{:.4f}", "diversity": "{:.4f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("#### Coherence vs. BERTScore")
        st.caption(
            "Coherence measures how tightly a topic's own words co-occur in the corpus. "
            "BERTScore measures how closely a topic's keywords match a manual risk category. "
            "Lower-left (low on both) topics are the weakest, good candidates for merging."
        )
        fig_scatter = px.scatter(
            eval_df.dropna(subset=["bertscore_f1"]),
            x="coherence", y="bertscore_f1",
            hover_name="topic_label", size="n_docs",
            color_discrete_sequence=["#4A90D9"],
            labels={"coherence": "Coherence (PMI)", "bertscore_f1": "BERTScore F1"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Manual label vs. model topic")
        st.caption(
            "Compares, chunk by chunk, the category a human assigned against the topic the "
            "model predicted. Two different taxonomies (39 model topics vs. 14 manual "
            "categories), so this shows alignment patterns rather than exact matches. "
            "Source: `outputs/evaluation_set.csv` joined with `outputs/evaluation_results.csv`."
        )

        eval_set_df = load_eval_set().copy()
        eval_set_df["model_topic_label"] = eval_set_df["predicted_topic_id"].map(id_to_label)
        eval_set_df.loc[eval_set_df["predicted_topic_id"] == -1, "model_topic_label"] = "Unassigned"

        cross = pd.crosstab(eval_set_df["manual_label"], eval_set_df["model_topic_label"])
        fig_cross = px.imshow(
            cross,
            labels={"x": "Model topic", "y": "Manual label", "color": "Chunks"},
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_cross.update_layout(height=max(550, len(cross) * 45))
        fig_cross.update_yaxes(tickmode="linear")
        st.plotly_chart(fig_cross, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Chunk-level comparison")
        st.caption(
            "Individual chunks with manual label, predicted topic, model confidence, and "
            "BERTScore, sorted to surface the worst-aligned chunks first. Source: "
            "`outputs/evaluation_set.csv`."
        )
        label_filter = st.selectbox(
            "Filter by manual label", options=["All"] + sorted(eval_set_df["manual_label"].unique().tolist())
        )
        table = eval_set_df if label_filter == "All" else eval_set_df[eval_set_df["manual_label"] == label_filter]
        table = table.sort_values("bertscore_f1", ascending=True)

        st.dataframe(
            table[["ticker", "manual_label", "model_topic_label", "topic_probability", "bertscore_f1", "text"]]
            .rename(columns={
                "manual_label": "Manual label",
                "model_topic_label": "Model topic",
                "topic_probability": "Model confidence",
                "bertscore_f1": "BERTScore F1",
                "text": "Chunk text",
            })
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )