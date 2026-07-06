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
DISTANCE_MAP_PATH = BASE_DIR / "outputs" / "visualizations" / "intertopic_distance_map.html"
BARCHARTS_PATH    = BASE_DIR / "outputs" / "visualizations" / "topic_barcharts.html"
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
hr {
    border-color: #E1E6EB !important;
}
</style>
""", unsafe_allow_html=True)

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
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    label_map = eval_df.set_index("topic_id")["topic_label"].to_dict()
    df["topic_label"] = df["topic_id"].map(label_map).fillna(df["topic_id"].astype(str))
    return df


@st.cache_data
def load_distance_map() -> str:
    return DISTANCE_MAP_PATH.read_text(encoding="utf-8")


@st.cache_data
def load_barcharts() -> str:
    return BARCHARTS_PATH.read_text(encoding="utf-8")


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

# Dynamic counts used in captions
n_model_topics = len(eval_df)
keyword_categories = load_keyword_categories()
n_categories = len(keyword_categories)

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
        f"This tab explores the corpus through a predefined keyword dictionary: {n_categories} "
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
                f"How many chunks matched each of the {n_categories} risk categories. Built from the "
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

            all_categories = sorted(exploded["matched_categories"].unique().tolist())
            cat_keys = [f"cat_time_{c}" for c in all_categories]

            with st.expander("Select categories to display", expanded=False):
                if st.button("Select all categories", key="cat_time_select_all"):
                    _set_all(cat_keys, True)
                if st.button("Clear all categories", key="cat_time_clear_all"):
                    _set_all(cat_keys, False)

                selected_categories = []
                for cat, key in zip(all_categories, cat_keys):
                    if st.checkbox(cat, value=True, key=key):
                        selected_categories.append(cat)

            cat_time_filtered = exploded[exploded["matched_categories"].isin(selected_categories)]
            cat_time = cat_time_filtered.groupby(["matched_categories", "year"]).size().reset_index(name="chunks")

            if not selected_categories:
                st.info("Select one or more categories above to display the chart.")
            elif cat_time.empty:
                st.info("No data for the selected categories and filters.")
            else:
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
        selected_cat = st.selectbox("Select a category", options=sorted(keyword_categories.keys()))
        st.write(", ".join(keyword_categories[selected_cat]))

# ══════════════════════════════════════════════════════════════════════════
# ③ MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════
with section_model:
    st.subheader("Model Results")
    st.caption(
        f"This section presents the results of the BERTopic model: an unsupervised, "
        f"data-driven method that discovered {n_model_topics} distinct topics directly from the "
        "corpus text, without using the keyword dictionary. Unlike the predefined categories in "
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

            all_topic_labels = sorted(tot_df["topic_label"].unique().tolist())
            tot_keys = [f"tot_topic_{lbl}" for lbl in all_topic_labels]

            with st.expander("Select topics to display", expanded=False):
                if st.button("Select all topics", key="tot_select_all"):
                    _set_all(tot_keys, True)
                if st.button("Clear all topics", key="tot_clear_all"):
                    _set_all(tot_keys, False)

                selected_topics = []
                for lbl, key in zip(all_topic_labels, tot_keys):
                    if st.checkbox(lbl, value=True, key=key):
                        selected_topics.append(lbl)

            time_filtered = tot_df[
                tot_df["topic_label"].isin(selected_topics) &
                tot_df["year"].between(selected_years[0], selected_years[1])
            ].copy()
            time_filtered["year"] = time_filtered["year"].astype(int)

            if not selected_topics:
                st.info("Select one or more topics above to display the chart.")
            elif time_filtered.empty:
                st.info("No data for the selected topics and year range.")
            else:
                fig = px.line(
                    time_filtered, x="year", y="frequency", color="topic_label", markers=True,
                    color_discrete_sequence=MUTED_PALETTE,
                    labels={"frequency": "Chunk frequency", "year": "Year", "topic_label": "Topic"},
                )
                fig.update_layout(legend_title_text="Topic", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    # -- Explorer --------------------------------------------------------------
    with sub_explorer:
        st.subheader("Explorer")
        st.caption(
            "Browse the topics discovered by the model: their number, top keywords, how many "
            "chunks were assigned to each, and their top-word bar charts."
        )

        with st.container(border=True):
            st.markdown("#### Discovered topics")
            st.caption(
                f"All {n_model_topics} topics with their topic number, top keywords, and number "
                "of assigned chunks. The topic number matches the labels in the bar charts "
                "below. Source: `outputs/evaluation_results.csv`."
            )
            explorer_df = eval_df[["topic_id", "topic_label", "top_keywords", "n_docs"]].copy()
            explorer_df["top_keywords"] = explorer_df["top_keywords"].str.replace(" ", "; ")
            explorer_df["Topic"] = explorer_df["topic_id"].apply(lambda t: f"Topic {int(t)}")

            st.dataframe(
                explorer_df[["Topic", "topic_label", "top_keywords", "n_docs"]]
                .rename(columns={"topic_label": "Label", "top_keywords": "Top keywords", "n_docs": "Chunks"})
                .sort_values("Chunks", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        with st.container(border=True):
            st.markdown("#### Top words per topic")
            st.caption(
                "Bar charts of the highest-scoring keywords for each topic (c-TF-IDF weights). "
                "Generated by the BERTopic model. Source: "
                "`outputs/visualizations/topic_barcharts.html`."
            )
            if BARCHARTS_PATH.exists():
                components.html(load_barcharts(), height=650, scrolling=True)
            else:
                st.warning("Bar chart file not found. Run visualize.py to generate it.")

        with st.container(border=True):
            st.markdown("#### Sample chunks per topic")
            st.caption(
                "Select a topic to read a sample of real text chunks the model assigned to it. "
                "Source: `corpus_with_ai_topics.csv`, filtered to the selected topic and the "
                "current year range."
            )

            picker_df = explorer_df.sort_values("n_docs", ascending=False).copy()
            picker_df["picker_label"] = picker_df.apply(
                lambda r: f"Topic {int(r['topic_id'])}: {r['topic_label']}", axis=1
            )
            selected_picker = st.selectbox(
                "Select a topic", options=picker_df["picker_label"].tolist(), key="explorer_chunk_topic"
            )
            topic_id = int(picker_df[picker_df["picker_label"] == selected_picker].iloc[0]["topic_id"])

            topic_chunks = filtered_df[filtered_df["bertopic_id"] == topic_id]
            sample = (
                topic_chunks.sample(min(5, len(topic_chunks)), random_state=42)
                if len(topic_chunks) else topic_chunks
            )

            if sample.empty:
                st.info("No chunks for this topic in the current year range.")
            else:
                st.markdown(f"**Showing 5 of {len(topic_chunks):,} chunks** in the current year range:")
                for _, row in sample.iterrows():
                    label = f"{row['ticker']} · {int(row['year'])}"
                    with st.expander(label):
                        st.write(row["text"])
                        cats = ", ".join(row["matched_categories"]) if row["matched_categories"] else "—"
                        st.caption(f"Keyword categories matched: {cats}")

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
                st.warning("Distance map file not found. Run visualize.py to generate it.")
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

    eval_set_df = load_eval_set().copy()
    n_eval_chunks = eval_set_df["chunk_id"].nunique() if "chunk_id" in eval_set_df.columns else len(eval_set_df)

    st.caption(
        "This section evaluates how well the BERTopic model's results align with "
        "reputational risk categories, comparing the model's output against a manually "
        f"labeled gold standard of {n_eval_chunks} text chunks, each manually labeled with "
        f"one of the {n_categories} keyword-dictionary categories."
    )

    n_topics = len(eval_df)
    n_covered = int(eval_df["bertscore_f1"].notna().sum())
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

        eval_display = eval_df[["topic_label", "bertscore_f1", "coherence"]].copy()
        try:
            styled = (
                eval_display.style
                .background_gradient(subset=["bertscore_f1", "coherence"], cmap="Blues")
                .format({"bertscore_f1": "{:.4f}", "coherence": "{:.4f}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(
                eval_display.round(4), use_container_width=True, hide_index=True
            )

    with st.container(border=True):
        st.markdown("#### Manual label vs. model topic")
        st.caption(
            "Compares, chunk by chunk, the category a human assigned against the topic the "
            f"model predicted. Two different taxonomies ({n_model_topics} model topics vs. "
            f"{n_categories} manual categories), so this shows alignment patterns rather than "
            "exact matches. Source: `outputs/evaluation_set.csv` joined with "
            "`outputs/evaluation_results.csv`."
        )

        eval_set_df["model_topic_label"] = eval_set_df["predicted_topic_id"].map(id_to_label)
        eval_set_df.loc[eval_set_df["predicted_topic_id"] == -1, "model_topic_label"] = "Unassigned"
        eval_set_df["model_topic_label"] = eval_set_df["model_topic_label"].fillna("Unassigned")

        cross = pd.crosstab(eval_set_df["manual_label"], eval_set_df["model_topic_label"])
        if cross.empty:
            st.info("No labeled chunks available to compare.")
        else:
            fig_cross = px.imshow(
                cross,
                labels={"x": "Model topic", "y": "Manual label", "color": "Chunks"},
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cross.update_layout(height=max(550, len(cross) * 45))
            fig_cross.update_yaxes(tickmode="linear")
            st.plotly_chart(fig_cross, use_container_width=True)