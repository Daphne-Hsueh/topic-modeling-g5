import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path

# 1. Setup the dynamic path
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "corpus_with_ai_topics.csv"

print("Loading clustered data...")
df = pd.read_csv(CSV_PATH)

# 2. Get the top 5 most common topics (excluding noise)
valid_topics = df[df['bertopic_id'] != -1]
top_5_topics = valid_topics['bertopic_id'].value_counts().head(5).index.tolist()

# 3. Define custom corporate and legal boilerplate to ignore
SEC_STOPWORDS = set(STOPWORDS).union({
    "may", "result", "business", "including", "operation", "operations", 
    "company", "could", "adversely", "affect", "subject", "significant", 
    "addition", "certain", "related", "u", "s", "cost", "costs", "future", 
    "requirement", "requirements", "law", "laws", "claim", "claims", 
    "impact", "potential", "effect", "material", "condition", "financial",
    "risk", "risks", "increase", "increased"
})

# 4. Generate the Word Clouds
for topic_id in top_5_topics:
    topic_text = " ".join(df[df['bertopic_id'] == topic_id]['text_clean'].dropna())
    
    wordcloud = WordCloud(
        width=1200, 
        height=600, 
        background_color='black', 
        colormap='viridis',
        collocations=False,
        stopwords=SEC_STOPWORDS, # Apply the custom filter here
        max_words=50
    ).generate(topic_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Risk Factor Cluster: Topic {topic_id}", fontsize=18, color="white", pad=20)
    
    plt.gcf().patch.set_facecolor('black')
    
    save_path = BASE_DIR / f"topic_{topic_id}_wordcloud.png"
    plt.savefig(save_path, facecolor='black', bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()