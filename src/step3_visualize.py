import pandas as pd
from bertopic import BERTopic
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
TOT_PATH = OUTPUT_DIR / "step3_topics_over_time.csv"

# Create a dedicated folder for images/html
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    print("ERROR: Model directory not found. Please run step3_model.py first.")
    exit()

print("Loading saved BERTopic model engine...")
topic_model = BERTopic.load(str(MODEL_DIR))

# Extract the topic information dataframe to calculate the topic count dynamically
topic_info = topic_model.get_topic_info()

# Filter out the -1 noise category so we only count valid, cohesive clusters
valid_topics = topic_info[topic_info['Topic'] != -1]
num_valid_topics = len(valid_topics)

print(f"Dynamic Check: Detected {num_valid_topics} valid topics in the saved model.")

# 1. Intertopic Distance Map
print("Generating Intertopic Distance Map...")
distance_fig = topic_model.visualize_topics()
distance_path = VIS_DIR / "intertopic_distance_map.html"
distance_fig.write_html(str(distance_path))
print(f"Saved: {distance_path}")

# 2. Top Words Barchart (Dynamic topic count passed here)
print(f"Generating Topic Barcharts for all {num_valid_topics} topics...")
barchart_fig = topic_model.visualize_barchart(top_n_topics=num_valid_topics)
barchart_path = VIS_DIR / "topic_barcharts.html"
barchart_fig.write_html(str(barchart_path))
print(f"Saved: {barchart_path}")

# 3. Topics Over Time
if TOT_PATH.exists():
    print("Generating Topics Over Time chart...")
    topics_over_time = pd.read_csv(TOT_PATH)
    tot_fig = topic_model.visualize_topics_over_time(topics_over_time)
    tot_path = VIS_DIR / "topics_over_time.html"
    tot_fig.write_html(str(tot_path))
    print(f"Saved: {tot_path}")

print("\nAll visualizations generated successfully in outputs/visualizations/")