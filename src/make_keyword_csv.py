import json
import pandas as pd

with open("keyword_categories.json") as f:
    categories = json.load(f)

max_len = max(len(v) for v in categories.values())
data = {cat: kws + [""] * (max_len - len(kws)) for cat, kws in categories.items()}

pd.DataFrame(data).to_csv("data/processed/keyword_dictionary.csv", index=False)
print("Saved data/processed/keyword_dictionary.csv")