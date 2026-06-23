import pandas as pd
from pathlib import Path

# 1. Build the path to the CSV
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "filtered_corpus.csv"

print(f"Inspecting file: {CSV_PATH}\n")

# 2. Load the data
df = pd.read_csv(CSV_PATH)

# 3. Get the absolute row count
total_rows = len(df)
print(f"Total rows (chunks) in CSV: {total_rows}")

# 4. Check for duplicates
# If the text, company, and year are identical, it is a duplicate row
duplicate_count = df.duplicated(subset=['text', 'ticker', 'year']).sum()

print(f"Exact duplicates found: {duplicate_count}")

if duplicate_count > 0:
    print(f"\nWARNING: Your dataset is inflated by {(duplicate_count/total_rows)*100:.1f}% due to duplicated rows.")
    print(f"Actual unique chunks: {total_rows - duplicate_count}")
else:
    print("\nSUCCESS: No duplicates found. Your chunk count is legitimate.")