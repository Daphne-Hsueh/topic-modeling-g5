import pandas as pd
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_TO_CHECK = {
    "STEP 1: RAW DATA (step1_corpus.csv)": BASE_DIR / "data" / "step1_corpus.csv",
    "STEP 2: CHUNKS (step2_chunks_clean.csv)": BASE_DIR / "data" / "processed" / "step2_chunks_clean.csv",
    "STEP 3: FILTERED (step2_filtered_corpus.csv)": BASE_DIR / "data" / "processed" / "step2_filtered_corpus.csv"
}

print("\n" + "="*50)
print("🔍 DATA PIPELINE VERIFICATION")
print("="*50)

for name, path in FILES_TO_CHECK.items():
    print(f"\n{name}")
    
    if not path.exists():
        print("❌ STATUS: FILE MISSING")
        print("   -> Action: You need to run the script/notebook that creates this file.")
        continue
        
    try:
        df = pd.read_csv(path, low_memory=False)
        print("✅ STATUS: FILE FOUND")
        print(f"   Total Rows: {len(df):,}")
        
        # Check for unique companies (ticker or cik)
        if 'ticker' in df.columns:
            print(f"   Unique Companies: {df['ticker'].nunique()}")
        elif 'cik' in df.columns:
            print(f"   Unique Companies: {df['cik'].nunique()}")
        else:
            print("   Unique Companies: Unknown (No ticker/cik column found)")
            
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")

print("\n" + "="*50 + "\n")