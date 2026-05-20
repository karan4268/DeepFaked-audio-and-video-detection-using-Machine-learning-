# check_itw_meta.py
import pandas as pd
import os

# Point this at wherever you unzipped it
ITW_ROOT = r"F:\Experiments\Mtech Project DeepFaked Video and Audio analyzer\datasets\in_the_wild\release_in_the_wild"

meta_path = os.path.join(ITW_ROOT, "meta.csv")
df = pd.read_csv(meta_path)

print("=== meta.csv ===")
print(f"Columns : {df.columns.tolist()}")
print(f"Rows    : {len(df)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLabel values  : {df['label'].unique()}")
print(f"Label counts  :\n{df['label'].value_counts()}")
print(f"\nFile sample   : {df['file'].values[:3]}")

if "speaker" in df.columns:
    print(f"\nSpeakers      : {df['speaker'].nunique()} unique")
    print(f"Sample names  : {df['speaker'].unique()[:8]}")
else:
    # Extract from filename
    df["speaker"] = (df["file"]
                     .str.split("-", n=1).str[1]
                     .str.rsplit(".", n=1).str[0])
    print(f"\nNo speaker column — extracted from filename")
    print(f"Speakers      : {df['speaker'].nunique()} unique")
    print(f"Sample names  : {df['speaker'].unique()[:8]}")

# Check audio format
exts = df["file"].str.split(".").str[-1].unique()
print(f"\nAudio format  : {exts}")

# Check if files are flat or in subfolders
sample_file = df["file"].values[0]
print(f"\nPath format   : {sample_file}")
has_subdir = "/" in sample_file or "\\" in sample_file
print(f"Has subfolders: {has_subdir}")
