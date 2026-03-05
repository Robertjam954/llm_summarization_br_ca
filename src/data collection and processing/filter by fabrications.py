import pandas as pd

FILE = r"C:\Users\jamesr4\loc\data_private\raw\merged_llm_summary_validation_datasheet_deidentified.xlsx"

df = pd.read_excel(FILE)

# Find columns with 'ai' or 'human' in their name (case-insensitive)
target_cols = [c for c in df.columns if 'ai' in str(c).lower() or 'human' in str(c).lower()]
print(f"Columns found: {target_cols}\n")

# Count rows where ANY of those columns has value 3
mask = df[target_cols].isin([3]).any(axis=1)
print(f"Rows with value 3 in at least one ai/human column: {mask.sum()}")

# Also show per-column breakdown
print("\nPer-column breakdown:")
for col in target_cols:
    count = (df[col] == 3).sum()
    print(f"  {col}: {count} rows with value 3")