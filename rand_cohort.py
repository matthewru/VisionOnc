import csv
import pandas as pd

# Read the CSV file
df = pd.read_csv('assets/2025_02_20_13_05_32.csv', index_col='Enrolled Patient #')

# Optional: set a seed for reproducibility; set to None for fully random
SEED = 42

# Explicit wound-related columns per user
explicit_cols = [
    'Major Wound Complications ',
    'Major Wound Complications AK + FCE',
    'Wound complication type',
    'Surgical Wound Complications',
]

available_cols = [c for c in explicit_cols if c in df.columns]
if not available_cols:
    raise ValueError("None of the specified wound complication columns were found in the dataset.")

def is_positive_indicator(series: pd.Series) -> pd.Series:
    v = series.astype(str).str.strip().str.lower()
    # Treat only clear negatives as no-complication; everything else flagged as complication
    positive_tokens = {'yes', 'y', 'true', '1'}
    negative_tokens = {'no', 'n', 'false', '0'}
    return (
        v.isin(positive_tokens)
        | (~v.isin(positive_tokens | negative_tokens) & v.ne(''))  # unknowns and other values => flag
    )

def has_type_value(series: pd.Series) -> pd.Series:
    v = series.astype(str).str.strip().str.lower()
    empties = {'', 'no', 'none', 'na', 'n/a', 'nan'}
    return ~v.isin(empties)

# Aggregate any-complication mask across the provided columns
has_complication = pd.Series(False, index=df.index)
for col in available_cols:
    if col == 'Wound complication type':
        has_complication = has_complication | has_type_value(df[col])
    else:
        has_complication = has_complication | is_positive_indicator(df[col])

# Eligible = rows with no complications in any of the columns
eligible = df[~has_complication]

if eligible.empty:
    raise ValueError("No rows without wound complications were found.")

n = min(20, len(eligible))
sampled = eligible.sample(n=n, random_state=SEED)

# Save and print results
out_path = 'assets/random_cohort_no_wound.csv'
sampled.to_csv(out_path)

print(f"Selected {n} patients without wound complications (out of {len(eligible)} eligible). Saved to: {out_path}")
print("Selected patient IDs:")
for pid in sampled.index.tolist():
    print(pid)

