import pandas as pd
import numpy as np
from typing import List

INPUT_PATH = 'assets/random_cohort_no_wound.csv'
OUTPUT_PATH = 'assets/random_cohort_no_wound_jittered.csv'

def jitter_dates_and_zip(df: pd.DataFrame, date_cols: List[str], zipcode_col: str = 'Zipcode', max_days: int = 3, max_zip_units: int = 3) -> pd.DataFrame:
    out = df.copy()

    # Jitter date columns by +/- max_days (vectorized per column)
    for col in date_cols:
        try:
            parsed = pd.to_datetime(out[col], errors='coerce')
        except Exception:
            continue

        # Generate per-row jitter in days
        days_delta = pd.to_timedelta(np.random.randint(-max_days, max_days + 1, size=len(out)), unit='D')
        jittered = parsed + days_delta

        # Keep original value if parsing failed (NaT)
        out[col] = np.where(parsed.notna(), jittered.dt.strftime('%Y/%m/%d'), out[col])

    # Jitter zipcode by +/- max_zip_units (first 5 digits only)
    if zipcode_col in out.columns:
        def jitter_zip(value):
            s = str(value)
            # Extract first 5 digits
            digits = ''.join(ch for ch in s if ch.isdigit())
            if len(digits) < 5:
                return value  # leave as-is if no 5-digit base
            base = int(digits[:5])
            delta = int(np.random.randint(-max_zip_units, max_zip_units + 1))
            new_zip = max(1, min(99999, base + delta))
            return f"{new_zip:05d}"

        out[zipcode_col] = out[zipcode_col].apply(jitter_zip)

    return out

def main():
    # Read input (preserve index if present)
    df = pd.read_csv(INPUT_PATH)

    # Identify date columns by name substring (case-insensitive)
    date_cols = [c for c in df.columns if 'date' in str(c).lower()]

    jittered = jitter_dates_and_zip(df, date_cols)
    jittered.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote jittered dataset to: {OUTPUT_PATH}")
    if date_cols:
        print(f"Date columns jittered (+/-3 days): {', '.join(date_cols)}")
    if 'Zipcode' in df.columns:
        print("Zipcode jittered (+/-3 units).")

if __name__ == '__main__':
    main()

