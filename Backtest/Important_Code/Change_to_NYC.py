import pandas as pd
from pathlib import Path

# -------- Configuration --------
INPUT_PATH  = Path(r"C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_60M.csv")
OUTPUT_PATH = INPUT_PATH.with_name(INPUT_PATH.stem + "_NYC.csv")   # e.g. NQ1!_MAIN_1M_NYC.csv
DATETIME_COL = "datetime"   # name of the column to convert
NEW_COL_NAME = "datetime"  # name for the converted column
TIMEZONE = "America/New_York"
# --------------------------------

def main():
    # Read CSV. Setting low_memory=False avoids mixed-type inference issues.
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    if DATETIME_COL not in df.columns:
        raise ValueError(f"Column '{DATETIME_COL}' not found in the CSV.")

    # Parse the datetime column. `utc=True` will localize naive timestamps; if they
    # already have '+00:00' they will be understood as UTC automatically.
    # `errors='coerce'` will turn unparseable rows into NaT so you can filter or inspect.
    dt_utc = pd.to_datetime(df[DATETIME_COL], utc=True, errors='coerce')

    # Report any rows that failed to parse (optional)
    bad_rows = dt_utc.isna().sum()
    if bad_rows > 0:
        print(f"Warning: {bad_rows} rows in '{DATETIME_COL}' could not be parsed and became NaT.")

    # Convert to America/New_York (handles DST automatically)
    dt_ny = dt_utc.dt.tz_convert(TIMEZONE)

    # Store as ISO 8601 string including offset, e.g. 2024-11-15 10:15:00-05:00
    df[NEW_COL_NAME] = dt_ny.astype(str)

    # If you would rather overwrite the original column instead of adding a new one,
    # uncomment the next line and optionally drop the extra column:
    # df[DATETIME_COL] = df[NEW_COL_NAME]; df.drop(columns=[NEW_COL_NAME], inplace=True); NEW_COL_NAME = DATETIME_COL

    # Save to new CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Conversion complete. Saved file with NYC times to:\n{OUTPUT_PATH}")

    # Quick sanity check: show first few converted rows
    print(df[[DATETIME_COL, NEW_COL_NAME]].head())

if __name__ == "__main__":
    main()
