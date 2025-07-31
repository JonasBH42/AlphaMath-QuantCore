import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1.  configuration
# --------------------------------------------------
ROOT = Path("C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs")
SRC_1M = ROOT / "NQ1!_MAIN_1M_NYC.csv"           # 1‑minute source
NAME   = "NQ1!_MAIN"                         # stem used for outputs

# --------------------------------------------------
# 2.  load 1‑minute bars
# --------------------------------------------------
df_1m = (
    pd.read_csv(SRC_1M, parse_dates=["ts_event"])   # read + parse timestamp
      .rename(columns={"ts_event": "datetime"})     # nicer index name
      .set_index("datetime")
      .sort_index()
)

# --------------------------------------------------
# 3.  helper to resample OHLCV
# --------------------------------------------------
def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame using the given pandas offset alias.
    Keeps right‑label/right‑closed bars (i.e., timestamp = bar close).
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, format='mixed', utc=None)
        
    agg = {
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }
    return (
        frame
        .resample(rule, label="right", closed="right")
        .agg(agg)
        .dropna(how="any")      # drop incomplete bars (e.g. last partial bar)
    )

# --------------------------------------------------
# 4.  build each higher‑timeframe DataFrame
# --------------------------------------------------
df_5m  = resample_ohlcv(df_1m,  "5T")      # 5‑minute from 1‑minute
df_10m = resample_ohlcv(df_5m,  "10T")     # 10‑minute from 5‑minute
df_15m = resample_ohlcv(df_5m,  "15T")     # 15‑minute from 5‑minute
df_30m = resample_ohlcv(df_15m, "30T")     # 30‑minute from 15‑minute
df_60m = resample_ohlcv(df_30m, "60T")     # 60‑minute from 30‑minute

# --------------------------------------------------
# 5.  save everything
# --------------------------------------------------
out_files = {
    "5M" : df_5m,
    "10M": df_10m,
    "15M": df_15m,
    "30M": df_30m,
    "60M": df_60m,
}

for suffix, frame in out_files.items():
    dst = ROOT / f"{NAME}_{suffix}.csv"
    frame.to_csv(dst, index_label="datetime")
    print(f"Saved {dst}")

