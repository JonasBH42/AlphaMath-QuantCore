from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats

# ------------- 0. ASSUMPTION -------------
# df is your DataFrame, indexed by timestamp, with columns ['open','high','low','close']
# -----------------------------------------


# 1. DEFINE VARIABLES & EMA
 # you can change this
# -*- coding: utf-8 -*-
"""
Speed‑Cross Statistical Workflow
================================
Python script that implements steps 1 through 6 of the workflow outlined
for a momentum‑plus‑mean‑reversion strategy on 1‑minute Nasdaq (NQ)
futures data.

Assumptions
-----------
* You already have a `pandas.DataFrame` called ``df`` with datetime index
  (timezone‑aware) and OHLC columns: ``open``, ``high``, ``low``,
  ``close``.
* A variable ``ema`` (int) defines the EMA length you wish to examine.
  If you want to sweep a grid, just populate ``ema_lengths`` with several
  integers.
* All time units are **bars** ( = 1 minute ).
* Transaction costs are ignored at this stage.

External packages required
--------------------------
```bash
pip install pandas numpy scipy statsmodels tqdm
```

How to use
----------
```python
from nq_speed_cross_analysis import run_full_workflow
result_dict = run_full_workflow(df, ema_lengths=[9, 13, 21],
                               speed_quantile=0.85,
                               forward_horizons=[5, 10, 20])
```
Each entry in ``result_dict`` contains:
    * stationarity test results (ADF & KPSS)
    * OU‑fit κ, σ, half‑life
    * crossing‑speed threshold θ
    * conditional forward‑return stats (mean, t‑stat, p‑value)
    * calibrated TP recommendation (distance and hit‑rate)
    * suggested averaging‑down z‑score levels

Step 7 (walk‑forward validation) is **not** included by design.
"""



import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    adf_stat: float
    adf_p: float
    kpss_stat: float
    kpss_p: float

@dataclass
class OUParams:
    phi: float  # discrete mean‑reversion coef
    kappa: float  # continuous‑time speed (≈ −ln(phi))
    sigma_eps: float  # residual std
    half_life: float  # bars

@dataclass
class EventStudyStats:
    horizon: int
    mean_ret: float
    t_stat: float
    p_value: float
    n_obs: int

@dataclass
class TPRecommendation:
    tp_distance: float  # absolute price distance
    hit_rate: float

@dataclass
class AveragingLevels:
    z_scores: List[float]
    revert_probs: List[float]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentially‑weighted moving average (Pandas wrapper)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def stationarity_tests(spread: pd.Series) -> StationarityResult:
    """Run ADF and KPSS tests on a spread series."""
    # ADF: H0 = unit root (non‑stationary)
    adf_out = adfuller(spread.dropna(), autolag="AIC")
    # KPSS: H0 = level‑stationary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_out = kpss(spread.dropna(), regression="c", nlags="auto")
    return StationarityResult(adf_stat=adf_out[0], adf_p=adf_out[1],
                              kpss_stat=kpss_out[0], kpss_p=kpss_out[1])


def fit_ou_params(spread: pd.Series) -> OUParams:
    """Fit discrete OU: S_{t+1} = φ S_t + ε_t and derive κ, σ, half‑life."""
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    s_now = s.loc[s_lag.index]
    # OLS φ
    phi = np.polyfit(s_lag.values, s_now.values, 1)[0]
    residuals = s_now.values - phi * s_lag.values
    sigma_eps = residuals.std(ddof=1)
    # continuous‑time κ approximation and half‑life
    if phi <= 0 or phi >= 1:
        kappa = np.nan
        half_life = np.inf
    else:
        kappa = -np.log(phi)
        half_life = np.log(2) / kappa
    return OUParams(phi=phi, kappa=kappa, sigma_eps=sigma_eps, half_life=half_life)


def compute_cross_speed(spread: pd.Series, normalize: bool = True) -> pd.Series:
    """Speed = first difference of spread, optionally z‑scored."""
    v = spread.diff()
    if normalize:
        std = v.rolling(60, min_periods=30).std()
        v = v / std
    return v


def get_speed_threshold(speed: pd.Series, quantile: float = 0.85) -> float:
    """Return symmetric threshold θ s.t. |v| > θ occurs in upper (1‑q) fraction."""
    return speed.abs().quantile(quantile)


def forward_returns(close: pd.Series, horizons: List[int]) -> Dict[int, pd.Series]:
    """Compute forward percentage returns for multiple horizons."""
    fr = {h: close.shift(-h) / close - 1.0 for h in horizons}
    return fr


def event_study(speed: pd.Series, close: pd.Series, theta: float,
                horizons: List[int]) -> List[EventStudyStats]:
    """Conditional expectancy of forward returns given |speed|>θ."""
    mask = speed.abs() > theta
    stats_list: List[EventStudyStats] = []
    fr = forward_returns(close, horizons)
    for h in horizons:
        r = fr[h].where(mask)
        r_non_missing = r.dropna()
        if len(r_non_missing) < 20:
            mean_ret = t_stat = p_val = np.nan
        else:
            mean_ret = r_non_missing.mean()
            t_stat, p_val = stats.ttest_1samp(r_non_missing.values, 0.0, nan_policy="omit")
        stats_list.append(EventStudyStats(horizon=h, mean_ret=mean_ret,
                                          t_stat=t_stat, p_value=p_val,
                                          n_obs=len(r_non_missing)))
    return stats_list


def calibrate_tp(close: pd.Series, entries: pd.Series,
                 lookahead: int = 20, quantile: Tuple[float, float] = (0.6, 0.7)) -> TPRecommendation:
    """Choose TP at 60–70th percentile of MFE up to *lookahead* bars."""
    entry_idx = entries[entries].index
    if len(entry_idx) == 0:
        return TPRecommendation(tp_distance=np.nan, hit_rate=np.nan)
    mfe = []
    hit = []
    for t in entry_idx:
        entry_px = close.at[t]
        window = close.loc[t:t + pd.Timedelta(minutes=lookahead - 1)]
        max_excursion = (window.max() - entry_px)
        mfe.append(max_excursion)
    mfe = np.array(mfe)
    if len(mfe) == 0:
        return TPRecommendation(tp_distance=np.nan, hit_rate=np.nan)
    tp = np.quantile(mfe, quantile)
    tp_distance = tp.mean()  # mid‑point of 60–70th percentile
    # Estimate hit‑rate
    for dist in mfe:
        hit.append(dist >= tp_distance)
    hit_rate = np.mean(hit)
    return TPRecommendation(tp_distance=float(tp_distance), hit_rate=float(hit_rate))


def derive_averaging_levels(spread: pd.Series, entries: pd.Series,
                            z_candidates: List[float] = None) -> AveragingLevels:
    """Pick z‑score bands where reversion probability > 55 %."""
    if z_candidates is None:
        z_candidates = [-1.0, -1.8, -2.5]
    s = spread.dropna()
    z = (s - s.mean()) / s.std(ddof=0)
    probs = []
    for z_i in z_candidates:
        mask = entries & (z <= z_i)
        idx = z[mask].index
        succ = 0
        for t in idx:
            # Did we revert to 0 before next band? Check 2×|z_i| bars ahead
            window = z.loc[t:t + pd.Timedelta(minutes=int(abs(z_i) * 2))]
            if (window >= 0).any():
                succ += 1
        prob = succ / len(idx) if len(idx) else np.nan
        probs.append(prob)
    return AveragingLevels(z_scores=z_candidates, revert_probs=probs)

# ---------------------------------------------------------------------------
# Master driver
# ---------------------------------------------------------------------------

def run_full_workflow(df: pd.DataFrame,
                      ema_lengths: List[int],
                      speed_quantile: float = 0.85,
                      forward_horizons: List[int] = (5, 10, 20)) -> Dict[int, Dict]:
    """Run steps 1‑6 across one or many EMA lengths.

    Returns a nested dict keyed by EMA length.
    """
    results: Dict[int, Dict] = {}
    close = df["close"].copy()

    for L in tqdm(ema_lengths, desc="EMA lengths"):
        ema_series = compute_ema(close, span=L)
        spread = close - ema_series

        # Step 2: stationarity & OU fit
        stat_res = stationarity_tests(spread)
        ou_res = fit_ou_params(spread)

        # Skip if non‑stationary or half‑life too long (> 4×L)
        if (stat_res.adf_p > 0.05) or (ou_res.half_life > 4 * L):
            results[L] = {"stationarity": stat_res, "ou": ou_res, "skipped": True}
            continue

        # Step 3: speed and threshold
        speed = compute_cross_speed(spread)
        theta = get_speed_threshold(speed, speed_quantile)

        # Step 4: event study
        evt_stats = event_study(speed, close, theta, list(forward_horizons))

        # Entry mask
        entries = (speed.abs() > theta)

        # Step 5: TP calibration
        tp_rec = calibrate_tp(close, entries, lookahead=int(ou_res.half_life))

        # Step 6: averaging levels
        avg_levels = derive_averaging_levels(spread, entries)

        # Collate
        results[L] = {
            "stationarity": stat_res,
            "ou": ou_res,
            "theta": float(theta),
            "event_stats": evt_stats,
            "tp": tp_rec,
            "averaging": avg_levels,
            "skipped": False,
        }

    return results

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    import json
    import sys

    df = pd.read_csv('C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/NQ1!_MAIN_1M.csv', parse_dates=True, index_col=0)
    df.drop(columns=["rtype", "publisher_id", "instrument_id", "symbol"], inplace=True)
    df = df.iloc[int(len(df) * 0.5):]
    if "df" not in globals():
        sys.exit("Please load a DataFrame `df` with NQ 1‑min data before running this script.")

    ema_lengths = [5, 8, 13, 21, 55]
    output = run_full_workflow(df, ema_lengths=ema_lengths)

    # Pretty‑print a summary for the first EMA length
    first_L = ema_lengths[0]
    print("\n=== Summary for EMA length", first_L, "===")
    print(json.dumps(output[first_L], default=lambda o: o.__dict__, indent=2))