"""
Feature Diagnostics for Daily Crude-Oil OHLCV Data
--------------------------------------------------
•  Univariate tests:
      – Point-Biserial correlation
      – Mutual Information
      – Mann–Whitney-U (distribution shift)
      – Raw directional hit-rate
•  Single-feature walk-forward Logit AUC
•  One-line “open-to-close” back-test & Sharpe
•  Multiple-testing correction (Benjamini–Hochberg FDR)
•  Multivariate L1-penalised Logit selector
•  Optional rolling-window stability plots (commented)

Requirements:
    pip install pandas numpy scipy scikit-learn statsmodels matplotlib
"""

import warnings, math
# warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from statsmodels.stats.multitest import multipletests
# -------------------------------------------------------------

CSV_PATH = r"C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/Crude_Oil_1D.csv"
DATE_COL  = "datetime"            # <-- adjust if necessary
OPEN_COL  = "open"
CLOSE_COL = "close"

# -------------------------------------------------------------
def load_data(path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    #  create binary “direction” target: 1 = up-day, 0 = down/flat
    df["direction"] = (df[CLOSE_COL] > df[OPEN_COL]).astype(int)
    return df


# -----------------------------------------------------------------
def walkforward_auc(feature: pd.Series, y: pd.Series, n_splits=5) -> float:
    X = feature.values.reshape(-1, 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train, test in tscv.split(X):
        clf = LogisticRegression(max_iter=200, solver="liblinear").fit(X[train], y.iloc[train])
        proba = clf.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test], proba))
    return float(np.mean(aucs))


def backtest_sharpe(feature: pd.Series, df: pd.DataFrame, annual_factor=math.sqrt(252)) -> float:
    """One-line trade: enter on sign(feature) at open, exit at close next bar."""
    signal = np.sign(feature).shift(1)          # decide at previous bar close
    pnl = signal * (df[CLOSE_COL] - df[OPEN_COL])
    if pnl.std() == 0:
        return 0.0
    daily_sr = pnl.mean() / pnl.std()
    return daily_sr * annual_factor


def evaluate_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    y = df["direction"]
    results = []
    pvals_mwu = []

    for col in features.columns:
        f = features[col].dropna()
        common_idx = y.index.intersection(f.index)
        f, y_ = f.loc[common_idx], y.loc[common_idx]

        # ------------ univariate statistics
        pb_r, pb_p = pointbiserialr(f, y_)
        mi = mutual_info_classif(f.values.reshape(-1, 1), y_, discrete_features=False)[0]

        up_grp, dn_grp = f[y_ == 1], f[y_ == 0]
        mwu_stat, mwu_p = mannwhitneyu(up_grp, dn_grp, alternative="two-sided")
        pvals_mwu.append(mwu_p)

        hit = accuracy_score(y_, (f > 0).astype(int))

        # ------------ single-feature CV AUC
        auc = walkforward_auc(f, y_)

        # ------------ economic value
        sharpe = backtest_sharpe(f, df.loc[f.index])

        results.append(dict(
            feature=col,
            point_biserial_r=pb_r,
            point_biserial_p=pb_p,
            mutual_info=mi,
            mannwhitney_p=mwu_p,
            hit_rate=hit,
            cv_auc=auc,
            sharpe=sharpe
        ))

    res = pd.DataFrame(results).set_index("feature")

    # ------ multiple-testing correction on MW-U p-values
    _, qvals, _, _ = multipletests(pvals_mwu, method="fdr_bh")
    res["mwu_q"] = qvals
    return res.sort_values("mwu_q")


# -----------------------------------------------------------------
def multivariate_l1_selector(features: pd.DataFrame, y: pd.Series, n_splits=5):
    """Returns list of selected column names via L1-penalised Logit."""
    common_idx = y.index.intersection(features.dropna(how="all").index)
    X, y_ = features.loc[common_idx].fillna(0), y.loc[common_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = LogisticRegressionCV(
        Cs=20, cv=tscv, penalty="l1", solver="liblinear",
        max_iter=500, scoring="roc_auc"
    ).fit(X, y_)
    non_zero = np.where(clf.coef_.ravel() != 0)[0]
    return list(X.columns[non_zero])

# -----------------------------------------------------------------
# Optional: rolling-window stability plot (uncomment if desired)

# -----------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    # --- EXAMPLE FEATURE ENGINEERING (DELETE / EXTEND AS YOU WISH) -------
    features = pd.DataFrame(index=df.index)
    features["trend_z"] = (df[CLOSE_COL] - df[CLOSE_COL].rolling(20).mean()) / df[CLOSE_COL].rolling(20).std()
    features["return"] = df[CLOSE_COL].diff().fillna(0)
    features["roll_std_20"] = df[CLOSE_COL].rolling(20).std()

    # ---------------------------------------------------------------------
    stats_table = evaluate_features(features, df)

    print("\n=== Univariate Feature Diagnostics ===")
    print(stats_table.round(4))
    # for idx, row in stats_table.iterrows():
    #     print(f"{idx}: {row.tolist()}")

    selected = multivariate_l1_selector(features, df["direction"])
    print("\nSelected by L1-Logit:", selected)
