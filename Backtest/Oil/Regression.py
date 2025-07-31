import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIG ===
CSV_PATH = r"C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/Crude_Oil_1D.csv"
DXY_PATH = r"C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/DXY.csv"
TARGET_COL = "close"
OPEN_COL = "open"
OUTPUT_DIR = r"C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Crude_Oil_Lasso_predictions.csv")

# === LOAD ===
df = pd.read_csv(CSV_PATH)
dxy = pd.read_csv(DXY_PATH, parse_dates=["datetime"])

# === FEATURE ENGINEERING PLACEHOLDER ===
# === YOU SHOULD ADD YOUR FEATURES HERE ===
# Example:
df["trend_z"] = (df[TARGET_COL] - df[TARGET_COL].rolling(20).mean()) / df[TARGET_COL].rolling(20).std().shift(1)
df["return"] = df[TARGET_COL].pct_change().shift(1)
df["lag_close"] = df[TARGET_COL].shift(1)
df["EMA_20"] = df[TARGET_COL].ewm(span=20, adjust=False).mean()
# df["DXY"] = df["datetime"].map(dxy.set_index("datetime")["close"])

df = df.dropna()

# After adding features, drop NaNs introduced by rolling/shifting
df = df.dropna().reset_index(drop=True)

# === SELECT FEATURES AND TARGET ===
# Automatically take everything except target and open as features
feature_cols = ["trend_z", "return", "lag_close", "EMA_20", ]
if not feature_cols:
    raise RuntimeError("No feature columns detected. Add features before running (see placeholder).")

X = df[feature_cols]
y = df[TARGET_COL]
open_vals = df[OPEN_COL]

# === SPLIT ===
X_train, X_test, y_train, y_test, open_train, open_test = train_test_split(
    X, y, open_vals, train_size=0.8, shuffle=True, random_state=42
)

# === SCALE ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# === FIT LASSO ===
lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
lasso.fit(X_train_s, y_train_s)

# === PREDICT ===
y_pred_s = lasso.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

# === BUILD RESULTS DF ===
results_df = pd.DataFrame({
    "actual_close": y_test.reset_index(drop=True),
    "predicted_close": y_pred,
    "open": open_test.reset_index(drop=True)
})

# === DIRECTIONAL VALUE ===
def directional_value(row):
    delta = row["actual_close"] - row["open"]
    if row["predicted_close"] < row["open"]:
        return -delta
    else:
        return delta

results_df["directional_value"] = results_df.apply(directional_value, axis=1)
directional_positive = (results_df["directional_value"] > 0).sum()
total = len(results_df)

# === METRICS ===
rmse = np.sqrt(mean_squared_error(results_df["actual_close"], results_df["predicted_close"]))
r2 = r2_score(results_df["actual_close"], results_df["predicted_close"])

# === OUTPUT ===
print(f"Test sample count: {total}")
print(f"Directional positive count: {directional_positive} / {total}")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.6f}")

# === PLOT ===
plt.figure(figsize=(10, 5))
plt.plot(results_df["actual_close"].values, label="Actual Close", linewidth=1)
plt.plot(results_df["predicted_close"].values, label="Predicted Close", linewidth=1)
plt.title("Lasso: Predicted vs Actual Close (Test Set)")
plt.xlabel("Test Sample Index")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === SAVE ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved predictions to: {OUTPUT_FILE}")
