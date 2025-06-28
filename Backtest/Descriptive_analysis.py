import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression


s_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv")
main_df = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv")
# -------------------------------
# 1. DESCRIPTIVE ANALYSIS & EXPLORATION
# -------------------------------

# Assume s_df is your DataFrame loaded from the CSV.
# For example: s_df = pd.read_csv('your_file.csv')

# Inspect the data
print("Data Head:")
print(s_df.head())
print("\nData Info:")
print(s_df.info())
print("\nStatistical Summary:")
print(s_df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(s_df.isnull().sum())

# -------------------------------
# 2. VISUALIZATION OF DISTRIBUTIONS & CORRELATIONS
# -------------------------------

# Plot the distribution for key numerical columns
num_cols = ['open', 'high', 'low', 'close', 'volume']
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(s_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot a heatmap of correlations (only for the numeric columns)
plt.figure(figsize=(8, 6))
corr_matrix = s_df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# -------------------------------
# 3. STATISTICAL TESTS (Stationarity Test)
# -------------------------------

# Augmented Dickey-Fuller test on the 'close' column
adf_result = adfuller(s_df['close'])
print("\nAugmented Dickey-Fuller Test on 'close':")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("The 'close' series appears stationary.")
else:
    print("The 'close' series appears non-stationary.")

# -------------------------------
# 4. FEATURE ENGINEERING (New Features for the Model)
# -------------------------------

# Example engineered features:
# a) High-Low range
s_df['hl_range'] = s_df['high'] - s_df['low']

# b) Open-Close difference (can be signed or absolute)
s_df['oc_range'] = s_df['close'] - s_df['open']
s_df['body_size'] = abs(s_df['oc_range'])

# c) Volatility as a percentage of the open price
s_df['volatility'] = s_df['hl_range'] / s_df['open']

# d) Percentage change in close (Note: first row will be NaN)
s_df['close_pct_change'] = s_df['close'].pct_change()

# -------------------------------
# 5. CREATING LAGGED FEATURES (Lag = 15 rows)
# -------------------------------

lag = 15

# We will create lagged features for every column except the timestamp.
# (We want only historical values to be used for prediction.)
cols_to_lag = [col for col in s_df.columns if col != 'ts_event']
for col in cols_to_lag:
    lag_col_name = f"{col}_lag{lag}"
    s_df[lag_col_name] = s_df[col].shift(lag)

# -------------------------------
# 6. DROP THE TIMESTAMP COLUMN
# -------------------------------

# s_df.drop(columns=['ts_event'], inplace=True)

# -------------------------------
# 7. CLEANING DATA: DROP ROWS WITH NA VALUES
# -------------------------------

# Because of the percentage change and lagging, the first several rows will have NaN values.
s_df.dropna(inplace=True)
s_df.reset_index(drop=True, inplace=True)

# -------------------------------
# 8. SELECT FEATURES FOR MODELING
# -------------------------------

# Our target is the current 'close' value.
# We want to use only historical (lagged) features for prediction,
# to avoid lookahead bias.

# Choose only the lagged features (i.e. those ending with '_lag15') as inputs.
feature_cols = [col for col in s_df.columns if col.endswith(f"_lag{lag}")]
feature_cols.append("ts_event")

# Print out the feature columns selected.
print("\nFeatures selected for modeling:")
print(feature_cols)

# Define X (features) and y (target)
X = s_df[feature_cols]
y = s_df['close']  # The current (today's) close

# -------------------------------
# 9. SPLIT DATA INTO TRAINING AND TEST SETS (80% train, 20% test, in order)
# -------------------------------

# Since this is time series data, we do not shuffle the rows.
train_size = int(len(s_df) * 0.8)
X_train = X.iloc[:train_size].copy()
y_train = y.iloc[:train_size].copy()
X_test = X.iloc[train_size:].copy()
y_test = y.iloc[train_size:].copy()

print(f"\nTraining set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

X_test_indices = X_test["ts_event"]
X_train = X_train.drop(columns=["ts_event"])
X_test = X_test.drop(columns=["ts_event"])

# -------------------------------
# 10. TRAIN THE LINEAR REGRESSION MODEL
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 11. MAKE PREDICTIONS ON THE TEST SET
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# 12. CREATE A RESULTS DATA FRAME
# -------------------------------

results_df = pd.DataFrame({
    "ts_event": X_test_indices,
    'close': y_test.values,
    'prediction': y_pred
})
results_df = results_df.merge(main_df[["ts_event", "open", "high", "low"]], on="ts_event", how="left")

def inverse_scale(df_scaled: pd.DataFrame, min_low, max_high, params) -> pd.DataFrame:
    df_original = df_scaled.copy()
    for param in params:
        df_original[param] = df_scaled[param] / 100 * (max_high - min_low) + min_low
    return df_original


descaled_df = inverse_scale(
    results_df, main_df["low"].min(), main_df["high"].max(), ["prediction", "close"]
)


print("\nSample of predictions vs. actuals:")


win = 0
loss = 0
total_predictions = 0
profit = 0
losess = 0

for index, row in descaled_df.iterrows():
    if not pd.isnull(row["prediction"]):
        if row["prediction"] > row["open"]:
            if (row["close"] - row["open"]) > 0:
                profit += abs(row["close"] - row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) < 0:
                losess += abs(row["close"] - row["open"]) + 1
                loss += 1
                total_predictions += 1
        elif row["prediction"] < row["open"]:
            if (row["close"] - row["open"]) < 0:
                profit += abs(row["close"] - row["open"]) - 1
                win += 1
                total_predictions += 1
            elif (row["close"] - row["open"]) > 0:
                losess += abs(row["close"] - row["open"]) + 1
                loss += 1
                total_predictions += 1


average_profit = profit / total_predictions
average_loss = losess / total_predictions
risk_reward_ratio = average_profit / average_loss
print("Evaluation of Predictions:")
print(f"Wining Ratio: {win/total_predictions:.3%}")
print(f"Risk Reward Ratio: {risk_reward_ratio:.3f}")
print(f"Total: {profit-losess}")

