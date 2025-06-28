import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# ----- Load Data -----
# Read CSV file and remove any NA rows
data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv"
data = pd.read_csv(data_path)
data = data.dropna()

# Ensure 'Close' is numeric
data["close"] = pd.to_numeric(data["close"], errors="coerce")

# ----- Calculate EMAs -----
# Using pandas.ewm to calculate Exponential Moving Averages (EMA)
data["EMA_5"] = data["close"].ewm(span=5, adjust=False).mean()
data["EMA_20"] = data["close"].ewm(span=20, adjust=False).mean()
data["EMA_9"] = data["close"].ewm(span=9, adjust=False).mean()
data["Return"] = data["close"] - data["open"]

# ----- Set up lagged values and target -----
# Assumes that the data has columns "Open" and "Return".
# Create new columns similar to R's mutate:
lagged_data_1 = data.copy()
lagged_data_1["Close_Lead"] = lagged_data_1["close"].shift(-1)  # lead by 1 (next row)
# 'Lagged_Return' is taken directly from the existing 'Return' column.
lagged_data_1["Lagged_Return"] = lagged_data_1["Return"].shift(1)
lagged_data_1["EMA_5_Lag1"] = lagged_data_1["EMA_5"].shift(1)
lagged_data_1["EMA_20_Lag1"] = lagged_data_1["EMA_20"].shift(1)
lagged_data_1["EMA_9_Lag1"] = lagged_data_1["EMA_9"].shift(1)
lagged_data_1["Ema_difference"] = lagged_data_1["close"] - lagged_data_1["EMA_5"]
lagged_data_1["Midpoint"] = (lagged_data_1["open"] + lagged_data_1["close"]) / 2
lagged_data_1["Open_weighted_price"] = lagged_data_1["open"] + (
    lagged_data_1["close"] / 2
)
lagged_data_1["Close_weighted_Price"] = lagged_data_1["close"] + (
    lagged_data_1["open"] / 2
)

# Remove any rows with NA values
lagged_data = lagged_data_1.dropna()

# ----- Write subset to CSV and Reload it -----
output_subset_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/AUDUSD_15min.csv"
lagged_data.to_csv(output_subset_path, index=False)
subset = pd.read_csv(output_subset_path)

# ----- Split data for training and test -----
# Note: In R, rows 1:9326 are training and 9327:13999 are test.
# In Python, indices are zero-based.
data_train = subset.iloc[:9326].copy()
test_data = subset.iloc[9326:13999].copy()

# ----- Prepare data for Lasso regression -----
feature_cols = [
    "Lagged_Return",
    "Midpoint",
    "Open_weighted_price",
    "Close_weighted_Price",
    "Ema_difference",
    "EMA_5",
    "EMA_20",
    "EMA_9",
]

X_train = data_train[feature_cols].values
Y_train = data_train["Close_Lead"].values

# Remove rows with any NA values in X_train or Y_train
mask = np.all(np.isfinite(X_train), axis=1) & np.isfinite(Y_train)
X_train = X_train[mask]
Y_train = Y_train[mask]

# ----- Train the Lasso regression model -----
# LassoCV performs cross-validation to select the best lambda (alpha in scikit-learn)
lasso_model = LassoCV(cv=10, random_state=0).fit(X_train, Y_train)
best_lambda = lasso_model.alpha_
print("Optimal Lambda Value:", best_lambda)

# ----- Run predictions on the test data -----
X_test = test_data[feature_cols].values

# Check if X_test has valid samples
if X_test.shape[0] == 0:
    raise ValueError(
        "Test data contains no valid samples. Please check the input data."
    )

predictions = lasso_model.predict(X_test)
test_data["Predicted_Close"] = predictions

# Add placeholder column to training data
data_train["Predicted_Close"] = np.nan

# Combine training and test datasets
final_dataset = pd.concat([data_train, test_data], axis=0)

# Write the updated dataset with predictions to a CSV file
output_final_path = (
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/AUDUSD_15min_with_predictions.csv"
)
final_dataset.to_csv(output_final_path, index=False)

# ----- Plot the actual vs predicted Close values for the test set -----
plt.figure(figsize=(10, 6))
plt.plot(test_data["Close_Lead"].values, label="Actual Close", color="blue")
plt.plot(test_data["Predicted_Close"].values, label="Predicted Close", color="red")
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Observation Index")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, print the first few rows of the updated test set
print(final_dataset.head())

# ----- GET THE COEFFICIENTS -----
# Extract coefficients from the Lasso model at the optimal lambda
# In scikit-learn, the intercept and coefficients are provided separately.
intercept = lasso_model.intercept_
coef = lasso_model.coef_
feature_names = [
    "Lagged_Return",
    "Midpoint",
    "Open_weighted_price",
    "Close_weighted_Price",
    "Ema_difference",
    "EMA_5",
    "EMA_20",
    "EMA_9",
]

# Create a DataFrame for easier readability (including intercept)
coefficients_df = pd.DataFrame(
    {"Feature": ["Intercept"] + feature_names, "Coefficient": [intercept] + list(coef)}
)

print("Lasso Regression Coefficients at Optimal Lambda:")
print(coefficients_df)

# Save the coefficients to a CSV file (optional)
coefficients_file_path = (
    "C:/Users/User/Desktop/Projects/Backtest/Csvs/lasso_coefficients.csv"
)
coefficients_df.to_csv(coefficients_file_path, index=False)

# Construct the regression equation (excluding intercept in the term list)
terms = coefficients_df[coefficients_df["Feature"] != "Intercept"]
equation = f"y = {round(intercept, 4)} + " + " + ".join(
    [f"{round(row.Coefficient, 4)} * {row.Feature}" for _, row in terms.iterrows()]
)
print("Lasso Regression Equation:")
print(equation)
