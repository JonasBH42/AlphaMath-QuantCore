import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.stats.diagnostic as diag

def perform_tests(series, tests, parameter_name):
    # Drop missing values for the tests
    data = series.dropna()
    print(f"\n===== Tests for {parameter_name} =====")
    
    # --- ADF Test ---
    if tests.get("adf", False):
        adf_result = ts.adfuller(data)
        print("\nADF Test:")
        print(f"  Test Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")

    # --- KPSS Test ---
    if tests.get("kpss", False):
        try:
            kpss_result = ts.kpss(data, regression='c', nlags="auto")
            print("\nKPSS Test:")
            print(f"  Test Statistic: {kpss_result[0]:.4f}")
            print(f"  p-value: {kpss_result[1]:.4f}")
        except Exception as e:
            print("\nKPSS Test: Failed to run:", e)

    # --- Ljung–Box Test ---
    if tests.get("ljung_box", False):
        # Here we use 10 lags as an example; adjust as needed.
        lb_result = diag.acorr_ljungbox(data, lags=[10], return_df=True)
        print("\nLjung–Box Test (10 lags):")
        print(lb_result)

    # --- Engle's ARCH Test ---
    if tests.get("arch", False):
        try:
            arch_result = diag.het_arch(data)
            print("\nEngle's ARCH Test:")
            print(f"  LM Statistic: {arch_result[0]:.4f}")
            print(f"  LM p-value: {arch_result[1]:.4f}")
            print(f"  F-Statistic: {arch_result[2]:.4f}")
            print(f"  F p-value: {arch_result[3]:.4f}")
        except Exception as e:
            print("\nEngle's ARCH Test: Failed to run:", e)

# Mapping for each parameter to the tests that should be applied.
test_mapping = {
    "Normalized Range":        {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "Squared Returns":         {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "Close-to-Open Returns":   {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "Log Returns":             {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "High-Low/Close Ratio":    {"adf": True, "kpss": True, "ljung_box": True, "arch": False},
    "ROC":                     {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "OBV":                     {"adf": True, "kpss": False, "ljung_box": False, "arch": False},
    "Volume Rate of Change":   {"adf": True, "kpss": True, "ljung_box": True, "arch": True},
    "normalized_volatility":   {"adf": True, "kpss": True, "ljung_box": True, "arch": False},
}

# Read the CSV file into a DataFrame
csv_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/Stats/NQ1_SCALED_1D_with_volatility_2.csv"
df = pd.read_csv(csv_path)

# Loop over the test mapping and perform tests for each available column in the DataFrame
for parameter, tests in test_mapping.items():
    if parameter in df.columns:
        perform_tests(df[parameter], tests, parameter)
    else:
        print(f"\n===== {parameter} =====")
        print("Column not found in DataFrame.")
