import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib.pyplot as plt

# Assume 'scaled_df' is your DataFrame with columns: close, open, high, low

data_path = "C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv"
scaled_df = pd.read_csv(data_path)
# Step 1: Create the restricted cubic spline basis for 'open'
spline_basis = dmatrix("cr(close, df=4)", data=scaled_df, return_type='dataframe')

# Step 2: Fit the linear regression model using the spline basis as predictors
model = sm.OLS(scaled_df['close'], spline_basis)
results = model.fit()
print(results.summary())

# Step 3 (Optional): Visualize the fitted relationship
# Generate a grid of 'open' values over the range observed in the data
open_grid = np.linspace(scaled_df['open'].min(), scaled_df['open'].max(), 100)
# Create spline basis for the grid (note: we use "x" as a placeholder here)
spline_basis_grid = dmatrix("cr(x, df=4)", {"x": open_grid}, return_type='dataframe')
# Predict the corresponding 'close' values using our fitted model
predicted_close = results.predict(spline_basis_grid)

# Plot the observed data and the fitted spline curve
plt.figure(figsize=(8, 6))
plt.scatter(scaled_df['open'], scaled_df['close'], facecolors='none', edgecolors='grey', label="Data")
plt.plot(open_grid, predicted_close, color='red', label="Fitted RCS Model")
plt.xlabel("Open")
plt.ylabel("Close")
plt.legend()
plt.title("Restricted Cubic Splines in Linear Regression")
plt.show()
