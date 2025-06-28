import pandas as pd
import numpy as np




def calculate_momentum_entropy(df, column='close', window=10):
    def momentum_entropy(prices):
        # Calculate price changes
        changes = np.diff(prices)
        # Classify changes: -1 (down), 0 (no change), 1 (up)
        categories = np.sign(changes)
        
        # Count occurrences of each category
        counts = np.array([
            (categories == -1).sum(),  # Down
            (categories == 0).sum(),   # No change
            (categories == 1).sum()    # Up
        ])
        
        # Convert counts to probabilities
        probabilities = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        
        # Compute entropy
        entropy = -np.nansum(probabilities * np.log(probabilities + 1e-10))  # Add small value to avoid log(0)
        return entropy

    # Apply rolling window and calculate entropy
    df['momentum_entropy'] = df[column].rolling(window=window).apply(
        lambda x: momentum_entropy(x), raw=True
    )
    return df

# Example usage:
# Assuming 'data' is a DataFrame with a 'close' column
# data = pd.DataFrame({'close': [100, 102, 101, 103, 104, 103, 105, 106, 107, 108]})

data = pd.read_csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_SCALED_1D.csv")
window_size = 5
result = calculate_momentum_entropy(data, column='close', window=window_size)

# Display the result
print(result["momentum_entropy"])
