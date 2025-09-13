import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# 1. Load the data
path = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/Crude_Oil_1D.csv"
df = pd.read_csv(path, parse_dates=["datetime"])
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

# 2. Quick data snapshot
print("=== Data Info ===")
print(df.info(), "\n")

print("=== Head ===")
print(df.head(), "\n")

# 3. Missing values check
print("=== Missing Values ===")
print(df.isna().sum(), "\n")

# 4. Descriptive statistics
print("=== Summary Statistics ===")
print(df.describe(), "\n")

# 5. Time series plots
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['close'], label='Close')
plt.title("Close Price Over Time")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(12, 3))
plt.bar(df.index, df['volume'], width=1.0, alpha=0.6)
plt.title("Volume Over Time")
plt.ylabel("Volume")
plt.show()

# 6. Compute daily returns
df['return'] = df['close'].pct_change()

# 7. Histogram and KDE of returns
plt.figure(figsize=(8, 4))
sns.histplot(df['return'].dropna(), kde=True, bins=50)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.show()

# 8. Identify top 5 largest moves (anomalies)
top_moves = df['return'].abs().nlargest(5)
print("=== Top 5 Largest Absolute Returns ===")
print(top_moves, "\n")

# 9. Boxplots to spot outliers in price & volume
plt.figure(figsize=(10, 4))
sns.boxplot(data=df[['open','high','low','close']])
plt.title("Boxplot of OHLC")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['volume'])
plt.title("Boxplot of Volume")
plt.show()

# 10. Correlation analysis
cols = ['open','high','low','close','volume','return']
corr = df[cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

# 11. Pair-plot for relationships
sns.pairplot(df[cols].dropna(), kind='scatter', plot_kws={'s':10, 'alpha':0.4})
plt.suptitle("Pair-plot of Features", y=1.02)
plt.show()

# 3. Scatter-plots + LOESS
raw_cols = ['open', 'high', 'low', 'volume']
for col in raw_cols:
    x = df[col].dropna()
    y = df.loc[x.index, 'close']
    # compute LOESS
    loess_sm = lowess(endog=y, exog=x, frac=0.1, return_sorted=True)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=10, alpha=0.3, label='data')
    plt.plot(loess_sm[:, 0], loess_sm[:, 1], color='red', linewidth=2, label='LOESS')
    plt.title(f'Close vs. {col.capitalize()} with LOESS smooth')
    plt.xlabel(col.capitalize())
    plt.ylabel('Close')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. Histograms + KDE for each variable
all_cols = raw_cols + ['close']
for col in all_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=50)
    plt.title(f'Distribution of {col.capitalize()} (hist + KDE)')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()