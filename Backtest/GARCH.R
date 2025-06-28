# install.packages("rugarch")
library(rugarch)

# 1. Load your data and compute the volatility series
df <- read.csv("C:/Users/User/Desktop/Projects/Backtest/Csvs/NQ1!_MAIN_1D.csv", stringsAsFactors = FALSE)

# Compute proxy volatility
df$volatility <- abs(df$close - df$open) * 0.1

# 2. Specify and fit a GARCH(2,2) with AR(5) mean and skew‐t errors
spec <- ugarchspec(
  mean.model     = list(armaOrder   = c(5, 0),
                        include.mean = TRUE),
  variance.model = list(model      = "sGARCH",
                        garchOrder = c(2, 2)),
  distribution.model = "sstd"    # skewed Student’s t
)

fit <- ugarchfit(spec = spec, data = df$volatility, solver = "hybrid")

# 3. In‐sample “forecast” ⇒ conditional volatility
df$vol_forecast <- sigma(fit)

# Extract standardized residuals
std_resid <- residuals(fit, standardize = TRUE)

# Ljung–Box tests at lags 10 and 20
lb1_10 <- Box.test(std_resid,      lag = 10, type = "Ljung-Box")
lb1_20 <- Box.test(std_resid,      lag = 20, type = "Ljung-Box")
lb2_10 <- Box.test(std_resid^2,    lag = 10, type = "Ljung-Box")
lb2_20 <- Box.test(std_resid^2,    lag = 20, type = "Ljung-Box")

cat("Ljung–Box on resid (lag 10): p =", lb1_10$p.value, "\n")
cat("Ljung–Box on resid (lag 20): p =", lb1_20$p.value, "\n\n")
cat("Ljung–Box on resid^2 (lag 10): p =", lb2_10$p.value, "\n")
cat("Ljung–Box on resid^2 (lag 20): p =", lb2_20$p.value, "\n\n")

# Print model summary
show(fit)

# 4. Plot observed vs. in‐sample conditional volatility
plot(df$volatility, type = "l", lwd = 1.2,
     xlab = "Time", ylab = "Volatility",
     main = "High–Low Volatility vs. GARCH In‐Sample Forecast")
lines(df$vol_forecast, lty = 2, lwd = 1.2)
legend("topright",
       legend = c("Observed Volatility", "Conditional Volatility"),
       lty    = c(1, 2),
       lwd    = c(1.2, 1.2))

