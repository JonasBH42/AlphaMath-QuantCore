library(quantmod)
library(glmnet)
library(dplyr)
library(readr)
library(ggplot2)

# Load your data
data <- read.csv("C:\\Users\\User\\Desktop\\Projects\\Backtest\\Csvs\\NQ1!_MAIN_1D.csv")
data <- na.omit(data)
data$Close <- as.numeric(data$Close)

# Calculate EMAs
ema5 <- EMA(data$Close, n = 5)
data$EMA_5 <- ema5
ema20 <- EMA(data$Close, n = 20)
data$EMA_20 <- ema20
ema9 <- EMA(data$Close, n = 9)
data$EMA_9 <- ema9
# Calculate Return
data$Return <- data$Close - data$Open

# Set up lagged values and target (Close lead by 1)
lagged_data_1 <- data %>%
  mutate(
    
    Close_Lead = lead(Close, 1),  # Close lead by 1
    Lagged_Return = data$Return,
    EMA_5_Lag1 = lag(EMA_5, 1),
    EMA_20_Lag1 = lag(EMA_20, 1),
    EMA_9_Lag1 = lag(EMA_9, 1),
    Ema_difference = Close - EMA_5,  # Example feature
    Midpoint = (Open + Close) / 2,
    Open_weighted_price = Open + (Close) / 2,
    Close_weighted_Price = Close + (Open) / 2
  )

# Omit NA values
lagged_data <- na.omit(lagged_data_1)
file_path <- "C:\\Users\\User\\Desktop\\Projects\\Backtest\\Csvs\\AUDUSD_15min.csv"
write.csv(lagged_data, file_path, row.names = FALSE)

# Reload the subset for processing
subset <- read.csv(file_path)

# Split data for training and test
train_index <- floor(0.8 * nrow(subset))

# Split the data into training and test sets
data_train <- subset[1:2481,]
lagged_data <- subset[2481:nrow(subset),]

# Prepare data for Lasso regression
lasso_input_data <- as.matrix(data_train[, c("Lagged_Return",
                                             "Midpoint", "Open_weighted_price",
                                             "Close_weighted_Price", "Ema_difference",
                                             "EMA_5", "EMA_20", "EMA_9")]) # Added EMA values here
Y <- data_train$Close_Lead

# Remove rows with any NA values
complete_cases <- complete.cases(lasso_input_data, Y)
X <- lasso_input_data[complete_cases, ]
Y <- Y[complete_cases]

# Train the Lasso regression model
lasso_model <- glmnet(X, Y, alpha = 1)
# Save the lasso_model to a file
save(lasso_model, file = "C:\\Users\\User\\Desktop\\Projects\\Backtest\\lasso_model.RData")
# Perform cross-validation to select lambda
cv_lasso <- cv.glmnet(X, Y, alpha = 1)
best_lambda <- cv_lasso$lambda.min

# Prepare test data matrix for prediction
lagged_data_matrix <- as.matrix(lagged_data[, c("Lagged_Return",
                                                "Midpoint", "Open_weighted_price",
                                                "Close_weighted_Price", "Ema_difference",
                                                "EMA_5", "EMA_20", "EMA_9")]) # Added EMA values here

# Run predictions
predictions <- predict(lasso_model, newx = lagged_data_matrix, s = best_lambda)

# Add predictions to the lagged_data dataset
lagged_data$Predicted_Close <- as.numeric(predictions)

# Add placeholder column to data_train
data_train$Predicted_Close <- NA

# Combine training and test datasets
final_dataset <- rbind(data_train, lagged_data)

# Write the updated dataset with predictions to a CSV file
output_file_path <- "C:\\Users\\User\\Desktop\\Projects\\Backtest\\Csvs\\AUDUSD_15min_with_predictions.csv"
write.csv(final_dataset, output_file_path, row.names = FALSE)

# Plot the actual vs predicted Close values
p <- ggplot(lagged_data, aes(x = seq_along(Close_Lead))) +
  geom_line(aes(y = Close_Lead, color = "Actual Close")) +
  geom_line(aes(y = Predicted_Close, color = "Predicted Close")) +
  labs(title = "Actual vs Predicted Close Prices",
       x = "Observation Index",
       y = "Close Price") +
  theme_minimal() +
  scale_color_manual(name = "Legend", values = c("Actual Close" = "blue", "Predicted Close" = "red"))

print(p)
# Optionally, print the first few rows of the updated test set
# print(final_dataset)

#--------------GET THE COEFFICIENTS----------------------

# Extract coefficients from the Lasso model at the optimal lambda
coefficients <- coef(lasso_model, s = best_lambda)

# Convert coefficients to a data frame for easier readability
coefficients_df <- as.data.frame(as.matrix(coefficients))
colnames(coefficients_df) <- c("Coefficient")
coefficients_df$Feature <- rownames(coefficients_df)

# Print the coefficients
print("Lasso Regression Coefficients at Optimal Lambda:")
print(coefficients_df)

# Print the lambda value
print(paste("Optimal Lambda Value:", best_lambda))

# Save the coefficients to a CSV file (optional)
coefficients_file_path <- "C:\\Users\\User\\Desktop\\Projects\\Backtest\\Csvs\\lasso_coefficients.csv"
write.csv(coefficients_df, coefficients_file_path, row.names = FALSE)

# Construct the regression equation (excluding intercept for now)
intercept <- coefficients_df$Coefficient[1]
terms <- coefficients_df[-1, ]  # Exclude intercept
equation <- paste0("y = ", round(intercept, 4), " + ",
                   paste0(round(terms$Coefficient, 4), " * ", terms$Feature, collapse = " + "))

# Print the equation
print("Lasso Regression Equation:")
print(equation)