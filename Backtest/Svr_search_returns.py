
# win = 0
# loss = 0
# total_predictions = 0
# profit = 0
# losess = 0

# for index, row in merged_df.iterrows():
#     if not pd.isnull(row["prediction"]):
#         if row["prediction"] > row["open"]:
#             if (row["close"] - row["open"]) > 0:
#                 profit += abs(row["close"] - row["open"]) - 1
#                 win += 1
#                 total_predictions += 1
#             elif (row["close"] - row["open"]) < 0:
#                 losess += abs(row["close"] - row["open"]) + 1
#                 loss += 1
#                 total_predictions += 1
#         elif row["prediction"] < row["open"]:
#             if (row["close"] - row["open"]) < 0:
#                 profit += abs(row["close"] - row["open"]) - 1
#                 win += 1
#                 total_predictions += 1
#             elif (row["close"] - row["open"]) > 0:
#                 losess += abs(row["close"] - row["open"]) + 1
#                 loss += 1
#                 total_predictions += 1


# average_profit = profit / total_predictions
# average_loss = losess / total_predictions
# risk_reward_ratio = average_profit / average_loss
# print("Evaluation of Predictions:")
# print(f"Wining Ratio: {win/total_predictions:.3%}")
# print(f"Risk Reward Ratio: {risk_reward_ratio:.3f}")
# print(f"Total: {profit-losess}")