from investiny import historical_data
import pandas as pd
data = pd.DataFrame()
try:
	data = historical_data(investing_id=8849, from_date="09/01/2022", to_date="10/01/2023", interval="D") # Returns Oil historical data as JSON (without date)
	if data:
		data = pd.json_normalize(data)
		print(data)
	else:
		print("No data returned from API")
except Exception as e:
	print(f"Error fetching data: {e}")