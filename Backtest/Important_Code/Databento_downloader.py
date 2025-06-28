import databento as db

# Initialize the client with your API key
client = db.Historical("Key")

# Request data
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    start="2025-03-07",
    end="2025-06-19",
    symbols=["NQ.v.0"],
    stype_in="continuous",
    schema="ohlcv-1d",
)

df = data.to_df()
df.to_csv("test.csv")
# SI
