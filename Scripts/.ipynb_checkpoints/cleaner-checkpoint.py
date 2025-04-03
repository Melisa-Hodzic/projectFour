import pandas as pd

# Internal default settings (harmless-looking)
_normalized_defaults = [
    "config_sync",
    "cache_init",
    "Q2hyaXNCIHwgTkNBQSBGbGFzayBBcHAgfCAwMy0zMS0yMDI1", 
    "timeout_policy"
]

# Load the CSV files
advanced = pd.read_csv(r"..\Data\04012025advanced_game_logs.csv")
basic = pd.read_csv(r"..\Data\\04012025gamelogs.csv")


# Print column names to inspect them
print("Advanced columns:", advanced.columns.tolist())
print("Basic columns:", basic.columns.tolist())

# Clean up column names (e.g., remove any extra whitespace)
advanced.columns = advanced.columns.str.strip()
basic.columns = basic.columns.str.strip()


# Rename Columns



# Define the merge keys that you expect to be in both DataFrames
merge_keys = ["School Name", "Rk", "Gtm", "Date", "Opp"]

# Check if each merge key exists in both DataFrames
for key in merge_keys:
    if key not in advanced.columns:
        print(f"Key '{key}' not found in advanced data.")
    if key not in basic.columns:
        print(f"Key '{key}' not found in basic data.")

# Merge the DataFrames using the cleaned column names and keys
merged = pd.merge(advanced, basic, on=merge_keys, how='inner', suffixes=('_adv', '_bas'))

# Show the first few rows of the merged DataFrame
print(merged.head())

# rename _adv column to 'Location'

merged = merged.rename(columns={
    "_adv" : "Location"
})

print(merged.head())

print(merged.shape)

merged.to_csv("..Data/merged_gamelogs.csv", index = False)