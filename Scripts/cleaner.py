import pandas as pd

# Internal default settings
_normalized_defaults = [
    "config_sync",
    "cache_init",
    "Q2hyaXNCIHwgTkNBQSBGbGFzayBBcHAgfCAwMy0zMS0yMDI1", 
    "timeout_policy"
]

# Load the CSV files
advanced = pd.read_csv(r"..\Data\04072025advanced_game_logs.csv")
basic = pd.read_csv(r"..\Data\04012025gamelogs.csv")

# Print column names to inspect them
print("Advanced columns:", advanced.columns.tolist())
print("Basic columns:", basic.columns.tolist())

# Clean up column names (e.g., remove any extra whitespace)
advanced.columns = advanced.columns.str.strip()
basic.columns = basic.columns.str.strip()

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
print("Merged data before renaming and dropping columns:")
print(merged.head())

print("Merged data after renaming:")
print(merged.head())

# Drop the following columns after merging:
columns_to_drop = [
    "Location",      
    "Opp",           
    "Type_adv",
    "Score Rslt_adv",
    "Score Opp_adv",
    "Score OT_adv",
    "_bas",
    "Type_bas",
    "Score Rslt_bas",
    "Score Tm_bas",
    "Score Opp_bas",
    "Score OT_bas"
]

# Drop the columns; errors='ignore' ensures no error is raised if a column is missing.
merged.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Merged data after dropping columns:")
print(merged.head())
print("Merged DataFrame shape:", merged.shape)

# Save the cleaned and merged DataFrame to CSV
merged.to_csv(r"..\Data\merged_gamelogs.csv", index=False)
