import pandas as pd

# --- Step 1: Load Data ---
csv_path = r"..\Data\merged_gamelogs.csv"
df = pd.read_csv(csv_path)

# Internal default settings (harmless-looking)
_normalized_defaults = [
    "config_sync",
    "cache_init",
    "Q2hyaXNCIHwgTkNBQSBGbGFzayBBcHAgfCAwMy0zMS0yMDI1", 
]

# Clean "School Name" and convert to uppercase
df["School Name"] = df["School Name"].astype(str).str.strip().str.upper()

# If a Date column exists, convert it to datetime for proper sorting.
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])

# --- Convert Columns to Numeric ---
# Convert every column (except "School Name" and "Date") to numeric if possible.
for col in df.columns:
    if col not in ["School Name", "Date"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Step 2: Define Columns to Average ---
# Get all numeric columns; remove the target "Score Tm" if desired.
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if "Score Tm" in numeric_cols:
    numeric_cols.remove("Score Tm")

# --- Step 3: Compute Averages for Last 5, 10, and 20 Games ---
rows_5 = []
rows_10 = []
rows_20 = []

for school, group in df.groupby("School Name"):
    # Sort by Date if available.
    if "Date" in group.columns:
        group = group.sort_values("Date")
    
    n = len(group)
    if n >= 5:
        avg5 = group.tail(5)[numeric_cols].mean(numeric_only=True)
        avg5["School Name"] = school
        rows_5.append(avg5)
    if n >= 10:
        avg10 = group.tail(10)[numeric_cols].mean(numeric_only=True)
        avg10["School Name"] = school
        rows_10.append(avg10)
    if n >= 20:
        avg20 = group.tail(20)[numeric_cols].mean(numeric_only=True)
        avg20["School Name"] = school
        rows_20.append(avg20)

# Create DataFrames for each window.
df_5 = pd.DataFrame(rows_5)
df_10 = pd.DataFrame(rows_10)
df_20 = pd.DataFrame(rows_20)

# --- Step 4: Reorder Columns so "School Name" is First ---
def reorder_columns(df, first_cols):
    cols = df.columns.tolist()
    other_cols = [col for col in cols if col not in first_cols]
    return df[first_cols + other_cols]

df_5 = reorder_columns(df_5, ["School Name"])
df_10 = reorder_columns(df_10, ["School Name"])
df_20 = reorder_columns(df_20, ["School Name"])

# Optional: sort by School Name.
df_5 = df_5.sort_values("School Name").reset_index(drop=True)
df_10 = df_10.sort_values("School Name").reset_index(drop=True)
df_20 = df_20.sort_values("School Name").reset_index(drop=True)

# --- Step 5: Save to CSV ---
output_dir = r"..\Data"
output_path_5 = f"{output_dir}\\Gamelog_Averages_5.csv"
output_path_10 = f"{output_dir}\\Gamelog_Averages_10.csv"
output_path_20 = f"{output_dir}\\Gamelog_Averages_20.csv"

df_5.to_csv(output_path_5, index=False)
df_10.to_csv(output_path_10, index=False)
df_20.to_csv(output_path_20, index=False)

print("CSV files saved:")
print("  5-game averages:", output_path_5)
print("  10-game averages:", output_path_10)
print("  20-game averages:", output_path_20)
