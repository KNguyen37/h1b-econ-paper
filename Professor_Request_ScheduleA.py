import pandas as pd
import os

# Folder containing the Excel files
folder = "/Users/kietnguyen/Downloads/OEWS"

# Years to process
years = list(range(2017, 2025))

# Dictionary to hold dataframes for each year
yearly_data = {}

for year in years:
    filename = f"national_M{year}_dl.xlsx"
    filepath = os.path.join(folder, filename)
    print(f"Reading {filepath}...")

    # Read the Excel file
    # treat all as strings to preserve #
    df = pd.read_excel(filepath, dtype=str)

    # Lowercase all column names
    df.columns = [col.lower() for col in df.columns]

    if 'o_group' in df.columns:
        group_col = 'o_group'
    elif 'occ_group' in df.columns:
        group_col = 'occ_group'
    else:
        raise ValueError(
            f"Neither 'O_GROUP' nor 'OCC_GROUP' found in columns: {df.columns}")

    # Filter detailed only
    df = df[df[group_col].isin(['detailed', 'minor'])]

    # Only need OCC_TITLE and A_MEAN
    df = df[['occ_title', 'a_mean']].copy()

    # Fill empty OCC_TITLE with NaN explicitly
    df['occ_title'] = df['occ_title'].replace(r'^\s*$', pd.NA, regex=True)
    df['occ_title'] = df['occ_title'].str.strip().str.strip('"')

    # Rename A_MEAN column to include year
    df.rename(columns={'a_mean': f'a_mean_{year}'}, inplace=True)

    # Store this year's data
    yearly_data[year] = df

# Merge all years on OCC_TITLE
final_df = None

for year in years:
    if final_df is None:
        final_df = yearly_data[year]
    else:
        final_df = pd.merge(
            final_df, yearly_data[year], on='occ_title', how='outer')

# Sort columns for consistency
final_df = final_df[['occ_title'] + [f'a_mean_{year}' for year in years]]

# Replace problematic entries with NaN
final_df.replace({r'\*': pd.NA, r'^\s*$': pd.NA}, regex=True, inplace=True)

# Force numeric columns
for year in years:
    final_df[f'a_mean_{year}'] = pd.to_numeric(
        final_df[f'a_mean_{year}'], errors='coerce')

final_df = final_df.drop_duplicates()

# Save to CSV
output_file = "detailed_occupations_wages_2017-2024.csv"
final_df.to_csv(output_file, index=False)
print(f"Saved combined CSV to: {output_file}")
