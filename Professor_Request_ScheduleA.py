import pandas as pd
import os

# Folder containing the Excel files
folder = "/Users/kietnguyen/Downloads/OEWS"

# Years to process
years = list(range(2017, 2025))
yearly_data = {}

for year in years:
    filename = f"national_M{year}_dl.xlsx"
    filepath = os.path.join(folder, filename)
    print(f"Reading {filepath}...")

    df = pd.read_excel(filepath, dtype=str)
    df.columns = [col.lower() for col in df.columns]

    if 'o_group' in df.columns:
        group_col = 'o_group'
    elif 'occ_group' in df.columns:
        group_col = 'occ_group'
    else:
        raise ValueError(
            f"Neither 'O_GROUP' nor 'OCC_GROUP' found in columns: {df.columns}")

    df = df[df[group_col].isin(['detailed', 'minor'])]
    df = df[['occ_code', 'occ_title', 'a_mean']].copy()

    df['occ_title'] = df['occ_title'].replace(r'^\s*$', pd.NA, regex=True)
    df['occ_title'] = df['occ_title'].str.strip().str.strip('"')

    df.rename(columns={
        'occ_title': f'occ_title_{year}',
        'a_mean': f'a_mean_{year}'
    }, inplace=True)

    yearly_data[year] = df

# Merge all years on occ_code
final_df = None
for year in years:
    if final_df is None:
        final_df = yearly_data[year]
    else:
        final_df = pd.merge(
            final_df, yearly_data[year], on='occ_code', how='outer')

# Clean and coerce
final_df.replace({r'\*': pd.NA, r'^\s*$': pd.NA}, regex=True, inplace=True)
for year in years:
    final_df[f'a_mean_{year}'] = pd.to_numeric(
        final_df[f'a_mean_{year}'], errors='coerce')

# List of title and wage columns
title_cols = [f'occ_title_{year}' for year in years]
wage_cols = [f'a_mean_{year}' for year in years]

# Drop rows with no title and no wages at all
final_df = final_df.dropna(subset=title_cols + wage_cols, how='all')

# 1️⃣ CSV with *just wages* and occ_code
just_wages_df = final_df[['occ_code'] + title_cols +
                         wage_cols].drop_duplicates().sort_values(by='occ_code')
just_wages_df.to_csv("occupations_wages_only.csv", index=False)
print("Saved: occupations_wages_only.csv")

# 2️⃣ Earliest available title
final_df['earliest_title'] = final_df[title_cols].bfill(axis=1).iloc[:, 0]
earliest_df = final_df[['occ_code', 'earliest_title'] +
                       wage_cols].drop_duplicates().sort_values(by='occ_code')
earliest_df.to_csv("occupations_earliest_title.csv", index=False)
print("Saved: occupations_earliest_title.csv")

# 3️⃣ Latest available title
final_df['latest_title'] = final_df[title_cols].ffill(axis=1).iloc[:, -1]
latest_df = final_df[['occ_code', 'latest_title'] +
                     wage_cols].drop_duplicates().sort_values(by='occ_code')
latest_df.to_csv("occupations_latest_title.csv", index=False)
print("Saved: occupations_latest_title.csv")

# 4️⃣ Both earliest and latest titles side by side + Change flag
both_titles_df = final_df[['occ_code', 'earliest_title', 'latest_title'] +
                          wage_cols].drop_duplicates().sort_values(by='occ_code')
both_titles_df['title_changed'] = both_titles_df.apply(
    lambda row: "Yes" if pd.notna(row['earliest_title']) and pd.notna(
        row['latest_title']) and row['earliest_title'] != row['latest_title'] else "No",
    axis=1
)
both_titles_df = both_titles_df[[
    'occ_code', 'earliest_title', 'latest_title', 'title_changed'] + wage_cols]
both_titles_df = both_titles_df[both_titles_df['title_changed'] == "Yes"]
both_titles_df.to_csv("occupations_earliest_latest_titles.csv", index=False)
print("Saved: occupations_earliest_latest_titles.csv")
