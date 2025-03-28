import pandas as pd

# List of years to process
years = range(2012, 2021)
dfs = []

# Read and combine all files
for year in years:
    file_path = f'/Users/kietnguyen/Downloads/ECON395/output_v4_total/processed_uscis_{year}.csv'
    try:
        # Read the file with UTF-16 encoding and tab delimiter
        df = pd.read_csv(file_path, low_memory=False)

        # # Standardize column names
        # df.rename(columns={
        #     'Fiscal Year   ': 'Fiscal Year',
        # }, inplace=True)

        dfs.append(df)
        print(f"Successfully read {file_path}")
    except FileNotFoundError:
        print(f"File for year {year} not found. Skipping.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if not dfs:
    print("No files were successfully read. Please check the file paths and encodings.")
    exit()

# Combine all data into a single DataFrame
data = pd.concat(dfs, ignore_index=True)

# Debug: Check the shape and first few rows of the combined data
print(f"Combined data shape: {data.shape}")
print(data.head())

# Clean NAICS codes


# def clean_naics(naics_str):
#     if pd.isna(naics_str) or str(naics_str).strip() in ['', 'nan']:
#         return ('Unknown', 'Unknown or Unclassified Industries')
#     parts = str(naics_str).split(' - ')
#     code = parts[0].strip() if parts[0].strip() else 'Unknown'
#     name = parts[1].strip() if len(
#         parts) > 1 else 'Unknown or Unclassified Industries'
#     return (code, name)


data['NAICS Code'] = data['Industry_NAICS_Standardized']
# .apply(clean_naics)

# Convert numeric columns
numeric_cols = ['Initial Approval', 'Initial Denial',
                'Continuing Approval', 'Continuing Denial']
data[numeric_cols] = data[numeric_cols].apply(
    pd.to_numeric, errors='coerce').fillna(0)

# Calculate metrics for each year
data['Total Petitions'] = data[numeric_cols].sum(axis=1)
data['Total Denials'] = data[[
    'Initial Denial', 'Continuing Denial']].sum(axis=1)
data['Denial Rate'] = round(
    data['Total Denials'] / data['Total Petitions'].replace(0, 1), 4)
data['Industry Share'] = round(
    (data['Total Petitions'] / data['Total Petitions'].sum()) * 100, 2)

# Debug: Check data before aggregation
print("Data before aggregation:")
print(data[['Fiscal Year', 'NAICS Code', 'Total Petitions',
      'Total Denials', 'Denial Rate', 'Industry Share']].head())

# Aggregate data for each year
yearly_data = data.groupby(['Fiscal Year', 'NAICS Code']).agg(
    Total_Petitions=('Total Petitions', 'sum'),
    Total_Denials=('Total Denials', 'sum')
).reset_index()

# Calculate denial rate and industry share for each year
yearly_data['Denial Rate'] = round(
    yearly_data['Total_Denials'] / yearly_data['Total_Petitions'].replace(0, 1), 4)
yearly_data['Industry Share'] = round((yearly_data['Total_Petitions'] / yearly_data.groupby(
    'Fiscal Year')['Total_Petitions'].transform('sum')) * 100, 2)

# Pivot the yearly data to make it horizontal
yearly_pivot = yearly_data.pivot(index='NAICS Code', columns='Fiscal Year', values=[
                                 'Denial Rate', 'Industry Share'])

# Flatten the multi-level column index
yearly_pivot.columns = [f'{col[0]} {col[1]}' for col in yearly_pivot.columns]

# Reset index to include NAICS Code as a column
yearly_pivot.reset_index(inplace=True)

# Debug: Check pivoted yearly data
print("Pivoted yearly data:")
print(yearly_pivot.head())

# Create group periods
periods = {
    # '2010-2013': (2010, 2013),
    # '2014-2020': (2014, 2020),
    # '2021-2022': (2021, 2022),
    # '2012-2015': (2012, 2015),
    # '2016-2020': (2016, 2020),
    '2012-2015': (2012, 2015),
    '2018-2019': (2018, 2019)
}

# Aggregate data for each period
period_data = []
for period, (start, end) in periods.items():
    period_df = yearly_data[(yearly_data['Fiscal Year'] >= start) & (
        yearly_data['Fiscal Year'] <= end)]
    period_agg = period_df.groupby('NAICS Code').agg(
        Total_Petitions=('Total_Petitions', 'sum'),
        Total_Denials=('Total_Denials', 'sum')
    ).reset_index()
    period_agg['Period'] = period
    period_agg['Denial Rate'] = round(
        period_agg['Total_Denials'] / period_agg['Total_Petitions'].replace(0, 1), 4)
    period_agg['Industry Share'] = round(
        (period_agg['Total_Petitions'] / period_agg['Total_Petitions'].sum()) * 100, 2)
    period_data.append(period_agg)

# Combine all period data
period_data = pd.concat(period_data, ignore_index=True)

# Pivot the period data to make it horizontal
period_pivot = period_data.pivot(index='NAICS Code', columns='Period', values=[
                                 'Denial Rate', 'Industry Share'])

# Flatten the multi-level column index
period_pivot.columns = [f'{col[0]} {col[1]}' for col in period_pivot.columns]

# Reset index to include NAICS Code as a column
period_pivot.reset_index(inplace=True)

# Debug: Check pivoted period data
print("Pivoted period data:")
print(period_pivot.head())

# Combine yearly and period data
final_data = pd.merge(yearly_pivot, period_pivot, on='NAICS Code', how='outer')

# Debug: Check final combined data
print("Final combined data:")
print(final_data.head())

# Save results
final_data.to_csv('h1b_analysis_results_Mar7_verBAHA.csv', index=False)

print("Analysis complete. Results saved to h1b_analysis_results_v2.csv")
