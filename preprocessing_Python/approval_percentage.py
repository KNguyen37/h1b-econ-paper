import pandas as pd

# List of years to process
years = range(2010, 2023)
dfs = []

# Read and combine all files
for year in years:
    file_path = f'/Users/kietnguyen/Downloads/ECON395/output_v4_total/processed_uscis_{year}.csv'
    try:
        # Read the file with UTF-16 encoding and tab delimiter
        df = pd.read_csv(file_path, low_memory=False)
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

data['NAICS Code'] = data['Industry_NAICS_Standardized']

# Convert numeric columns
numeric_cols = ['Initial Approval', 'Initial Denial',
                'Continuing Approval', 'Continuing Denial']
data[numeric_cols] = data[numeric_cols].apply(
    pd.to_numeric, errors='coerce').fillna(0)

# Calculate total approvals for each industry
# + data['Continuing Approval']
data['Total Approvals'] = data['Initial Approval'] + data['Continuing Approval']

# Aggregate data for each year and NAICS Code
yearly_data = data.groupby(['Fiscal Year', 'NAICS Code']).agg(
    Total_Approvals=('Total Approvals', 'sum')
).reset_index()

# Calculate total approvals across all industries for each year
yearly_total_approvals = yearly_data.groupby(
    'Fiscal Year')['Total_Approvals'].transform('sum')

# Calculate the percentage of total approvals for each industry within each year
yearly_data['Approval_Percentage'] = (
    yearly_data['Total_Approvals'] / yearly_total_approvals) * 100

# Pivot the yearly data to make it horizontal
yearly_pivot = yearly_data.pivot(index='NAICS Code', columns='Fiscal Year', values=[
                                 'Total_Approvals', 'Approval_Percentage'])

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
    # '2016-2020': (2016, 2020)
    '2012-2015': (2012, 2015), '2018-2019': (2018, 2019)
}

# Aggregate data for each period
period_data = []
for period, (start, end) in periods.items():
    period_df = yearly_data[(yearly_data['Fiscal Year'] >= start) & (
        yearly_data['Fiscal Year'] <= end)]
    period_agg = period_df.groupby('NAICS Code').agg(
        Total_Approvals=('Total_Approvals', 'sum')
    ).reset_index()
    period_agg['Period'] = period

    # Calculate total approvals across all industries for the period
    period_total_approvals = period_agg['Total_Approvals'].sum()

    # Calculate the percentage of total approvals for each industry within the period
    period_agg['Approval_Percentage'] = (
        period_agg['Total_Approvals'] / period_total_approvals) * 100

    period_data.append(period_agg)

# Combine all period data
period_data = pd.concat(period_data, ignore_index=True)

# Pivot the period data to make it horizontal
period_pivot = period_data.pivot(index='NAICS Code', columns='Period', values=[
                                 'Total_Approvals', 'Approval_Percentage'])

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
final_data.to_csv(
    'h1b_analysis_results_with_approval_percentage_Mar6_v5_fullapproved.csv', index=False)

print("Analysis complete. Results saved to h1b_analysis_results_with_approval_percentage.csv")
