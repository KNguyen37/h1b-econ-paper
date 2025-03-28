import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import re
import os
from collections import defaultdict
import ast
from datetime import datetime

# Configure logging
log_filename = f'h1b_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)
logger = logging.getLogger(__name__)
# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# NAICS sector code mapping
NAICS_SECTORS = {
    '11': 'Agriculture, Forestry, Fishing and Hunting',
    '21': 'Mining, Quarrying, and Oil and Gas Extraction',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale Trade',
    '44': 'Retail Trade',
    '45': 'Retail Trade',
    '48': 'Transportation and Warehousing',
    '49': 'Transportation and Warehousing',
    '51': 'Information',
    '52': 'Finance and Insurance',
    '53': 'Real Estate and Rental and Leasing',
    '54': 'Professional, Scientific, and Technical Services',
    '55': 'Management of Companies and Enterprises',
    '56': 'Administrative and Support and Waste Management and Remediation Services',
    '61': 'Educational Services',
    '62': 'Health Care and Social Assistance',
    '71': 'Arts, Entertainment, and Recreation',
    '72': 'Accommodation and Food Services',
    '81': 'Other Services (except Public Administration)',
    '92': 'Public Administration',
}

# Special cases for combined sectors
COMBINED_SECTORS = {
    '31-33': 'Manufacturing',
    '44-45': 'Retail Trade',
    '48-49': 'Transportation and Warehousing'
}


class H1BProcessor:
    def __init__(self, uscis_file_path, lca_file_path, output_dir="./output"):
        self.uscis_file_path = uscis_file_path
        self.lca_file_path = lca_file_path
        self.output_dir = output_dir

        # Extract year from filename for output naming
        year_match = re.search(r'20\d{2}', os.path.basename(uscis_file_path))
        self.year = year_match.group(0) if year_match else "unknown"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Company name abbreviation mapping
        self.abbrev_mapping = {
            "US": "UNITED STATES", "USA": "UNITED STATES", "U S": "UNITED STATES",
            "U S A": "UNITED STATES", "AMERICA": "UNITED STATES", "INTL": "INTERNATIONAL",
            "INT": "INTERNATIONAL", "TECH": "TECHNOLOGY", "TECHS": "TECHNOLOGIES",
            "SVCS": "SERVICES", "SVC": "SERVICE", "MGMT": "MANAGEMENT",
            "CORP": "CORPORATION", "GRP": "GROUP", "ASSOC": "ASSOCIATES",
            "ASSN": "ASSOCIATION", "SYS": "SYSTEMS", "SYST": "SYSTEMS", "&": "AND",
        }

    def assign_naics_based_on_highest_applications(self, df):

        logger.info("Assigning NAICS codes based on highest applications")

        # Columns to sum for total applications
        application_cols = ['Initial Approval', 'Initial Denial',
                            'Continuing Approval', 'Continuing Denial']

        # Ensure numeric columns are properly typed
        for col in application_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Group by employer name and NAICS code, then sum applications
        grouped = df.groupby(['Employer (Petitioner) Name', 'Final_NAICS'])[
            application_cols].sum().reset_index()

        # Calculate total applications for each group
        grouped['Total_Applications'] = grouped[application_cols].sum(axis=1)

        # Find the NAICS code with the highest total applications for each employer
        highest_naics = grouped.loc[grouped.groupby(
            'Employer (Petitioner) Name')['Total_Applications'].idxmax()]

        # Create a mapping of employer name to the NAICS code with the highest applications
        employer_naics_mapping = dict(
            zip(highest_naics['Employer (Petitioner) Name'], highest_naics['Final_NAICS']))

        # Apply the mapping to the original dataframe
        df['Final_NAICS'] = df['Employer (Petitioner) Name'].map(
            employer_naics_mapping)

        # Update the standardized NAICS columns
        df['Standardized_Final_NAICS'] = df['Final_NAICS'].apply(
            self.convert_to_uscis_naics_format)
        df['Industry_NAICS_Standardized'] = df['Standardized_Final_NAICS']

        logger.info("NAICS codes assigned based on highest applications")
        return df

    def run_workflow(self):
        """Main processing workflow"""
        logger.info(f"Starting H1B data processing for year {self.year}")
        logger.info(f"USCIS file: {self.uscis_file_path}")
        logger.info(f"LCA file: {self.lca_file_path}")

        # Load USCIS data
        uscis_df = self.load_uscis_data()
        if uscis_df is None:
            logger.error("Failed to load USCIS data. Exiting.")
            return

        # IMPORTANT: Create a copy of the original NAICS column for reference
        uscis_df['Original_NAICS'] = uscis_df['Industry (NAICS) Code'].copy()

        # Check for empty NAICS codes
        empty_naics_count = uscis_df['Industry (NAICS) Code'].isna().sum()
        if pd.isna(empty_naics_count):
            empty_naics_count = 0
        empty_naics_count += (uscis_df['Industry (NAICS) Code'] == '').sum()

        logger.info(
            f"Found {empty_naics_count} empty NAICS codes in USCIS data")

        # Add a new column to track the source of NAICS code
        uscis_df['NAICS_Source'] = 'Original'

        # Create a column for matched NAICS codes
        uscis_df['Matched_NAICS'] = None

        # Extract empty NAICS rows
        if empty_naics_count > 0:
            # Extract unique employer data from LCA
            lca_df = self.extract_unique_employer_data()
            if lca_df is None or lca_df.empty:
                logger.error(
                    "Failed to extract employer data from LCA file. Exiting.")
                return

            # Find rows with empty NAICS codes
            empty_naics_df = self.find_empty_naics(uscis_df)

            # Keep track of the original indices
            empty_naics_df = empty_naics_df.reset_index().rename(
                columns={'index': 'original_index'})

            # Create temporary files for the matcher
            empty_naics_file = os.path.join(
                self.output_dir, f"empty_naics_{self.year}.csv")
            lca_file = os.path.join(
                self.output_dir, f"unique_lca_{self.year}.csv")

            empty_naics_df.to_csv(empty_naics_file, index=False)
            lca_df.to_csv(lca_file, index=False)

            # Match and fill NAICS codes
            matcher = H1BIndustryMatcher(empty_naics_file, lca_file)
            matched_df = matcher.match_industries()

            logger.info(
                f"Matched {matcher.stats['matched']} out of {matcher.stats['total_empty_naics']} records")

            # IMPORTANT: Update the dataframe with matched NAICS codes using original indices
            if matcher.match_log:
                # Create a lookup of original index to match information
                match_lookup = {}
                for log in matcher.match_log:
                    if 'original_index' in log:
                        match_lookup[log['original_index']] = log
                    else:
                        # Find the corresponding original index
                        idx = log['uscis_index']
                        if idx < len(empty_naics_df):
                            original_idx = empty_naics_df.loc[idx,
                                                              'original_index']
                            match_lookup[original_idx] = log

                # Apply the matches to the original dataframe
                for idx, log in match_lookup.items():
                    naics_code = log['naics_code']
                    match_type = log['match_type']
                    employer_name = log['employer_name']

                    # Update the Matched_NAICS column
                    uscis_df.loc[idx, 'Matched_NAICS'] = naics_code
                    uscis_df.loc[idx,
                                 'NAICS_Source'] = f'Matched ({match_type})'

                    # Add a debugging entry
                    logger.info(
                        f"Matched: {employer_name} (idx {idx}) -> {naics_code} via {match_type}")

            # Save matching log with additional diagnostic information
            matching_log_file = os.path.join(
                self.output_dir, f"matching_log_{self.year}.csv")
            if matcher.match_log:
                log_df = pd.DataFrame(matcher.match_log)

                # Add information from USCIS dataset for verification
                log_df['uscis_original_naics'] = log_df['uscis_index'].apply(
                    lambda idx: empty_naics_df.loc[idx, 'Original_NAICS'] if idx < len(
                        empty_naics_df) else None
                )
                log_df['original_index'] = log_df['uscis_index'].apply(
                    lambda idx: empty_naics_df.loc[idx, 'original_index'] if idx < len(
                        empty_naics_df) else None
                )

                log_df.to_csv(matching_log_file, index=False)
                logger.info(f"Saved matching log to {matching_log_file}")

        # Create a Final_NAICS column that uses matched when available, original otherwise
        uscis_df['Final_NAICS'] = uscis_df.apply(
            lambda row: row['Matched_NAICS'] if pd.notna(
                row['Matched_NAICS']) else row['Original_NAICS'],
            axis=1
        )

        # Standardize all NAICS codes to USCIS format
        uscis_df['Standardized_Original_NAICS'] = uscis_df['Original_NAICS'].apply(
            self.convert_to_uscis_naics_format
        )

        uscis_df['Standardized_Matched_NAICS'] = uscis_df['Matched_NAICS'].apply(
            lambda x: self.convert_to_uscis_naics_format(
                x) if pd.notna(x) else None
        )

        uscis_df['Standardized_Final_NAICS'] = uscis_df['Final_NAICS'].apply(
            self.convert_to_uscis_naics_format
        )

        # Keep the Industry (NAICS) Code column unchanged for reference
        # Add a new column for the standardized version that will be used in analysis
        uscis_df['Industry_NAICS_Standardized'] = uscis_df['Standardized_Final_NAICS']

        # ########
        # uscis_df = self.assign_naics_based_on_highest_applications(uscis_df)
        # ########

        # Save processed data
        # output_file = os.path.join(
        #     self.output_dir, f"processed_uscis_{self.year}.csv")
        # uscis_df.to_csv(output_file, index=False)

        # When saving processed data
        output_file = os.path.join(
            self.output_dir, f"processed_uscis_{self.year}.csv")

        # Select only relevant columns
        relevant_columns = [
            'Fiscal Year',
            'Employer (Petitioner) Name',
            'Industry_NAICS_Standardized',
            'NAICS_Source',
            'Initial Approval',
            'Initial Denial',
            'Continuing Approval',
            'Continuing Denial'
        ]

        # Only keep columns that actually exist in your dataframe
        final_columns = [
            col for col in relevant_columns if col in uscis_df.columns]

        # Save only these columns to CSV
        uscis_df[final_columns].to_csv(output_file, index=False)

        logger.info(f"Saved processed data to {output_file}")

        # Create a diagnostic file showing only changed NAICS codes
        changed_naics_df = uscis_df[uscis_df['Original_NAICS']
                                    != uscis_df['Final_NAICS']]
        changed_naics_file = os.path.join(
            self.output_dir, f"changed_naics_{self.year}.csv")
        if not changed_naics_df.empty:
            # Select just the important columns for diagnostics
            diag_cols = [
                'Employer (Petitioner) Name', 'Petitioner State',
                'Original_NAICS', 'Matched_NAICS', 'Final_NAICS',
                'Standardized_Original_NAICS', 'Standardized_Matched_NAICS',
                'Standardized_Final_NAICS', 'NAICS_Source'
            ]
            # Only include columns that exist
            diag_cols = [
                col for col in diag_cols if col in changed_naics_df.columns]
            changed_naics_df[diag_cols].to_csv(changed_naics_file, index=True)
            logger.info(
                f"Saved {len(changed_naics_df)} changed NAICS records to {changed_naics_file}")

        logger.info("Processing completed successfully")
        return uscis_df

    def load_uscis_data(self):
        """Load USCIS data file"""
        logger.info(f"Loading USCIS data from {self.uscis_file_path}")

        try:
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(self.uscis_file_path,
                                 encoding='utf-16', sep='\t', low_memory=False)
            except:
                try:
                    df = pd.read_csv(self.uscis_file_path,
                                     encoding='utf-8', low_memory=False)
                except:
                    df = pd.read_csv(self.uscis_file_path, low_memory=False)

            # Ensure Fiscal Year column exists
            if 'Fiscal Year' not in df.columns and 'Fiscal Year   ' in df.columns:
                df.rename(
                    columns={'Fiscal Year   ': 'Fiscal Year'}, inplace=True)

            logger.info(
                f"Successfully loaded USCIS data with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading USCIS data: {e}")
            return None

    def extract_unique_employer_data(self):
        """Extract unique employer data from LCA file"""
        logger.info(
            f"Extracting unique employer data from {self.lca_file_path}")

        try:
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(self.lca_file_path, low_memory=False)
            except:
                try:
                    df = pd.read_csv(self.lca_file_path,
                                     delimiter='\t', low_memory=False)
                except:
                    df = pd.read_csv(
                        self.lca_file_path, encoding='utf-16', sep='\t', low_memory=False)

            # Check for required columns
            required_cols = [
                'LCA_CASE_EMPLOYER_NAME',
                'LCA_CASE_EMPLOYER_CITY',
                'LCA_CASE_EMPLOYER_STATE',
                'LCA_CASE_NAICS_CODE'
            ]

            # Check if all required columns exist
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in LCA data: {missing_cols}")
                # Try to find similar column names
                for missing_col in missing_cols:
                    for col in df.columns:
                        if missing_col.lower() in col.lower():
                            df.rename(columns={col: missing_col}, inplace=True)
                            logger.info(
                                f"Mapped column {col} to {missing_col}")

            # Select the columns of interest (only existing ones)
            available_cols = [
                col for col in required_cols if col in df.columns]
            df_selected = df[available_cols]

            # Drop duplicates to get unique combinations
            df_unique = df_selected.drop_duplicates()

            logger.info(
                f"Extracted {len(df_unique)} unique employer records from {len(df)} total records")

            # Save for reference
            output_file = os.path.join(
                self.output_dir, f"unique_lca_data_{self.year}.csv")
            df_unique.to_csv(output_file, index=False)

            return df_unique
        except Exception as e:
            logger.error(f"Error extracting unique employer data: {e}")
            return pd.DataFrame()

    def find_empty_naics(self, df):
        """Find rows with empty NAICS codes"""
        logger.info("Finding rows with empty NAICS codes")

        naics_col = 'Industry (NAICS) Code'
        # Find truly empty NAICS codes (NaN or empty string)
        empty_naics_df = df[df[naics_col].isna() | (df[naics_col] == '')]

        logger.info(f"Found {len(empty_naics_df)} rows with empty NAICS codes")
        return empty_naics_df

    def standardize_naics_format(self, df):
        """Convert all NAICS codes to USCIS tuple format ('XX', 'Description')"""
        logger.info("Standardizing NAICS code format to match USCIS format")

        naics_col = 'Industry (NAICS) Code'
        if naics_col not in df.columns:
            logger.warning(
                f"NAICS column '{naics_col}' not found in dataframe")
            return df

        # Create a new column to hold standardized NAICS codes
        df['Standardized_NAICS'] = df[naics_col].apply(
            self.convert_to_uscis_naics_format)

        # Replace original column with standardized values
        df[naics_col] = df['Standardized_NAICS']
        df.drop('Standardized_NAICS', axis=1, inplace=True)

        logger.info("NAICS code standardization completed")
        return df

    def convert_to_uscis_naics_format(self, naics_code):
        """Convert a NAICS code to USCIS format"""
        if pd.isna(naics_code) or naics_code == '':
            return "('Unknown', 'Unknown or Unclassified Industries')"

        # Check if it's already in USCIS format
        if isinstance(naics_code, str) and naics_code.startswith("('") and naics_code.endswith("')"):
            return naics_code

        # Process 6-digit NAICS from LCA
        try:
            # If it's a numeric NAICS code
            if isinstance(naics_code, (int, float)) or (isinstance(naics_code, str) and naics_code.strip().isdigit()):
                # Convert to string, handling floats if needed
                naics_str = str(int(float(naics_code)))

                # Get first 2 digits to determine sector
                sector_code = naics_str[:2]

                # Handle manufacturing (31-33)
                if sector_code in ['31', '32', '33']:
                    sector_code = '31-33'
                # Handle retail trade (44-45)
                elif sector_code in ['44', '45']:
                    sector_code = '44-45'
                # Handle transportation (48-49)
                elif sector_code in ['48', '49']:
                    sector_code = '48-49'

                # Get sector description
                if sector_code in COMBINED_SECTORS:
                    sector_desc = COMBINED_SECTORS[sector_code]
                elif sector_code in NAICS_SECTORS:
                    sector_desc = NAICS_SECTORS[sector_code]
                else:
                    sector_desc = 'Unknown or Unclassified Industries'

                return f"('{sector_code}', '{sector_desc}')"

            # If it's a string but not in proper format, try parsing
            elif isinstance(naics_code, str):
                # Check if it might be a tuple/list representation
                if ',' in naics_code and "'" in naics_code:
                    try:
                        # Try parsing as literal if it looks like a tuple
                        parsed = ast.literal_eval(naics_code)
                        if isinstance(parsed, tuple) and len(parsed) == 2:
                            return f"('{parsed[0]}', '{parsed[1]}')"
                    except:
                        pass

                # Try extracting numbers from the string
                numbers = re.findall(r'\d+', naics_code)
                if numbers:
                    sector_code = numbers[0][:2]

                    # Handle special cases
                    if sector_code in ['31', '32', '33']:
                        sector_code = '31-33'
                    elif sector_code in ['44', '45']:
                        sector_code = '44-45'
                    elif sector_code in ['48', '49']:
                        sector_code = '48-49'

                    # Get sector description
                    if sector_code in COMBINED_SECTORS:
                        sector_desc = COMBINED_SECTORS[sector_code]
                    elif sector_code in NAICS_SECTORS:
                        sector_desc = NAICS_SECTORS[sector_code]
                    else:
                        sector_desc = 'Unknown or Unclassified Industries'

                    return f"('{sector_code}', '{sector_desc}')"

        except Exception as e:
            logger.warning(f"Error converting NAICS code '{naics_code}': {e}")

        # Default if all else fails
        return "('Unknown', 'Unknown or Unclassified Industries')"


class H1BIndustryMatcher:
    def __init__(self, uscis_path, lca_path):
        logger.info("Initializing H1BIndustryMatcher...")
        self.uscis_path = uscis_path
        self.lca_path = lca_path
        self.uscis = None
        self.lca = None
        self.lca_lookup = {}
        self.lca_name_only_lookup = {}
        self.match_log = []
        self.stats = defaultdict(int)

        # NAICS sectors for conversion
        self.naics_sectors = NAICS_SECTORS
        self.combined_sectors = COMBINED_SECTORS

        self.abbrev_mapping = {
            "US": "UNITED STATES",
            "USA": "UNITED STATES",
            "U S": "UNITED STATES",
            "U S A": "UNITED STATES",
            "AMERICA": "UNITED STATES",
            "INTL": "INTERNATIONAL",
            "INT": "INTERNATIONAL",
            "TECH": "TECHNOLOGY",
            "TECHS": "TECHNOLOGIES",
            "SVCS": "SERVICES",
            "SVC": "SERVICE",
            "MGMT": "MANAGEMENT",
            "CORP": "CORPORATION",
            "GRP": "GROUP",
            "ASSOC": "ASSOCIATES",
            "ASSN": "ASSOCIATION",
            "SYS": "SYSTEMS",
            "SYST": "SYSTEMS",
            "&": "AND",
        }

        self._load_data()
        self._preprocess_data()
        self._build_lookup_tables()

    def _load_data(self):
        """Load USCIS and LCA data with error handling."""
        logger.info("Loading data...")
        try:
            self.uscis = pd.read_csv(self.uscis_path, low_memory=False)
            self.lca = pd.read_csv(self.lca_path, low_memory=False)

            # Only keep rows in USCIS where NAICS is empty
            if 'Industry (NAICS) Code' in self.uscis.columns:
                self.uscis = self.uscis[pd.isna(self.uscis['Industry (NAICS) Code']) |
                                        (self.uscis['Industry (NAICS) Code'] == '')]

            logger.info(
                f"USCIS data shape after filtering empty NAICS: {self.uscis.shape}")
            logger.info(f"LCA data shape: {self.lca.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _normalize_name(self, name):
        """Advanced name normalization with better handling of variations."""
        if pd.isna(name):
            return '', ''

        # Convert to string and uppercase
        name = str(name).upper()

        # Replace ampersands and special characters
        name = name.replace('&', ' AND ')

        # Remove all punctuation
        name = re.sub(r'[^\w\s]', ' ', name)

        # Remove legal entities and common suffixes
        name = re.sub(r'\b(INC|LLC|LTD|CORP|CO|NA|LP|LLP|GROUP|COMPANY|CORPORATION|LIMITED|PC|PA|PLLC)\b\.?',
                      '', name)

        # Remove common business words
        name = re.sub(r'\b(THE|OF|A|FOR)\b', '', name)

        # Expand abbreviations
        words = name.split()
        expanded_words = []

        for word in words:
            if word in self.abbrev_mapping:
                expanded_words.append(self.abbrev_mapping[word])
            else:
                expanded_words.append(word)

        name = ' '.join(expanded_words)

        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Get core name (first 3 words often contain the most identifying information)
        core_words = name.split()[:3] if len(
            name.split()) >= 3 else name.split()
        core_name = ' '.join(core_words)

        return name, core_name

    def _preprocess_data(self):
        """Standardize company names and addresses."""
        logger.info("Preprocessing data...")

        # Clean company names - using a safer approach
        self.uscis['clean_name'] = ''
        self.uscis['core_name'] = ''
        self.lca['clean_name'] = ''
        self.lca['core_name'] = ''

        # Process USCIS names
        for idx, row in self.uscis.iterrows():
            try:
                if 'Employer (Petitioner) Name' in row:
                    clean, core = self._normalize_name(
                        row['Employer (Petitioner) Name'])
                    self.uscis.at[idx, 'clean_name'] = clean
                    self.uscis.at[idx, 'core_name'] = core
            except Exception as e:
                logger.error(f"Error normalizing USCIS name at idx {idx}: {e}")
                self.uscis.at[idx, 'clean_name'] = ''
                self.uscis.at[idx, 'core_name'] = ''

        # Process LCA names
        for idx, row in self.lca.iterrows():
            try:
                if 'LCA_CASE_EMPLOYER_NAME' in row:
                    clean, core = self._normalize_name(
                        row['LCA_CASE_EMPLOYER_NAME'])
                    self.lca.at[idx, 'clean_name'] = clean
                    self.lca.at[idx, 'core_name'] = core
            except Exception as e:
                logger.error(f"Error normalizing LCA name at idx {idx}: {e}")
                self.lca.at[idx, 'clean_name'] = ''
                self.lca.at[idx, 'core_name'] = ''

        # Extract state
        if 'Petitioner State' in self.uscis.columns:
            self.uscis['state'] = self.uscis['Petitioner State'].str.strip(
            ).str.upper().fillna('')
        else:
            self.uscis['state'] = ''

        if 'LCA_CASE_EMPLOYER_STATE' in self.lca.columns:
            self.lca['state'] = self.lca['LCA_CASE_EMPLOYER_STATE'].str.strip(
            ).str.upper().fillna('')
        else:
            self.lca['state'] = ''

        # Drop rows with empty names
        self.uscis = self.uscis[self.uscis['clean_name']
                                != ''].reset_index(drop=True)
        self.lca = self.lca[self.lca['clean_name']
                            != ''].reset_index(drop=True)

        logger.info("Data preprocessing completed.")

    def _build_lookup_tables(self):
        """Build efficient lookup tables for matching."""
        logger.info("Building lookup tables...")

        # Group LCA data by state and name for faster lookup
        for _, row in self.lca.iterrows():
            state = row['state']
            full_name = row['clean_name']
            core_name = row['core_name']

            # Check if LCA_CASE_NAICS_CODE exists in the row
            if 'LCA_CASE_NAICS_CODE' in row:
                naics = row['LCA_CASE_NAICS_CODE']
            else:
                continue

            # Skip rows with empty NAICS
            if pd.isna(naics) or naics == '':
                continue

            # Create keys for different matching strategies
            state_full_key = (state, full_name)
            state_core_key = (state, core_name)

            # Add to state+name lookup
            if state_full_key not in self.lca_lookup:
                self.lca_lookup[state_full_key] = []
            self.lca_lookup[state_full_key].append(naics)

            if state_core_key not in self.lca_lookup:
                self.lca_lookup[state_core_key] = []
            self.lca_lookup[state_core_key].append(naics)

            # Add to name-only lookup
            if full_name not in self.lca_name_only_lookup:
                self.lca_name_only_lookup[full_name] = []
            self.lca_name_only_lookup[full_name].append(naics)

            if core_name not in self.lca_name_only_lookup:
                self.lca_name_only_lookup[core_name] = []
            self.lca_name_only_lookup[core_name].append(naics)

        logger.info(
            f"Built lookup tables with {len(self.lca_lookup)} state+name combinations and {len(self.lca_name_only_lookup)} name-only entries.")

    def _get_best_naics(self, naics_list):
        """Get the most frequent NAICS code from a list and convert to USCIS format."""
        if not naics_list:
            return None, 0.0

        # Count NAICS codes
        naics_counts = {}
        for naics in naics_list:
            if naics not in naics_counts:
                naics_counts[naics] = 0
            naics_counts[naics] += 1

        # Find the most frequent
        best_naics_raw = max(naics_counts.items(), key=lambda x: x[1])
        confidence = best_naics_raw[1] / len(naics_list)

        # Convert to USCIS format
        best_naics = self._convert_to_uscis_naics_format(best_naics_raw[0])

        return best_naics, confidence

    def _convert_to_uscis_naics_format(self, naics_code):
        """Convert a NAICS code to USCIS format ('XX', 'Description')"""
        if pd.isna(naics_code) or naics_code == '':
            return "('Unknown', 'Unknown or Unclassified Industries')"

        try:
            naics_str = str(naics_code)

            # Get first 2 digits to determine sector
            sector_code = naics_str[:2]

            # Handle manufacturing (31-33)
            if sector_code in ['31', '32', '33']:
                sector_code = '31-33'
            # Handle retail trade (44-45)
            elif sector_code in ['44', '45']:
                sector_code = '44-45'
            # Handle transportation (48-49)
            elif sector_code in ['48', '49']:
                sector_code = '48-49'

            # Get sector description
            if sector_code in self.combined_sectors:
                sector_desc = self.combined_sectors[sector_code]
            elif sector_code in self.naics_sectors:
                sector_desc = self.naics_sectors[sector_code]
            else:
                sector_desc = 'Unknown or Unclassified Industries'

            return f"('{sector_code}', '{sector_desc}')"
        except:
            return "('Unknown', 'Unknown or Unclassified Industries')"

    def _token_match(self, uscis_name, lca_names, min_common=2, min_ratio=0.5):
        """Perform token-based matching between company names."""
        if not uscis_name:
            return []

        uscis_tokens = set(uscis_name.split())
        if len(uscis_tokens) == 0:
            return []

        matches = []

        for lca_name in lca_names:
            if not lca_name:
                continue

            lca_tokens = set(lca_name.split())
            if len(lca_tokens) == 0:
                continue

            # Calculate overlap
            common_tokens = uscis_tokens.intersection(lca_tokens)

            if len(common_tokens) >= min_common:  # At least N words in common
                ratio = len(common_tokens) / \
                    max(len(uscis_tokens), len(lca_tokens))
                if ratio >= min_ratio:  # At least 50% overlap by default
                    matches.append(lca_name)

        return matches

    def match_industries(self):
        """Main matching workflow using multiple matching strategies."""
        logger.info("Starting industry matching...")

        # Prepare the results dataframe
        results = self.uscis.copy()

        # Track statistics
        total_empty = len(results)
        matched = 0
        match_types = defaultdict(int)

        # Store the original index mapping if it exists
        if 'original_index' in results.columns:
            index_mapping = results['original_index'].tolist()
        else:
            index_mapping = list(range(len(results)))
            results['original_index'] = index_mapping

        # For each row in USCIS with empty NAICS
        for idx, row in tqdm(results.iterrows(), total=len(results), desc="Matching industries"):
            state = row['state']
            full_name = row['clean_name']
            core_name = row['core_name']
            orig_name = row['Employer (Petitioner) Name'] if 'Employer (Petitioner) Name' in row else ''
            naics_list = []
            match_type = None

            # Try exact match first (state + full name)
            key = (state, full_name)
            if key in self.lca_lookup:
                naics_list = self.lca_lookup[key]
                match_type = "exact_state_full"

            # Try state + core name
            if not naics_list:
                key = (state, core_name)
                if key in self.lca_lookup:
                    naics_list = self.lca_lookup[key]
                    match_type = "exact_state_core"

            # Try full name across all states
            if not naics_list and full_name in self.lca_name_only_lookup:
                naics_list = self.lca_name_only_lookup[full_name]
                match_type = "exact_full_any_state"

            # Try core name across all states
            if not naics_list and core_name in self.lca_name_only_lookup:
                naics_list = self.lca_name_only_lookup[core_name]
                match_type = "exact_core_any_state"

            # Try token-based matching if still no match
            if not naics_list:
                # Look for companies with at least 2 common words
                lca_names = list(self.lca_name_only_lookup.keys())
                matching_lca_names = self._token_match(full_name, lca_names)

                for lca_name in matching_lca_names:
                    naics_list.extend(self.lca_name_only_lookup[lca_name])

                if naics_list:
                    match_type = "token_based"

            # Get best NAICS code in USCIS format
            if naics_list:
                naics_code, confidence = self._get_best_naics(naics_list)
                results.at[idx, 'Industry (NAICS) Code'] = naics_code
                match_types[match_type] += 1

                # Log the match with the original index
                original_idx = index_mapping[idx]
                self.match_log.append({
                    'uscis_index': idx,
                    'original_index': original_idx,
                    'employer_name': orig_name,
                    'clean_name': full_name,
                    'core_name': core_name,
                    'state': state,
                    'naics_code': naics_code,
                    'confidence': confidence,
                    'match_count': len(naics_list),
                    'match_type': match_type
                })

                matched += 1

        # Update statistics
        self.stats['total_empty_naics'] = total_empty
        self.stats['matched'] = matched
        self.stats['match_rate'] = matched / \
            total_empty if total_empty > 0 else 0

        # Add match type statistics
        for match_type, count in match_types.items():
            self.stats[f'match_type_{match_type}'] = count

        logger.info(
            f"Matched {matched} out of {total_empty} records ({self.stats['match_rate']:.2%})")
        logger.info(f"Match types: {dict(match_types)}")

        return results


# Process multiple years
def process_multiple_years(years_data, output_dir="./output"):
    """Process multiple years of data"""
    logger.info(f"Processing {len(years_data)} years of data")

    results = {}
    for year, (uscis_file, lca_file) in years_data.items():
        logger.info(f"Processing year {year}")
        processor = H1BProcessor(uscis_file, lca_file, output_dir)
        result_df = processor.run_workflow()
        results[year] = result_df

    logger.info("All years processed successfully")
    return results


if __name__ == "__main__":
    # Base directory
    base_dir = '/Users/kietnguyen/Downloads/ECON395'
    output_dir = os.path.join(base_dir, 'output_v4_total')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define all years from 2010 to 2022
    years_data = {}
    for year in range(2010, 2023):  # 2010 to 2022 inclusive
        uscis_file = os.path.join(base_dir, f'Employer_Information_{year}.csv')
        lca_file = os.path.join(base_dir, f'LCA_FY{year}.csv')

        # Check if both files exist before adding to the list
        if os.path.exists(uscis_file) and os.path.exists(lca_file):
            years_data[str(year)] = (uscis_file, lca_file)
            print(f"Found data files for year {year}")
        else:
            print(f"Missing files for year {year}, skipping")

    print(f"Found data for {len(years_data)} years")

    # Process each year
    for year, (uscis_file, lca_file) in years_data.items():
        print(f"\n--- Processing year {year} ---")
        processor = H1BProcessor(uscis_file, lca_file, output_dir)
        processed_df = processor.run_workflow()
        print(f"Completed processing for year {year}")

    print(f"\nAll processing complete! Output saved to {output_dir}")
    print(f"Log file: {log_filename}")
