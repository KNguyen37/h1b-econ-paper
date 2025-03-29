# H-1B Visa Restrictions and Wage Dynamics

## "Buy American, Hire American": Did the 2017 Visa Restrictions Affect Wages in H-1B Dependent Industries?

This repository contains the data, code, and analysis for our research examining how the Trump administration's "Buy American, Hire American" executive order affected wage dynamics across industries with varying degrees of H-1B dependency from 2012-2020.

### Research Abstract

This study examines how the Trump administration's "Buy American, Hire American" executive order affected wage dynamics across industries with varying degrees of H-1B dependency from 2012-2020. Following this policy, H-1B denial rates rose sharply from 6% in 2016 to 18% by 2018, creating a significant labor supply shock. This research investigates whether this shock propagated differently through labor markets depending on industries' reliance on foreign skilled workers.

Using a Difference-in-Differences framework, we analyze how industries with high versus low H-1B dependency experienced differential wage responses to increased denial rates. This approach isolates the causal effect of reduced access to foreign talent on domestic wage structures, controlling for broader economic trends. To enable this analysis, we have undertaken extensive data preparation, systematically standardizing and matching administrative records from USCIS and Labor Condition Applications that were not originally structured for statistical analysis. We successfully matched 95% of employers with previously unclassified industry codes, creating reliable measures of H-1B dependency across sectors.

### Repository Structure

```
h1b-econ-paper/
├── analysis_result/        # Results of statistical analysis
│   └── graphs/             # Visualizations and plots
├── clean_combined_data/    # Processed and merged datasets
├── h1b_cleaned_data/       # Cleaned H-1B petition data
├── original_data/          # Raw data files
├── preprocessing_Python/   # Python scripts for data preparation
│   ├── approval_percentage.py
│   ├── denial_rate_v3.py
│   └── preprocessing_v3.py
└── process_Rstudio/        # R scripts for statistical analysis
    ├── data_exploration.Rmd
    ├── preprocessing_clean_combine.Rmd
    └── Regression.Rmd
```

### Data Sources

The analysis integrates three primary data sources covering 2012-2020:
1. **USCIS H-1B Petition Data**: 3.2 million petition records with approval/denial decisions and employer information
2. **Labor Condition Application (LCA) Data**: 6.8 million applications with standardized NAICS industry codes and detailed employer information
3. **BLS Wage Data**: Industry-level quarterly wage data aligned with standardized NAICS codes

### Methodology

Our research employs a Difference-in-Differences approach with the following specification:

```
Wage_it = β₀ + β₁(DenialRate_it) + β₂(Dependent_i) + β₃(DenialRate_it × H1BDependent_i) + X_it + μ_i + λ_t + ε_it
```

Where:
- `DenialRate_it` is the quarterly H-1B denial rate
- `Dependent_i` is a binary indicator for high H-1B dependency
- `X_it` is a vector of controls including quarterly employment growth, inflation, productivity, and offshoring rate
- `μ_i` and `λ_t` are industry and time fixed effects

Our coefficient of interest, β₃, captures the differential wage effect of denial rates on high vs. low H-1B-dependent industries.

### Key Files

1. **Data Preprocessing**:
   - `preprocessing_v3.py`: Processes USCIS and LCA data, standardizes NAICS codes
   - `denial_rate_v3.py`: Calculates denial rates by industry
   - `approval_percentage.py`: Analyzes approval percentages by industry

2. **Statistical Analysis**:
   - `preprocessing_clean_combine.Rmd`: Merges and prepares datasets
   - `data_exploration.Rmd`: Exploratory data analysis
   - `Regression.Rmd`: Difference-in-Differences regression models

### Results

Our analysis shows that the "Buy American, Hire American" policy had significant differential effects on wages across industries based on their reliance on H-1B workers. The event study analysis confirms parallel trends before the policy implementation and divergence afterward.

For detailed findings, see the `/analysis_result` directory.

### Requirements

#### Python Dependencies
- pandas
- numpy
- tqdm
- re
- matplotlib
- scikit-learn

#### R Dependencies
- dplyr
- tidyr
- stringr
- ggplot2
- plm
- lmtest
- sandwich
- fixest
- corrplot

### Authors

- Kiet Nguyen - Case Western Reserve University
- Lien Tran - Case Western Reserve University

### Citation

If you use this code or data in your research, please cite:

```
Nguyen, K., & Tran, L. (2025). "Buy American, Hire American": Did the 2017 Visa Restrictions
Affect Wages in H-1B Dependent Industries?. Case Western Reserve University.
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

We thank professor Mark Schweitzer and Jenny Hawkins as well as TAs of ECON395 and the Federal Reserve Bank of Cleveland's Economic Scholars Program for their support. 
