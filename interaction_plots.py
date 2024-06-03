#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:58:48 2024

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu

# Load the data from the CSV file
file_path = '/Users/emmabarash/Desktop/out.csv'
data = pd.read_csv(file_path)

# Ensure the data is correctly loaded
data.head()

# Create the 'Taste_Type' column based on 'Concentration'
data['Taste_Type'] = data['Concentration'].apply(lambda x: 'Palatable' if x in ['nacl_0.1M', 'suc_0.3M'] else 'Unpalatable')

# Define the window size for the rolling average
window_size = 10

# Calculate the correct delivery rate for each trial within a session
data['Corrected_Delivery_Rate'] = data['group_cumsum'] / data['Time']

# Define the time intervals for analysis (e.g., every 100 seconds)
time_intervals = range(0, 3601, 100)

# Initialize lists to store results
time_points = []
rate_differences = []
rate_p_values = []

# Calculate delivery rates for each time interval
for t in time_intervals:
    # Filter data for the current time interval
    interval_data = data[(data['Time'] >= t) & (data['Time'] < t + 100)]
    
    # Calculate average delivery rates for palatable and unpalatable tastes
    palatable_rates = interval_data[interval_data['Taste_Type'] == 'Palatable']['Corrected_Delivery_Rate']
    unpalatable_rates = interval_data[interval_data['Taste_Type'] == 'Unpalatable']['Corrected_Delivery_Rate']
    
    # Calculate the difference in average delivery rates
    rate_diff = palatable_rates.mean() - unpalatable_rates.mean()
    
    # Perform Mann-Whitney U test to check for significance
    if len(palatable_rates) > 0 and len(unpalatable_rates) > 0:
        t_stat, p_value = mannwhitneyu(palatable_rates, unpalatable_rates, alternative='two-sided')
    else:
        t_stat, p_value = (None, None)
    
    # Store the results
    time_points.append(t)
    rate_differences.append(rate_diff)
    rate_p_values.append(p_value)

# Create a DataFrame to store and display results
rate_discriminability_df = pd.DataFrame({
    'Time': time_points,
    'Rate_Difference': rate_differences,
    'p-value': rate_p_values
})

# Number of tests (time bins)
num_tests = len(rate_discriminability_df)

# Original significance level
alpha = 0.05

# Adjusted significance level
alpha_adjusted = alpha / num_tests

# Apply the Bonferroni correction
rate_discriminability_df['Bonferroni_Significant'] = rate_discriminability_df['p-value'] < alpha_adjusted

# Display the DataFrame
rate_discriminability_df.head()

# Perform a two-way ANOVA to test for interaction between time and taste type
# Prepare the data for ANOVA
anova_data = data.copy()

# Create time bins for the analysis
anova_data['Time_Bin'] = pd.cut(anova_data['Time'], bins=range(0, 3700, 100), labels=range(0, 3600, 100))

# Drop rows with missing values after binning
anova_data = anova_data.dropna(subset=['Time_Bin'])

# Fit the model
model = ols('Corrected_Delivery_Rate ~ C(Time_Bin) * C(Taste_Type)', data=anova_data).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the ANOVA table
anova_table

# Create the interaction plot
plt.figure(figsize=(14, 7))

# Calculate mean delivery rates for each combination of time bin and taste type
interaction_data = anova_data.groupby(['Time_Bin', 'Taste_Type'])['Corrected_Delivery_Rate'].mean().reset_index()

# Plot interaction plot
sns.lineplot(data=interaction_data, x='Time_Bin', y='Corrected_Delivery_Rate', hue='Taste_Type', marker='o')

plt.xlabel('Time Bin (seconds)')
plt.ylabel('Average Delivery Rate')
plt.title('Interaction Plot: Delivery Rates Over Time by Taste Type')
plt.legend(title='Taste Type')
plt.grid(True)
plt.show()

# Highlight significant time bins after Bonferroni correction in the interaction plot
plt.figure(figsize=(14, 7))

# Plot interaction plot with significant time bins highlighted
sns.lineplot(data=interaction_data, x='Time_Bin', y='Corrected_Delivery_Rate', hue='Taste_Type', marker='o')

# Highlight significant time bins after Bonferroni correction
bonferroni_significant_bins = rate_discriminability_df[rate_discriminability_df['Bonferroni_Significant']]
plt.scatter(bonferroni_significant_bins['Time'], 
            [interaction_data[(interaction_data['Time_Bin'] == bin) & (interaction_data['Taste_Type'] == 'Palatable')]['Corrected_Delivery_Rate'].mean() 
             for bin in bonferroni_significant_bins['Time']], color='blue', s=100, label='Significant Palatable')
plt.scatter(bonferroni_significant_bins['Time'], 
            [interaction_data[(interaction_data['Time_Bin'] == bin) & (interaction_data['Taste_Type'] == 'Unpalatable')]['Corrected_Delivery_Rate'].mean() 
             for bin in bonferroni_significant_bins['Time']], color='red', s=100, label='Significant Unpalatable')

plt.xlabel('Time Bin (seconds)')
plt.ylabel('Average Delivery Rate')
plt.title('Interaction Plot: Delivery Rates Over Time by Taste Type with Bonferroni Significant Time Bins Highlighted')
plt.legend(title='Taste Type')
plt.grid(True)
plt.show()
