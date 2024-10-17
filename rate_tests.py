#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:56:25 2024

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm



# Load the data from the CSV file
file_path = '/Users/emmabarash/Desktop/out.csv'
data = pd.read_csv(file_path)

# Ensure the data is correctly loaded
data.head()

# Create the 'Taste_Type' column based on 'Concentration'
data['Taste_Type'] = data['Concentration'].apply(lambda x: 'Palatable' if x in ['nacl_0.1M', 'suc_0.3M'] else 'Unpalatable')

# Define the window size for the rolling average
window_size = 10

data.round({'Time': 0})

# Calculate the correct delivery rate for each trial within a session
data['Corrected_Delivery_Rate'] = data['group_cumsum'] / data['Time']

# Function to apply rolling average within each group
def rolling_mean_within_group(df, window_size):
    df['Rolling_Avg_Rate'] = df['Corrected_Delivery_Rate'].rolling(window=window_size, min_periods=1).mean()
    return df

# Apply the rolling average within each AnID and Date group
data = data.groupby(['AnID', 'Date', 'TasteID']).apply(rolling_mean_within_group, window_size=window_size)

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
    palatable_rates = interval_data[interval_data['Taste_Type'] == 'Palatable']['Rolling_Avg_Rate']
    unpalatable_rates = interval_data[interval_data['Taste_Type'] == 'Unpalatable']['Rolling_Avg_Rate']
    
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

# Plot the p-values for each time point before Bonferroni correction
plt.figure(figsize=(14, 7))
plt.plot(rate_discriminability_df['Time'], rate_discriminability_df['p-value'], marker='o', linestyle='-')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
plt.xlabel('Time (seconds)')
plt.ylabel('p-value')
plt.title('P-values of Delivery Rate Differences Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the rolling average rates for palatable and unpalatable tastes over time
rolling_avg_palatable = []
rolling_avg_unpalatable = []

for t in time_intervals:
    # Filter data for the current time interval
    interval_data = data[(data['Time'] >= t) & (data['Time'] < t + 100)]
    
    # Calculate average rolling rates for palatable and unpalatable tastes
    palatable_avg = interval_data[interval_data['Taste_Type'] == 'Palatable']['Rolling_Avg_Rate'].mean()
    unpalatable_avg = interval_data[interval_data['Taste_Type'] == 'Unpalatable']['Rolling_Avg_Rate'].mean()
    
    # Store the results
    rolling_avg_palatable.append(palatable_avg)
    rolling_avg_unpalatable.append(unpalatable_avg)

# Plot the rolling window average rates for palatable and unpalatable tastes
plt.figure(figsize=(14, 7))
plt.plot(time_intervals, rolling_avg_palatable, marker='o', linestyle='-', label='Palatable')
plt.plot(time_intervals, rolling_avg_unpalatable, marker='o', linestyle='-', label='Unpalatable')
plt.xlabel('Time (seconds)')
plt.ylabel('Rolling Average Delivery Rate')
plt.title('Rolling Window Average Delivery Rates Over Time')
plt.legend()
plt.grid(True)
plt.show()

#############
###############
#############

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

# Group the data by session time and taste type to calculate the sum of cumulative deliveries
cumulative_deliveries_corrected = data.groupby(['Time', 'Taste_Type'])['group_cumsum'].sum().reset_index()

# Separate the data into palatable and unpalatable
palatable_cumulative_corrected = cumulative_deliveries_corrected[cumulative_deliveries_corrected['Taste_Type'] == 'Palatable']
unpalatable_cumulative_corrected = cumulative_deliveries_corrected[cumulative_deliveries_corrected['Taste_Type'] == 'Unpalatable']

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

# Highlight significant time bins in the interaction plot
plt.figure(figsize=(14, 7))

# Plot interaction plot with significant time bins highlighted
sns.lineplot(data=interaction_data, x='Time_Bin', y='Corrected_Delivery_Rate', hue='Taste_Type', marker='o')

# Highlight significant time bins
significant_bins = rate_discriminability_df[rate_discriminability_df['p-value'] < 0.05]
plt.scatter(significant_bins['Time'], [interaction_data[(interaction_data['Time_Bin'] == bin) & (interaction_data['Taste_Type'] == 'Palatable')]['Corrected_Delivery_Rate'].mean() for bin in significant_bins['Time']], color='blue', s=100, label='Significant Palatable')
plt.scatter(significant_bins['Time'], [interaction_data[(interaction_data['Time_Bin'] == bin) & (interaction_data['Taste_Type'] == 'Unpalatable')]['Corrected_Delivery_Rate'].mean() for bin in significant_bins['Time']], color='red', s=100, label='Significant Unpalatable')

plt.xlabel('Time Bin (seconds)')
plt.ylabel('Average Delivery Rate')
plt.title('Interaction Plot: Delivery Rates Over Time by Taste Type with Significant Time Bins Highlighted')
plt.legend(title='Taste Type')
plt.grid(True)
plt.show()

#############
###############
#############

# Define a function to find the last trial time for a specific taste
def find_last_trial_time(df, taste):
    # Filter the data for the specific taste
    taste_data = df[df['TasteID'] == taste]
    # Find the last trial time
    last_trial_time = taste_data.groupby(['AnID', 'Date'])['Time'].max().reset_index()
    return last_trial_time

# Define a function to find the last trial time for a specific taste, with a filter for minimum time
def find_last_trial_time_with_filter(df, taste, min_time):
    # Filter the data for the specific taste and time
    taste_data = df[(df['TasteID'] == taste) & (df['Time'] >= min_time)]
    # Find the last trial time
    last_trial_time = taste_data.groupby(['AnID', 'Date'])['Time'].max().reset_index()
    return last_trial_time

# Find the last trial time for 'suc' and 'nacl_l' with a minimum time of 1500 seconds
last_trial_suc_filtered = find_last_trial_time_with_filter(data, 'suc', 1500)
last_trial_nacl_l_filtered = find_last_trial_time_with_filter(data, 'nacl_l', 1500)
# For 'nacl_h', we do not apply the filter
last_trial_nacl_h = find_last_trial_time(data, 'nacl_h')

# Calculate the average time of the last trials for each taste
average_last_trial_suc_filtered = last_trial_suc_filtered['Time'].mean()
average_last_trial_nacl_l_filtered = last_trial_nacl_l_filtered['Time'].mean()
average_last_trial_nacl_h = last_trial_nacl_h['Time'].mean()

print(f"Average time of the last trial for 'suc': {average_last_trial_suc_filtered} seconds")
print(f"Average time of the last trial for 'nacl_l': {average_last_trial_nacl_l_filtered} seconds")
print(f"Average time of the last trial for 'nacl_h': {average_last_trial_nacl_h} seconds")

#############
###############
#############


# Ensure 'Taste_Delivery' is numeric
data['Taste_Delivery'] = pd.to_numeric(data['Taste_Delivery'], errors='coerce')

# Calculate the delivery rate for each taste at each trial
data['suc_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'suc' else 0, axis=1)
data['nacl_l_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'nacl_l' else 0, axis=1)
data['nacl_h_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'nacl_h' else 0, axis=1)

# Ensure the delivery rate columns are numeric
data['suc_Delivery_Rate'] = pd.to_numeric(data['suc_Delivery_Rate'], errors='coerce')
data['nacl_l_Delivery_Rate'] = pd.to_numeric(data['nacl_l_Delivery_Rate'], errors='coerce')
data['nacl_h_Delivery_Rate'] = pd.to_numeric(data['nacl_h_Delivery_Rate'], errors='coerce')

# Calculate cumulative delivery rates
data['cumulative_suc_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['suc_Delivery_Rate'].cumsum()
data['cumulative_nacl_l_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['nacl_l_Delivery_Rate'].cumsum()
data['cumulative_nacl_h_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['nacl_h_Delivery_Rate'].cumsum()

# Calculate total delivery rate at each trial
data['total_Delivery_Rate'] = data['cumulative_suc_Delivery_Rate'] + data['cumulative_nacl_l_Delivery_Rate'] + data['cumulative_nacl_h_Delivery_Rate']

# Compute preference scores
data['suc_Preference_Score'] = data['cumulative_suc_Delivery_Rate'] / data['total_Delivery_Rate']
data['nacl_l_Preference_Score'] = data['cumulative_nacl_l_Delivery_Rate'] / data['total_Delivery_Rate']
data['nacl_h_Preference_Score'] = data['cumulative_nacl_h_Delivery_Rate'] / data['total_Delivery_Rate']

# Replace NaN values (resulting from division by zero) with 0
data['suc_Preference_Score'].fillna(0, inplace=True)
data['nacl_l_Preference_Score'].fillna(0, inplace=True)
data['nacl_h_Preference_Score'].fillna(0, inplace=True)

# Plot the preference scores throughout the session
plt.figure(figsize=(14, 7))
plt.plot(data['Trial_Num'], data['suc_Preference_Score'], label='suc Preference Score', alpha=0.7)
plt.plot(data['Trial_Num'], data['nacl_l_Preference_Score'], label='nacl_l Preference Score', alpha=0.7)
plt.plot(data['Trial_Num'], data['nacl_h_Preference_Score'], label='nacl_h Preference Score', alpha=0.7)

plt.xlabel('Trial Number')
plt.ylabel('Preference Score')
plt.title('Preference Scores Throughout the Session')
plt.legend()
plt.grid(True)
plt.show()

#dan's versini in sns
xmin = data['Trial_Num'].min()
xmax = data['Trial_Num'].max()

fig, ax = plt.subplots(1,1)
sns.lineplot(data = data, x = 'Trial_Num', y='suc_Preference_Score', ax = ax)
sns.lineplot(data = data, x = 'Trial_Num', y='nacl_l_Preference_Score', ax = ax)
sns.lineplot(data = data, x = 'Trial_Num', y='nacl_h_Preference_Score', ax = ax)
#to add to this figure, use the ax handle
ax.hlines(0.333, linestyles='dashed',xmin = xmin, xmax = xmax)
plt.show()

# Reshape data for ANOVA
preference_scores = pd.melt(data, id_vars=['Trial_Num'], value_vars=['suc_Preference_Score', 'nacl_l_Preference_Score', 'nacl_h_Preference_Score'],
                            var_name='Taste', value_name='Preference_Score')

# Rename the Taste categories for better readability
preference_scores['Taste'] = preference_scores['Taste'].replace({'suc_Preference_Score': 'suc', 'nacl_l_Preference_Score': 'nacl_l', 'nacl_h_Preference_Score': 'nacl_h'})

# Perform ANOVA
anova_model = ols('Preference_Score ~ C(Taste)', data=preference_scores).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Perform post-hoc test if ANOVA is significant
if anova_table['PR(>F)'][0] < 0.05:
    tukey = pairwise_tukeyhsd(endog=preference_scores['Preference_Score'], groups=preference_scores['Taste'], alpha=0.05)
    print(tukey)
else:
    print("No significant differences found between the groups.")
    
# Prepare data for ANOVA
preference_scores = pd.melt(data, id_vars=['Trial_Num'], value_vars=['suc_Preference_Score', 'nacl_l_Preference_Score', 'nacl_h_Preference_Score'],
                            var_name='Taste', value_name='Preference_Score')

# Rename the Taste categories for better readability
preference_scores['Taste'] = preference_scores['Taste'].replace({'suc_Preference_Score': 'suc', 'nacl_l_Preference_Score': 'nacl_l', 'nacl_h_Preference_Score': 'nacl_h'})

# Perform ANOVA at each trial
significant_trials = []
for trial_num in preference_scores['Trial_Num'].unique():
    trial_data = preference_scores[preference_scores['Trial_Num'] == trial_num]
    # Ensure we have enough data points for each taste
    if all(trial_data['Taste'].value_counts() >= 2):  # Require at least 2 observations per taste
        anova_model = ols('Preference_Score ~ C(Taste)', data=trial_data).fit()
        try:
            anova_table = anova_lm(anova_model, typ=2)
            if anova_table['PR(>F)'][0] < 0.05:
                significant_trials.append(trial_num)
        except ValueError as e:
            print(f"Skipping trial {trial_num} due to insufficient data: {e}")

print("Trials with significant differences in preference scores:", significant_trials)
#dan's versini in sns
xmin = data['Trial_Num'].min()
xmax = data['Trial_Num'].max()

fig, ax = plt.subplots(1,1)
sns.lineplot(data = data, x = 'Trial_Num', y='suc_Preference_Score', ax = ax)
sns.lineplot(data = data, x = 'Trial_Num', y='nacl_l_Preference_Score', ax = ax)
sns.lineplot(data = data, x = 'Trial_Num', y='nacl_h_Preference_Score', ax = ax)
#to add to this figure, use the ax handle
ax.hlines(0.333,xmin = xmin, xmax = xmax, color='black', linestyles='dashed')
ax.fill_between([min(significant_trials), max(significant_trials)], 0,0.5, facecolor='blue', alpha=0.2)
ax.legend(labels=['sucrose', '_', 'low NaCl', '_', 'high NaCl'])
plt.title('Average preference score over number of trials (N=3)')
plt.xlabel('Trial Number')
plt.ylabel('Preference Score')
plt.show()

#############
###############
#############

# Prepare data for ANOVA
preference_scores = pd.melt(data, id_vars=['Trial_Num'], value_vars=['suc_Preference_Score', 'nacl_l_Preference_Score', 'nacl_h_Preference_Score'],
                            var_name='Taste', value_name='Preference_Score')

# Rename the Taste categories for better readability
preference_scores['Taste'] = preference_scores['Taste'].replace({'suc_Preference_Score': 'suc', 'nacl_l_Preference_Score': 'nacl_l', 'nacl_h_Preference_Score': 'nacl_h'})

# Perform ANOVA at each trial and store p-values
p_values = []
for trial_num in preference_scores['Trial_Num'].unique():
    trial_data = preference_scores[preference_scores['Trial_Num'] == trial_num]
    if all(trial_data['Taste'].value_counts() >= 2):  # Require at least 2 observations per taste
        anova_model = ols('Preference_Score ~ C(Taste)', data=trial_data).fit()
        try:
            anova_table = anova_lm(anova_model, typ=2)
            p_values.append((trial_num, anova_table['PR(>F)'][0]))
        except ValueError as e:
            p_values.append((trial_num, 1.0))  # Assign a non-significant p-value
    else:
        p_values.append((trial_num, 1.0))  # Assign a non-significant p-value

# Convert p-values to DataFrame
p_values_df = pd.DataFrame(p_values, columns=['Trial_Num', 'p_value'])

# Identify significant trials
significant_trials = p_values_df[p_values_df['p_value'] < 0.05]['Trial_Num']

# Calculate mean and standard deviation of preference scores for each taste at each trial
mean_std_df = preference_scores.groupby(['Trial_Num', 'Taste']).agg({'Preference_Score': ['mean', 'std']}).reset_index()
mean_std_df.columns = ['Trial_Num', 'Taste', 'Mean_Preference_Score', 'Std_Preference_Score']

# Plot the preference scores with error bars
plt.figure(figsize=(14, 7))
for taste in mean_std_df['Taste'].unique():
    taste_data = mean_std_df[mean_std_df['Taste'] == taste]
    plt.errorbar(taste_data['Trial_Num'], taste_data['Mean_Preference_Score'], yerr=taste_data['Std_Preference_Score'], label=f'{taste} Preference Score', alpha=0.7, capsize=3)

# Highlight significant trials
for trial in significant_trials:
    plt.axvline(x=trial, color='red', linestyle='--', alpha=0.3)

plt.xlabel('Trial Number')
plt.ylabel('Preference Score')
plt.title('Preference Scores Throughout the Session with Significant Trials Highlighted')
plt.legend(title='Taste')
plt.grid(True)
plt.show()

#############
###############
#############


# Ensure 'Taste_Delivery' is numeric
data['Taste_Delivery'] = pd.to_numeric(data['Taste_Delivery'], errors='coerce')

# Calculate the delivery rate for each taste at each trial
data['suc_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'suc' else 0, axis=1)
data['nacl_l_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'nacl_l' else 0, axis=1)
data['nacl_h_Delivery_Rate'] = data.apply(lambda row: row['Taste_Delivery'] if row['TasteID'] == 'nacl_h' else 0, axis=1)

# Ensure the delivery rate columns are numeric
data['suc_Delivery_Rate'] = pd.to_numeric(data['suc_Delivery_Rate'], errors='coerce')
data['nacl_l_Delivery_Rate'] = pd.to_numeric(data['nacl_l_Delivery_Rate'], errors='coerce')
data['nacl_h_Delivery_Rate'] = pd.to_numeric(data['nacl_h_Delivery_Rate'], errors='coerce')

# Calculate cumulative delivery rates
data['cumulative_suc_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['suc_Delivery_Rate'].cumsum()
data['cumulative_nacl_l_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['nacl_l_Delivery_Rate'].cumsum()
data['cumulative_nacl_h_Delivery_Rate'] = data.groupby(['AnID', 'Date'])['nacl_h_Delivery_Rate'].cumsum()

# Calculate total delivery rate at each trial
data['total_Delivery_Rate'] = data['cumulative_suc_Delivery_Rate'] + data['cumulative_nacl_l_Delivery_Rate'] + data['cumulative_nacl_h_Delivery_Rate']

# Compute preference scores
data['suc_Preference_Score'] = data['cumulative_suc_Delivery_Rate'] / data['total_Delivery_Rate']
data['nacl_l_Preference_Score'] = data['cumulative_nacl_l_Delivery_Rate'] / data['total_Delivery_Rate']
data['nacl_h_Preference_Score'] = data['cumulative_nacl_h_Delivery_Rate'] / data['total_Delivery_Rate']

# Replace NaN values (resulting from division by zero) with 0
data['suc_Preference_Score'].fillna(0, inplace=True)
data['nacl_l_Preference_Score'].fillna(0, inplace=True)
data['nacl_h_Preference_Score'].fillna(0, inplace=True)

# Calculate average last trial for each taste
def find_last_trial(df, taste):
    taste_data = df[df['TasteID'] == taste]
    last_trial = taste_data.groupby(['AnID', 'Date'])['Trial_Num'].max().reset_index()
    return last_trial['Trial_Num'].mean()

average_last_trial_suc = find_last_trial(data, 'suc')
average_last_trial_nacl_l = find_last_trial(data, 'nacl_l')
average_last_trial_nacl_h = find_last_trial(data, 'nacl_h')

# Prepare data for ANOVA
preference_scores = pd.melt(data, id_vars=['Trial_Num'], value_vars=['suc_Preference_Score', 'nacl_l_Preference_Score', 'nacl_h_Preference_Score'],
                            var_name='Taste', value_name='Preference_Score')

# Rename the Taste categories for better readability
preference_scores['Taste'] = preference_scores['Taste'].replace({'suc_Preference_Score': 'suc', 'nacl_l_Preference_Score': 'nacl_l', 'nacl_h_Preference_Score': 'nacl_h'})

# Perform ANOVA at each trial and store p-values
p_values = []
for trial_num in preference_scores['Trial_Num'].unique():
    trial_data = preference_scores[preference_scores['Trial_Num'] == trial_num]
    if all(trial_data['Taste'].value_counts() >= 2):  # Require at least 2 observations per taste
        anova_model = ols('Preference_Score ~ C(Taste)', data=trial_data).fit()
        try:
            anova_table = anova_lm(anova_model, typ=2)
            p_values.append((trial_num, anova_table['PR(>F)'][0]))
        except ValueError as e:
            p_values.append((trial_num, 1.0))  # Assign a non-significant p-value
    else:
        p_values.append((trial_num, 1.0))  # Assign a non-significant p-value

# Convert p-values to DataFrame
p_values_df = pd.DataFrame(p_values, columns=['Trial_Num', 'p_value'])

# Identify significant trials
significant_trials = p_values_df[p_values_df['p_value'] < 0.05]['Trial_Num']

# Calculate mean and standard deviation of preference scores for each taste at each trial
mean_std_df = preference_scores.groupby(['Trial_Num', 'Taste']).agg({'Preference_Score': ['mean', 'std']}).reset_index()
mean_std_df.columns = ['Trial_Num', 'Taste', 'Mean_Preference_Score', 'Std_Preference_Score']

# Plot the preference scores with error bars
plt.figure(figsize=(14, 7))
for taste in mean_std_df['Taste'].unique():
    taste_data = mean_std_df[mean_std_df['Taste'] == taste]
    plt.errorbar(taste_data['Trial_Num'], taste_data['Mean_Preference_Score'], yerr=taste_data['Std_Preference_Score'], label=f'{taste} Preference Score', alpha=0.7, capsize=3)

# Highlight significant trials
for trial in significant_trials:
    plt.axvline(x=trial, color='red', linestyle='--', alpha=0.3)

# Add average last trial times as distinct markers and text annotations
plt.axvline(x=average_last_trial_suc, color='blue', linestyle=':', alpha=0.7)
plt.axvline(x=average_last_trial_nacl_l, color='green', linestyle=':', alpha=0.7)
plt.axvline(x=average_last_trial_nacl_h, color='purple', linestyle=':', alpha=0.7)

plt.scatter([average_last_trial_suc], [0.80], color='blue', s=100, zorder=5)
plt.scatter([average_last_trial_nacl_l], [0.70], color='green', s=100, zorder=5)
plt.scatter([average_last_trial_nacl_h], [0.80], color='purple', s=100, zorder=5)

plt.text(average_last_trial_suc, 0.85, f'{average_last_trial_suc:.2f}', color='blue', ha='center', va='bottom')
plt.text(average_last_trial_nacl_l, 0.75, f'{average_last_trial_nacl_l:.2f}', color='green', ha='center', va='bottom')
plt.text(average_last_trial_nacl_h, 0.75, f'{average_last_trial_nacl_h:.2f}', color='purple', ha='center', va='bottom')

plt.xlabel('Trial Number')
plt.ylabel('Preference Score')
plt.title('Preference Scores Throughout the Session with Significant Trials Highlighted')
plt.legend(title='Taste')
plt.ylim(0, 1)  # Ensure the y-axis ranges from 0 to 1
#plt.grid(True)
plt.show()
