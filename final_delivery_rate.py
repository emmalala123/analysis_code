#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:25:34 2024

@author: emmabarash
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the data from the CSV file
file_path = '/Users/emmabarash/Desktop/out.csv'
data = pd.read_csv(file_path)

# Ensure 'Taste_Delivery' and 'group_cumsum' are numeric
data['Taste_Delivery'] = pd.to_numeric(data['Taste_Delivery'], errors='coerce')
data['group_cumsum'] = pd.to_numeric(data['group_cumsum'], errors='coerce')

# Find the last trial for each taste
def find_last_trial_time(df, taste):
    taste_data = df[df['TasteID'] == taste]
    last_trial = taste_data.groupby(['AnID', 'Date'])['Time'].max().reset_index()
    return last_trial

last_trial_suc = find_last_trial_time(data, 'suc')
last_trial_nacl_l = find_last_trial_time(data, 'nacl_l')
last_trial_nacl_h = find_last_trial_time(data, 'nacl_h')

# Calculate final delivery rates for each taste
def calculate_final_delivery_rate(df, last_trials):
    merged_data = pd.merge(df, last_trials, on=['AnID', 'Date', 'Time'])
    final_rate = merged_data['group_cumsum'] / merged_data['Time']
    return final_rate

final_delivery_rate_suc = calculate_final_delivery_rate(data, last_trial_suc)
final_delivery_rate_nacl_l = calculate_final_delivery_rate(data, last_trial_nacl_l)
final_delivery_rate_nacl_h = calculate_final_delivery_rate(data, last_trial_nacl_h)

# Combine final delivery rates into a DataFrame
final_delivery_rates = pd.DataFrame({
    'AnID': last_trial_suc['AnID'],
    'Date': last_trial_suc['Date'],
    'suc': final_delivery_rate_suc,
    'nacl_l': final_delivery_rate_nacl_l,
    'nacl_h': final_delivery_rate_nacl_h
})

final_delivery_rates = final_delivery_rates.dropna()
# Calculate session rates for each session
session_rates = final_delivery_rates.groupby(['AnID', 'Date']).mean().reset_index()
# Perform t-tests for each pair of tastes
t_test_suc_vs_nacl_l = ttest_ind(session_rates['suc'], session_rates['nacl_l'])
t_test_suc_vs_nacl_h = ttest_ind(session_rates['suc'], session_rates['nacl_h'])
t_test_nacl_l_vs_nacl_h = ttest_ind(session_rates['nacl_l'], session_rates['nacl_h'])

# Print t-test results
print("T-test results for 'suc' vs 'nacl_l':", t_test_suc_vs_nacl_l)
print("T-test results for 'suc' vs 'nacl_h':", t_test_suc_vs_nacl_h)
print("T-test results for 'nacl_l' vs 'nacl_h':", t_test_nacl_l_vs_nacl_h)

# Visualize the final delivery rates using a box plot
sf = '/Users/emmabarash/lab/ISOT/pfinal_rateplot.svg'
plt.figure(figsize=(10, 6))
plt.boxplot([session_rates['suc'], session_rates['nacl_l'], session_rates['nacl_h']], labels=['suc', 'nacl_l', 'nacl_h'])
plt.xlabel('Taste')
plt.ylabel('Final Delivery Rate')
plt.title('Final Delivery Rates at the End of Each Session')
plt.grid(True)
plt.savefig(sf)
plt.show()