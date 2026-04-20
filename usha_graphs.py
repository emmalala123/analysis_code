#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:52:29 2025

@author: emmabarash
"""

#%%
# IMPORTS
# this might cause problems, 
# try "pip inistall [library_name]" in your console
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import glob
import random
import inflect
from scipy import stats
import re
import math
import easygui
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind
#%%

#directory = easygui.diropenbox(msg='Choose where the .csv files are...',
                                # add a default file path for it to look foor data
                               # default='[your directory here]')  


directory = '/Users/emmabarash/lab/data/three_tastes/usha_scifest_data26/'
filelist = glob.glob(os.path.join(directory,'**/*.csv'))

#boilerplate code to establish nosepoke entry, deliveries, etc

finaldf = pd.DataFrame(columns = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
       'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date', 'Taste_Delivery',
       'Delivery_Time', 'Latencies'])
filelist.sort()

for f in range(len(filelist)):
    df = pd.read_csv(filelist[f])
    group = df
    col = ['Line1', 'Line2']
    
    def parse_edges(group,col):
        delivery_idx = []
        group['TasteID'] = None
        if 'eb' in filelist[f]:
           group['AnID'] = filelist[f][-34:-30] 
        else:
            group['AnID'] = filelist[f][-33:-30]       
        group['Date'] = filelist[f][-29:-21] # for new data, -27 and -21 give date
        if 'higher' in filelist[f]:
           group['Category'] = 'higherNaCl' 
        else:
            group['Category'] = 'lowerNaCl'
            
        for j in col:
            col = j
            if col == 'Line1': 
                taste = 'suc'
            if col == 'Line2':
                taste = 'nacl_l'
            if col == 'Line3':
                taste = 'nacl_h'
            
            cols = ['Time']+[col]
            data = group[cols]
            try: edges = data[data[col].diff().fillna(False)]
            except: return None
            edgeON = edges[edges[col]==True]
            edgeON.col = True
            edgeON = edgeON.rename(columns={'Time':'TimeOn'})
            edgeON = edgeON.drop(col,axis=1)
            edgeON.index = np.arange(len(edgeON))
            edgeOFF = edges[edges[col]==False]
            edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
            edgeOFF = edgeOFF.drop(col,axis=1)
            edgeOFF.index = np.arange(len(edgeOFF))
            test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
            test['dt'] = test.TimeOff-test.TimeOn
            
            for i, row in test.iterrows():
                if len(np.where(df['Time'] == test['TimeOn'][i])[0]) == 1 and\
                    len(np.where(df['Time'] == test['TimeOff'][i])[0]) == 1:
                    # print(len(np.where(df['Time'] == test['TimeOn'][i])[0]))
                    start = int(np.where(df['Time'] == test['TimeOn'][i])[0])
                    stop = int(np.where(df['Time'] == test['TimeOff'][i])[0])
                    
                    group.loc[group.index[range(start,stop)],'TasteID'] = taste
                    delivery_idx.append(start)
            
        return group, delivery_idx
    
    group['Line1'] = group['Line1'].astype(bool)
    group['Line2'] = group['Line2'].astype(bool)
    group['Line3'] = group['Line3'].astype(bool)
    new_df, delivery_idx = parse_edges(df, ['Line1', 'Line2', 'Line3'])
    
    def create_edge_frame(copy):
        
        edge_df = copy
        
        isdelivery = copy.Line1+copy.Line2
        isdelivery_copy = isdelivery.shift(1)
        edge_df['Lines_Edge'] = isdelivery - isdelivery_copy
        edge_df['Lines_Edge'] = edge_df.Lines_Edge.fillna(0).astype(int)
        edge_df['Lines_Edge'] = edge_df.Lines_Edge.astype(int)
        # lines_pos_edges = np.where(sub_lines == 1)[0]
        # lines_neg_edges = np.where(sub_lines == -1)[0]
        
        ispoked = copy.Poke1
        ispoked_copy = ispoked.shift(1)
        edge_df['Pokes_Edge'] = ispoked - ispoked_copy
        edge_df['Pokes_Edge'] = edge_df.Pokes_Edge.fillna(0).astype(int)
        edge_df['Pokes_Edge'] = edge_df.Pokes_Edge.astype(int)
        # poke_pos_edges = np.where(subtract == 1)[0]
        # poke_neg_edges = np.where(subtract == -1)[0]
        
        ispoked = copy.Poke1
        ispoked_copy = ispoked.shift(1)
        edge_df['Pokes_Edge'] = ispoked - ispoked_copy
        edge_df['Pokes_Edge'] = edge_df.Pokes_Edge.fillna(0).astype(int)
        edge_df['Pokes_Edge'] = edge_df.Pokes_Edge.astype(int)
        # poke_pos_edges = np.where(subtract == 1)[0]
        # poke_neg_edges = np.where(subtract == -1)[0]
        
        # edge_df = edge_df.loc[(edge_df['Lines_Edge'] == 1) | (edge_df['Lines_Edge'] == -1)\
        #                     | (edge_df['Pokes_Edge'] == 1) | (edge_df['Pokes_Edge'] == 1)]
        
        return edge_df
        edge_df = create_edge_frame(df)
    
    
    
    def find_poke_dat(copy, poke, delivery_idx):
        # instantiate new columns with null values for later use
        copy['Taste_Delivery'] = False
        copy['Delivery_Time'] = None
        
        pokes = ['Time'] + [poke]
        data = copy[pokes]
        try: edges = data[data[poke].diff().fillna(False)]
        except: return None
        if not edges.empty:
            edgeON = edges[edges[poke]==True].shift(1)
           # print('edges', edges,'edgeON', edgeON, "copy time", copy['Time'][0])
            edgeON.iloc[0] = copy['Time'][0]
            edgeON[poke].iloc[0] = True
            edgeON.col = True
            edgeON = edgeON.rename(columns={'Time':'TimeOn'})
            edgeON = edgeON.drop(poke,axis=1)
            edgeON.index = np.arange(len(edgeON))
            
            edgeOFF = edges[edges[poke]==False]
            edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
            edgeOFF = edgeOFF.drop(poke,axis=1)
            edgeOFF.index = np.arange(len(edgeOFF))
            test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
            test['dt'] = test.TimeOff-test.TimeOn
        
        delivery_time = []
        for i in delivery_idx:
            copy.loc[i,'Taste_Delivery'] = True
            copy.loc[i,'Delivery_Time'] = copy['Time'][i]
            
            # collect delivery time to erase Poke2 dat within 10 seconds of delivery
            delivery_time.append(copy['Time'][i])
        
        # generatees a new df with only delivery times (marked 'true')
        deliveries_only = copy.loc[copy['Taste_Delivery'] == True].reset_index(drop=True)
        
        second_copy = copy
        for i in delivery_time:
            second_copy = second_copy.loc[~((second_copy['Time'] > i) & (second_copy['Time'] < i+5)),:]
        
        for i, row in second_copy.iterrows():
            poke1 = np.where(second_copy['Taste_Delivery'] == True)[0]
            poke2 = poke1-1
        lat1 = second_copy['Time'].iloc[poke2].reset_index(drop=True)
        lat2 = second_copy['Time'].iloc[poke1].reset_index(drop=True)
        
        latencies = lat2.subtract(lat1) #look up how to subtract series from each other
        
        deliveries_only['Latencies'] = latencies
        
        return deliveries_only
    
    deliveries_only = find_poke_dat(new_df,'Poke2', delivery_idx)
    finaldf = finaldf.append(deliveries_only)

def add_days_elapsed(finaldf):
    
    new_df = finaldf
    
    res = []
    for name, group in new_df.groupby('AnID'):
        i=1
        for n, g in group.groupby('Date'):
            print(g)
            bit = np.zeros(len(g))
            bit = bit + i
            res.extend(bit)
            i += 1
    new_df['Sessions'] = res
    
    return new_df

new_df = add_days_elapsed(finaldf)

def add_days_elapsed_again(finaldf):
    
    new_df = finaldf
    
    #res = []
    new_df['ones'] = 1
    
    tst= new_df[['AnID','Date','TasteID', 'Concentration','ones']].drop_duplicates()
    tst = pd.pivot(tst, index = ['AnID','Date'], columns = 'TasteID', values='Concentration')
    # tst['tasteset'] = tst['suc'] +'_&_'+ tst['qhcl']
    tst['tasteset'] = tst['suc']
    tst = tst.reset_index()
    
    tst['ones'] = 1
    tst['tastesession'] = tst.groupby(['tasteset'])['ones'].cumsum()
    
    new_df = new_df.merge(tst)    
    #new_df['tastesession'] = new_df.groupby(['AnID','TasteID','tasteset'])['ones'].cumsum()
    # for name, group in new_df.groupby('AnID','Concentration'):
    #     i=1
    #     for n, g in group.groupby(['Date', 'Concentration']):
    #         print(g)
    #         bit = np.zeros(len(g))
    #         bit = bit + i
    #         res.extend(bit)
    #         i += 1
    
    return new_df

# just check the low latenceis
less_than_one = new_df.loc[new_df['Latencies'] < 0.2]
less_than_one['ones'] = 1
num_less_than_one = less_than_one.groupby(['Date', 'AnID']).ones.sum()

def cumulativedels(new_df):
    csum = new_df.groupby(['AnID','Sessions','TasteID', 'Latencies']).Delivery_Time.sum()
    csum = csum.reset_index()
    return csum

csum = cumulativedels(new_df)
means = csum.groupby(["TasteID","Sessions"]).Delivery_Time.mean().reset_index()
fig, ax = plt.subplots(figsize=(10,5))
p1 = sns.scatterplot(data = csum, x = "Sessions", y = "Delivery_Time", hue = "TasteID", style = "AnID", s=65)
p2 = sns.lineplot(data = means, x = "Sessions", y = "Delivery_Time", hue = "TasteID")
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#%%
new_df.loc[(new_df['TasteID'] == 'nacl_l', 'Concentration')] = 'nacl_0.1M'
new_df.loc[new_df['TasteID'] == 'suc', 'Concentration'] = 'suc_0.3M'
new_df.loc[(new_df['TasteID'] == 'nacl_h') & (new_df['Category'] == 'higherNaCl'), 'Concentration'] = 'nacl_1.5M'
new_df.loc[(new_df['TasteID'] == 'nacl_h') & (new_df['Category'] == 'lowerNaCl'), 'Concentration'] = 'nacl_.75M'

# take out too low latencies
new_df = new_df.loc[new_df['Latencies'] > 0.2]


new_df3 = add_days_elapsed_again(new_df)

copy = new_df.reset_index(drop=True)

copy['group_cumsum'] = copy.groupby(['TasteID', 'AnID', 'Date'])['ones'].cumsum()
for name, group in copy.groupby(['Date']):
    sf = '/Users/emmabarash/lab/auto_save_days/' + name +'_cumplot.svg' 
    sns.relplot(data = group, x='Time', y='group_cumsum', kind = 'line', hue='TasteID', row='AnID', hue_order=['suc', 'nacl_l', 'nacl_h'])
    plt.savefig(sf)
    
################# subplots
# Calculate the global maximum value of 'group_cumsum'
y_max = copy['group_cumsum'].max()

# Grouping by 'AnID' and 'Sessions'
groups = list(copy.groupby(['AnID', 'Sessions']))

# Set the number of rows and columns for the subplots grid
n_rows = math.ceil(len(groups) / 2)  # Adjust columns as needed (e.g., 2 columns)
n_cols = 2

# Create the figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), tight_layout=True)

# Flatten axes for easy indexing if needed
axes = axes.flatten()

# Loop through each group and its corresponding subplot
for i, (anid, group) in enumerate(groups):
    ax = axes[i]
    sns.lineplot(
        data=group,
        x='Time',
        y='group_cumsum',
        hue='TasteID',
        hue_order=['suc', 'nacl_l', 'nacl_h'],
        ax=ax
    )
    ax.set_title(f"AnID: {anid}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Sum")
    ax.set_ylim(0, y_max)  # Set y-axis limit to the global maximum
    ax.legend(title='TasteID')

# Remove any unused subplots
for j in range(len(groups), len(axes)):
    fig.delaxes(axes[j])
plt.show()
#%% find palatability scores
data = copy
# Ensure 'Taste_Delivery' and 'group_cumsum' are numeric
data['Taste_Delivery'] = pd.to_numeric(data['Taste_Delivery'], errors='coerce')
data['group_cumsum'] = pd.to_numeric(data['group_cumsum'], errors='coerce')

# get cumulative values for each taste per session
all_cumsum_per_sesh = data.groupby(['AnID', 'Sessions', 'TasteID', 'Category']).group_cumsum.max().reset_index()

ax = sns.boxplot(data=all_cumsum_per_sesh, x = 'Sessions', y='group_cumsum', hue='Category')
ax1 = sns.swarmplot(data=all_cumsum_per_sesh, x = 'Sessions', y='group_cumsum', color='black')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax1.set_title('Tastes over Sessions')

# get preferences for each taste
ax = sns.boxplot(data=all_cumsum_per_sesh, x = 'TasteID', y='group_cumsum', hue='Category')
ax1 = sns.swarmplot(data=all_cumsum_per_sesh, x = 'TasteID', y='group_cumsum', hue='AnID')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax1.set_title('Cumulative Deliveries')

# get zscores for the tastes / session
all_cumsum_per_sesh['zscores'] = stats.zscore(all_cumsum_per_sesh['group_cumsum'])

#lineplot over sessions
ax = sns.lineplot(data = all_cumsum_per_sesh, x='Sessions', y='zscores', hue='TasteID')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_title('Zscore of Tastes / Session')

#boxplot for each taste
ax = sns.boxplot(data = all_cumsum_per_sesh, x='Category', y='zscores', hue='TasteID')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_title('Zscore of Tastes')

# Lineplot with error
sns.catplot(data=all_cumsum_per_sesh, x='TasteID', y='zscores', hue='Category', kind="point", 
            capsize=.15, aspect=1.5, errwidth=0.8)

#%%% One-way ANOVAs
hf_statistic, hp_value = stats.f_oneway(all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'suc') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_l') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_h') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True))
lf_statistic, lp_value = stats.f_oneway(all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'suc') & (all_cumsum_per_sesh['Category'] == 'lowerNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_l') & (all_cumsum_per_sesh['Category'] == 'lowerNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_h') & (all_cumsum_per_sesh['Category'] == 'lowerNaCl'), 'zscores'].reset_index(drop=True))

print('high', hf_statistic, hp_value)
print('low', lf_statistic, lp_value)

higher_t_test_suc_vs_nacl_l = ttest_ind(all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'suc') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_l') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'),'zscores'].reset_index(drop=True))
higher_t_test_suc_vs_nacl_h = ttest_ind(all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'suc') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_h') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'),'zscores'].reset_index(drop=True))
higher_t_test_nacl_l_vs_nacl_h = ttest_ind(all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_l') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'), 'zscores'].reset_index(drop=True), all_cumsum_per_sesh.loc[(all_cumsum_per_sesh['TasteID'] == 'nacl_h') & (all_cumsum_per_sesh['Category'] == 'higherNaCl'),'zscores'].reset_index(drop=True))


#%% Two-way ANOVA
model = ols('group_cumsum ~ C(TasteID) + C(Category) + C(TasteID):C(Category)', data=all_cumsum_per_sesh).fit()
sm.stats.anova_lm(model, typ=2)


# Find the last trial for each taste
def find_cumsum_all(df, taste):
    taste_data = df[df['TasteID'] == taste]
    cumsum_all = taste_data.groupby(['AnID', 'Date'])['group_cumsum'].max().reset_index()
    return cumsum_all

cumsum_all_suc = find_cumsum_all(data, 'suc')
cumsum_all_nacl_l = find_cumsum_all(data, 'nacl_l')
cumsum_all_nacl_h = find_cumsum_all(data, 'nacl_h')

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
#%%
# get rates and visualize palatable (two tastes combined) and unpalatable.
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
    
    # Calculate average delivery rates
    palatable_rates = interval_data[interval_data['Taste_Type'] == 'Palatable']['Corrected_Delivery_Rate']
    unpalatable_rates = interval_data[interval_data['Taste_Type'] == 'Unpalatable']['Corrected_Delivery_Rate']
    
    # Ensure numeric and drop NaNs
    palatable_rates = pd.to_numeric(palatable_rates, errors='coerce').dropna()
    unpalatable_rates = pd.to_numeric(unpalatable_rates, errors='coerce').dropna()

    # Calculate difference
    rate_diff = palatable_rates.mean() - unpalatable_rates.mean()
    
    # Mann-Whitney U test
    if len(palatable_rates) > 0 and len(unpalatable_rates) > 0:
        t_stat, p_value = mannwhitneyu(palatable_rates, unpalatable_rates, alternative='two-sided')
    else:
        t_stat, p_value = (None, None)
    
    # Store results
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

# Convert to numeric
anova_data['Corrected_Delivery_Rate'] = pd.to_numeric(anova_data['Corrected_Delivery_Rate'], errors='coerce')

# Bin the time into intervals
anova_data['Time_Bin'] = pd.cut(anova_data['Time'], bins=range(0, 3700, 100), labels=range(0, 3600, 100))

# Drop rows with missing values in either variable
anova_data = anova_data.dropna(subset=['Time_Bin', 'Corrected_Delivery_Rate'])

# Run the ANOVA model
model = ols('Corrected_Delivery_Rate ~ C(Time_Bin) * C(Taste_Type)', data=anova_data).fit()

# Create the interaction plot
plt.figure(figsize=(14, 7))

# Calculate mean delivery rates for each combination of time bin and taste type
interaction_data = anova_data.groupby(['Time_Bin', 'Taste_Type'])['Corrected_Delivery_Rate'].mean().reset_index()

# Plot interaction plot
sns.lineplot(data=interaction_data, x='Time_Bin', y='Corrected_Delivery_Rate', hue='Taste_Type', marker='o')

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