#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:47:55 2024

@author: emmabarash
"""
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
from scipy.stats import f_oneway
from statsmodels.formula.api import ols

if os.sep == '/':
    directory = '/Users/emmabarash/Lab/data/three_tastes'
else:
    directory = r'C:\Users\Emma_PC\Documents\data\paradigm_23'

# directory = '/Users/emmabarash/Lab/blacklist'

filelist = glob.glob(os.path.join(directory,'**','*.csv'))
#filelist = glob.glob(os.path.join(directory,'*.csv'))

finaldf = pd.DataFrame(columns = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
       'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date', 'Taste_Delivery',
       'Delivery_Time', 'Latencies'])
filelist.sort()

#filelist = filelist[-3:]

for f in range(len(filelist)):
    df = pd.read_csv(filelist[f])
    group = df
    col = ['Line1', 'Line2', 'Line3']
    
    def parse_edges(group,col):
        delivery_idx = []
        group['TasteID'] = None
        group['AnID'] = filelist[f][-34:-30]
        group['Date'] = filelist[f][-29:-21] # for new data, -27 and -21 give date
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

new_df.loc[(new_df['TasteID'] == 'nacl_l', 'Concentration')] = 'nacl_0.1M'
new_df.loc[new_df['TasteID'] == 'suc', 'Concentration'] = 'suc_0.3M'
new_df.loc[new_df['TasteID'] == 'nacl_h', 'Concentration'] = 'nacl_1.5M'

# take out too low latencies
new_df = new_df.loc[new_df['Latencies'] > 0.2]


new_df3 = add_days_elapsed_again(new_df)

copy = new_df

copy['group_cumsum'] = copy.groupby(['TasteID', 'AnID', 'Date'])['ones'].cumsum()
for name, group in copy.groupby('Date'):
#     sf = '/Users/emmabarash/lab/auto_save_days/' + name +'_cumplot.svg' 
    sns.relplot(data = group, x='Time', y='group_cumsum', kind = 'line', hue='TasteID', row='AnID', hue_order=['suc', 'nacl_l', 'nacl_h'])

trials_df = copy

# Create a column for trial numbers within each group
trials_df['Trial_Num'] = trials_df.groupby(['Sessions', 'AnID']).cumcount() + 1

trials_df = trials_df.reset_index()
    
sns.lineplot(data=trials_df, x='Trial_Num', y='group_cumsum', hue='TasteID')

# TODO - take the average for all animals. Currently all raw data

# Calculate mean and SEM of group_cumsum for each TasteID and Trial_Num
grouped = trials_df.groupby(['TasteID', 'Trial_Num']).agg(
    mean_cumsum=('group_cumsum', 'mean'),
    sem_cumsum=('group_cumsum', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
).reset_index()

# Plotting
plt.figure(figsize=(10, 6))
for taste in grouped['TasteID'].unique():
    taste_data = grouped[grouped['TasteID'] == taste]
    plt.errorbar(
        taste_data['Trial_Num'], taste_data['mean_cumsum'], yerr=taste_data['sem_cumsum'],
        label=f'Taste {taste}', capsize=5, marker='o', linestyle='-'
    )

plt.xlabel('Trial Number')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Number of Trials')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()

# try smoothing data by binning trials
# Define the bin size
bin_size = 3

# Create a new column for trial bins
trials_df['Trial_Bin'] = (trials_df['Trial_Num'] - 1) // bin_size + 1

# Calculate mean and SEM of group_cumsum for each TasteID and Trial_Bin
grouped_smooth = trials_df.groupby(['TasteID', 'Trial_Bin']).agg(
    mean_cumsum=('group_cumsum', 'mean'),
    sem_cumsum=('group_cumsum', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
).reset_index()

# Plotting
plt.figure(figsize=(10, 6))
for taste in grouped_smooth['TasteID'].unique():
    taste_data = grouped_smooth[grouped_smooth['TasteID'] == taste]
    plt.errorbar(
        taste_data['Trial_Bin'] * bin_size, taste_data['mean_cumsum'], yerr=taste_data['sem_cumsum'],
        label=f'Taste {taste}', capsize=5, marker='o', linestyle='-'
    )

plt.xlabel('Trial Number')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Number of Trials')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()

# Create a new DataFrame for plotting with seaborn
# plot_data = pd.melt(grouped_smooth, id_vars=['TasteID', 'Trial_Bin'], value_vars=['mean_cumsum', 'sem_cumsum'])

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped_smooth, x='Trial_Bin', y='mean_cumsum', hue='TasteID')

plt.xlabel('Trial Number')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Number of Trials')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()

##################
# Create a new column for trial bins
bin_size = 5
trials_df['Trial_Bin'] = (trials_df['Trial_Num'] - 1) // bin_size + 1

# Calculate mean of group_cumsum for each TasteID and Trial_Bin
grouped = trials_df.groupby(['TasteID', 'Trial_Bin']).agg(
    mean_cumsum=('group_cumsum', 'mean')
).reset_index()

# Define a function to perform rolling window averaging
def rolling_window_avg(df, window, iterations):
    for _ in range(iterations):
        df['mean_cumsum'] = df.groupby('TasteID')['mean_cumsum'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    return df

# Apply the rolling window average function
iterations = 0  # Number of iterations for smoothing
window = 2  # Window size for rolling average
grouped = rolling_window_avg(grouped, window, iterations)

# Recalculate SEM for the smoothed data
grouped['sem_cumsum'] = grouped.groupby('TasteID')['mean_cumsum'].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=1) / np.sqrt(window))

# Plotting with error bars
plt.figure(figsize=(10, 6))
for taste in grouped['TasteID'].unique():
    taste_data = grouped[grouped['TasteID'] == taste]
    plt.errorbar(
        taste_data['Trial_Bin'], taste_data['mean_cumsum'], yerr=taste_data['sem_cumsum'],
        label=f'Taste {taste}', capsize=5, marker='o', linestyle='-'
    )

plt.xlabel('Trial Number')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Number of Trials (Smoothed)')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()

#############  #using time and time bins
# Create a new column for time bins
time_bin_size = 100  # 100-second bins
trials_df['Time_Bin'] = (trials_df['Time'] // time_bin_size) * time_bin_size

# Calculate mean and SEM of group_cumsum for each TasteID and Time_Bin
grouped = trials_df.groupby(['TasteID', 'Time_Bin']).agg(
    mean_cumsum=('group_cumsum', 'mean'),
    sem_cumsum=('group_cumsum', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
).reset_index()

# Apply a single rolling window average function
window = 2  # Window size for rolling average
grouped['mean_cumsum'] = grouped.groupby('TasteID')['mean_cumsum'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

# Recalculate SEM for the smoothed data
grouped['sem_cumsum'] = grouped.groupby('TasteID')['mean_cumsum'].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=1) / np.sqrt(window))

# Plotting with error bars
plt.figure(figsize=(10, 6))
for taste in grouped['TasteID'].unique():
    taste_data = grouped[grouped['TasteID'] == taste]
    plt.errorbar(
        taste_data['Time_Bin'], taste_data['mean_cumsum'], yerr=taste_data['sem_cumsum'],
        label=f'Taste {taste}', capsize=5, marker='o', linestyle='-'
    )

plt.xlabel('Time (s)')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Time (Smoothed)')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()

#############

# Group by Sessions, AnID, TasteID, and Trial_Num to calculate the mean of cumulative deliveries
mean_cumulative_deliveries = trials_df.groupby(['Sessions', 'AnID', 'TasteID', 'Trial_Num'])['group_cumsum'].mean().reset_index()

# Pivot the DataFrame to have columns for each taste
pivot_df = mean_cumulative_deliveries.pivot_table(index=['Sessions', 'AnID', 'Trial_Num'], columns='TasteID', values='group_cumsum').reset_index()

# Rename columns for clarity
pivot_df.columns.name = None
pivot_df.rename(columns=lambda x: 'Average_' + str(x) if x.startswith('T') else x, inplace=True)

print(pivot_df)

# Ensure the DataFrame is sorted by Sessions and AnID
trials_df.sort_values(by=['Trial_Num', 'AnID', 'TasteID'], inplace=True)

# Group by Sessions, AnID, TasteID, and Trial_Num to calculate the mean of cumulative deliveries
mean_cumulative_deliveries = trials_df.groupby(['Sessions', 'AnID', 'TasteID', 'Trial_Num']).agg(
    mean_delivery=('group_cumsum', 'mean'),
    std_delivery=('group_cumsum', 'std')
).reset_index()

# Calculate the standard error
mean_cumulative_deliveries['sem_delivery'] = mean_cumulative_deliveries['std_delivery'] / np.sqrt(trials_df.groupby(['Sessions', 'AnID', 'TasteID', 'Trial_Num']).size().values)

# Plotting with error bars
plt.figure(figsize=(10, 6))
for taste in grouped['TasteID'].unique():
    taste_data = grouped[grouped['TasteID'] == taste]
    plt.errorbar(
        taste_data['Time_Bin'], taste_data['mean_cumsum'], yerr=taste_data['sem_cumsum'],
        label=f'Taste {taste}', capsize=5, marker='o', linestyle='-'
    )

plt.xlabel('Time (s)')
plt.ylabel('Average Cumulative Deliveries')
plt.title('Average Cumulative Deliveries for Each Taste Over Time (Smoothed)')
plt.legend(title='TasteID')
plt.grid(True)
plt.show()
### ANOVA ###
