#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:07:03 2024

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
# filelist = glob.glob(os.path.join(directory,'*.csv'))

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

# copy['group_cumsum'] = copy.groupby(['TasteID', 'AnID', 'Date'])['ones'].cumsum()
# for name, group in copy.groupby(['Date']):
#     sf = '/Users/emmabarash/lab/auto_save_days/' + name +'_cumplot.svg' 
#     sns.relplot(data = group, x='Time', y='group_cumsum', kind = 'line', hue='TasteID', row='AnID', hue_order=['suc', 'nacl_l', 'nacl_h'])

df = copy

# Define the rolling window size (e.g., 5 minutes)
window_size = 500

# Filter rows where a delivery occurs
df_deliveries = df[df['Taste_Delivery'] == 1]

# Calculate the delivery rates using Delivery_Time
df_deliveries['Delivery_Rate'] = df_deliveries.groupby('TasteID')['Delivery_Time'].transform(lambda x: x.rolling(window=window_size).count() / window_size)

# Define which TasteID corresponds to palatable and unpalatable
palatable_tastes = ['suc', 'nacl_l']  # Add the TasteID values that correspond to palatable tastes
unpalatable_taste = 'nacl_h'  # Add the TasteID value that corresponds to the unpalatable taste

# Separate data for palatable and unpalatable tastes
df_palatable = df_deliveries[df_deliveries['TasteID'].isin(palatable_tastes)]
df_unpalatable = df_deliveries[df_deliveries['TasteID'] == unpalatable_taste]

# Compute the rolling average delivery rates
rolling_avg_palatable = df_palatable.groupby('Delivery_Time')['Delivery_Rate'].mean().rolling(window=window_size).mean()
rolling_avg_unpalatable = df_unpalatable.groupby('Delivery_Time')['Delivery_Rate'].mean().rolling(window=window_size).mean()

# plot with no ci
plt.figure(figsize=(10, 6))
plt.plot(rolling_avg_palatable, label='Palatable Tastes', color='green')
plt.plot(rolling_avg_unpalatable, label='Unpalatable Taste', color='red')
plt.xlabel('Delivery Time')
plt.ylabel('Rolling Average Delivery Rate')
plt.title('Rolling Average Delivery Rate of Tastes Over Time')
plt.legend()
plt.show()

df_combined = pd.concat([df_palatable, df_unpalatable])

#slice_df = df_deliveries.groupby(['Delivery_Time', 'Taste_Type'])['Delivery_Rate'].mean().reset_index()


# Plotting the results with confidence intervals
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_combined, x='Delivery_Time', y='Delivery_Rate', hue='TasteID', ci='sd')
plt.xlabel('Delivery Time')
plt.ylabel('Rolling Average Delivery Rate')
plt.title('Rolling Average Delivery Rate of Tastes Over Time with Confidence Intervals')
plt.legend(title='Taste Type')
plt.show()

#ANOVA
# Label the tastes as palatable or unpalatable
df_deliveries['Taste_Type'] = df_deliveries['TasteID'].apply(lambda x: 'Palatable' if x in palatable_tastes else 'Unpalatable' if x == unpalatable_taste else 'Other')

# Group by Delivery_Time and Taste_Type to get mean delivery rates
grouped = df_deliveries.groupby(['Delivery_Time', 'Taste_Type'])['Delivery_Rate'].mean().reset_index()

# Pivot the table to have separate columns for palatable and unpalatable
pivot_table = grouped.pivot(index='Delivery_Time', columns='Taste_Type', values='Delivery_Rate').reset_index()
pivot_table = pivot_table.dropna(subset=['Palatable', 'Unpalatable'])

# Rename columns for clarity
pivot_table.columns = ['Delivery_Time', 'Palatable_Rate', 'Unpalatable_Rate']

# Define function to perform ANOVA at each time point
def perform_anova(data):
    time_points = data['Delivery_Time'].unique()
    anova_results = []

    for time in time_points:
        subset = data[data['Delivery_Time'] == time]
        palatable_rates = subset['Palatable_Rate']
        unpalatable_rates = subset['Unpalatable_Rate']
        f_stat, p_value = f_oneway(palatable_rates, unpalatable_rates)
        anova_results.append({'Delivery_Time': time, 'F-Statistic': f_stat, 'P-Value': p_value})
    
    return pd.DataFrame(anova_results)

anova_results = perform_anova(pivot_table)

# Plotting the p-values over time
plt.figure(figsize=(12, 8))
plt.plot(anova_results['Delivery_Time'], anova_results['P-Value'], label='P-Value', color='blue')
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.xlabel('Delivery Time')
plt.ylabel('P-Value')
plt.title('ANOVA P-Values Over Time')
plt.legend()
plt.show()
