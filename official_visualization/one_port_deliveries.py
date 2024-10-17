#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:10:01 2024

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

if os.sep == '/':
    directory = '/Users/emmabarash/Lab/data/one_port/eb26'
else:
    directory = r'C:\Users\Emma_PC\Documents\data\paradigm_23'


filelist = glob.glob(os.path.join(directory,'*.csv'))
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
        group['AnID'] = filelist[f][-22:-18]
        group['Date'] = filelist[f][-17:-11]
        for j in col:
            col = j
            if col == 'Line1': 
                taste = 'suc'
            if col == 'Line2':
                taste = 'qhcl'
            
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
    new_df, delivery_idx = parse_edges(df, ['Line1', 'Line2'])
    
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
    tst['tasteset'] = tst['suc'] +'_&_'+ tst['qhcl']
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

# get the cumulative deliveries per session
# Convert the series to a DataFrame
df_deliveries = pd.DataFrame(num_less_than_one, columns=['ones']).reset_index()

# Rename the columns
df_deliveries.columns = ['Date', 'AnID', 'Deliveries']
df_deliveries = new_df.merge(df_deliveries)

# create the plot
sns.barplot(data=df_deliveries, x="Sessions", y="Deliveries", hue="AnID").set(title='Total deliveries across sessions')

