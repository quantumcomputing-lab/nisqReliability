#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:03:16 2022

@author: samudra
"""

#import matplotlib.pyplot as plt
#import calendar
import pandas as pd
import numpy as np
#import scipy.stats
#import seaborn as sns

data_folder = '/home/samudra/Desktop/QuantumReliability/data/' # caution: data is not saved in git
device_data = data_folder + 'device_data.csv'
#May31_2022 = '/home/samudra/Desktop/mc1_device_reliability/data/ibm_washington_May31_2022.csv'
#Oct8_2022 = '/home/samudra/Desktop/mc1_device_reliability/data/ibm_washington_Oct8_2022.csv'
apr30_2023 = data_folder + 'ibm_washington_apr30_2023.csv'


df0 = pd.read_csv(device_data)
df0 = df0.loc[:, ~df0.columns.str.contains('^Unnamed')]

#df1 = pd.read_csv(May31_2022)
#df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]

df2 = pd.read_csv(apr30_2023)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]

# sort each df by datetime
df0['time_label_datetime'] = pd.to_datetime(df0['query_date'].str.strip('+'), format='%Y - %m - %d')
df0 = df0.sort_values(by='time_label_datetime').reset_index(drop=True)
    
#df1['time_label_datetime'] = pd.to_datetime(df1['query_date'].str.strip('+'), format='%Y - %m - %d')
#df1 = df1.sort_values(by='time_label_datetime').reset_index(drop=True)

df2['time_label_datetime'] = pd.to_datetime(df2['query_date'].str.strip('+'), format='%Y - %m - %d')
df2 = df2.sort_values(by='time_label_datetime').reset_index(drop=True)

# get the boundaries
print( df0.iloc[0,:]['query_date'])
print( df0.iloc[-1,:]['query_date'])

#print( df1.iloc[0,:]['query_date'])
#print( df1.iloc[-1,:]['query_date'])

print( df2.iloc[0,:]['query_date'])
print( df2.iloc[-1,:]['query_date'])

# decide the overlaps (keep the new one)
# from df0 delete 
df0=df0.loc[df0['query_date'] != '2022 - 10 - 07',].reset_index(drop=True)
#df0=df0.loc[df0['query_date'] != '2022 - 03 - 09',].reset_index(drop=True)

# from df1 delete 05-25
#df1=df1.loc[df1['query_date'] != '2022 - 05 - 25',].reset_index(drop=True)

# delete the overlaps (keep the new one)
#df0=df0.append(df1).reset_index(drop=True)
df0=df0.append(df2).reset_index(drop=True)
df0=df0.reset_index(drop=True)
df0.to_csv('temp.csv')
