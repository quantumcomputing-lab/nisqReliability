#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:49:52 2023

@author: samudra
"""
import pandas as pd
import numpy as np

data_folder = '/home/samudra/Desktop/QuantumReliability/data/' # caution: data is not saved in git
device_data = data_folder + 'device_data.csv'

df0 = pd.read_csv(device_data)
df0 = df0.loc[:, ~df0.columns.str.contains('^Unnamed')]
df0['time_label_datetime'] = pd.to_datetime(df0['query_date'].str.strip('+'), format='%Y - %m - %d')

for my_year in (2022, 2023):
    for my_month in np.arange(1, 13):
        print(my_year, my_month)
        set1 = [item.year==my_year for item in df0['time_label_datetime'] ]
        set2 = [item.month==my_month for item in df0['time_label_datetime']]
        indices = np.logical_and(set1, set2)
        if np.sum(indices):
            filename = 'device_data_'+ str(my_month) + '_' + str(my_year) + '.csv'                   
            df0.loc[indices, :].reset_index(drop=True).to_csv(filename)
