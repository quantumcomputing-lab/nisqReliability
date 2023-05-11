#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:59:08 2022

@author: samudra
"""

import matplotlib.pyplot as plt
import calendar
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
from datetime import datetime

filename = '/home/samudra/Desktop/QuantumReliability/data/device_data.csv'
df = pd.read_csv(filename)

dates = df['query_date'].to_numpy().ravel()

df['time_label_datetime'] = pd.to_datetime(df['query_date'].str.strip('+'), format='%Y - %m - %d')
df = df.sort_values(by='time_label_datetime').reset_index(drop=True)


low = pd.to_datetime('2022 - 1 - 01', format='%Y - %m - %d')
high = pd.to_datetime('2023 - 4 - 30', format='%Y - %m - %d')
rows_df = np.logical_and(df['time_label_datetime']>=low, df['time_label_datetime']<=high)
df = df.loc[rows_df, :]

# ts plots
plt.clf()
for i in (3,5):
    qstr=str(i)
    FI_all = 1-df.loc[:, 'q_'+qstr+'__readout_error_value'].to_numpy().ravel()
    plt.plot(FI_all)
    sns.despine(left=True, bottom=True)
    #plt.legend(loc="best")
    plt.grid(True, which='both')    
    plt.ylabel('Value of SPAM Fidelity\n for qubits 3 and 5')
    plt.xlabel('Day of year\n (2022-23)')
    
    ticks = np.arange(0, len(df))
    labels = [item.timetuple().tm_yday for item in df['time_label_datetime']]
    plt.gca().set_xticks(ticks[::40])
    plt.gca().set_xticklabels(labels[::40])
    
    #plt.axhline(y=np.percentile(FI_all, 97.5), color='darkred', linestyle='dashed')
    #plt.axhline(y=np.percentile(FI_all, 2.5), color='darkred', linestyle='dashed')

plt.savefig('q_3_5_washington_jan2022_apr2023.png', bbox_inches = "tight", dpi=300)
