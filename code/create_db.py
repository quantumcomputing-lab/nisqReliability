#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:40:11 2022
@author: samudra
Purpose: creates historical database for any online IBM device
"""

device_list = {
'ibm_washington': 127,
}

#import qiskit
from qiskit import IBMQ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


import warnings
warnings.filterwarnings("ignore")

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc406')
stop_dt = datetime(year=2023, month = 4, day = 30) # e.g. today
start_dt = datetime(year = 2023, month = 1, day = 1) # e.g. your last database date OR 
# if you know when the dveice came online. Otherwise, enter (1,1,1)

def make_row_entry(tiny_dict, device_data, row_prefix, row_counter):
    date = tiny_dict['date']
    name = tiny_dict['name']
    unit = tiny_dict['unit']
    value = tiny_dict['value']

    cols_check = [row_prefix + '_' + name + '_unit', 
                  row_prefix + '_' + name + '_cal_date', 
                  row_prefix + '_' + name + '_value']

    for col in cols_check:
        if col not in device_data:
            device_data[col]=np.nan

    device_data.loc[row_counter,row_prefix+'_'+name+'_unit'] = unit
    device_data.loc[row_counter,row_prefix+'_'+name+'_cal_date'] = date
    device_data.loc[row_counter,row_prefix+'_'+name+'_value'] = value

# INPUTS
for DEVICE in device_list.keys():
    #DEVICE = 'ibmq_brooklyn' # Enter the device name here
    filename = DEVICE + '_temp.csv'
    
    backend = provider.get_backend(DEVICE)
    columns = ['query_date', 'last_update_date']
    device_data = pd.DataFrame(columns = columns)
       
    row_counter=0
    dt = stop_dt
    while (dt >= start_dt):
    #while (1):
        #dt = datetime(year = 2022, month = 3, day = 15)        
        for try_counter in range(100):
            dt0 = dt -  timedelta(days=try_counter)
            try:
                backend.properties(datetime=dt0)
                print(DEVICE + dt0.strftime(': OK: %Y - %m - %d - %H'))
                dt = dt0
                break
            except Exception:
                print(DEVICE + dt0.strftime(': ERROR: %Y - %m - %d - %H'))
                continue
    
        print(DEVICE + dt.strftime(': %Y - %m - %d - %H'))
        prop = backend.properties(datetime=dt)

        # this will work for all "online" devices but not tested for offline ones
        if prop is None:
            break
    
        prop_dict=prop.to_dict()
    
        device_data.loc[row_counter,'query_date'] = dt.strftime('%Y - %m - %d')
        device_data.loc[row_counter,'last_update_date'] = prop_dict['last_update_date']
    
        ### gate data
        num_of_gates_entries = len(prop_dict['gates'])
        for counter1 in range(num_of_gates_entries):
            row_prefix = prop_dict['gates'][counter1]['name']
            for counter2 in range(len(prop_dict['gates'][counter1]['parameters'])):
                tiny_dict = prop_dict['gates'][counter1]['parameters'][counter2]
                make_row_entry(tiny_dict, device_data, row_prefix, row_counter)
    
        ### qubit data
        num_of_qubit_entries = len(prop_dict['qubits'])
        for counter1 in range(num_of_qubit_entries):
            qubit_data_dict = prop_dict['qubits'][counter1]
            for counter2 in range(len(qubit_data_dict)):
                tiny_dict = qubit_data_dict[counter2]
                row_prefix = 'q_'+str(counter1)+'_'
                make_row_entry(tiny_dict, device_data, row_prefix, row_counter)
        
        row_counter=row_counter+1
        dt = dt -  timedelta(days=1)
    
    reversed_data = device_data.iloc[::-1].reset_index(drop=True)
    reversed_data.to_csv(filename)