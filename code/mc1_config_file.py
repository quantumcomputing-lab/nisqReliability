#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:13:43 2023
@author: samudra
"""
# Configuration file
data_folder = '/home/samudra/Desktop/QuantumReliability/data/' # caution: data is not saved in git

# threshold_for_low_correlation = 0
num_mcmc_samples = 100_000
n_qiskit_samples=100
n_resets = 3
num_shots = 100#100_000

TIME_U1 = 0   # virtual gate
TIME_U2 = 50  # (single X90 pulse)
TIME_U3 = 100 # (two X90 pulses)
TIME_CX = 300
TIME_RESET = 1000  # 1 microsecond
TIME_MEASURE = 1000 # 1 microsecond

# #smax = 0.05
# #hmax = np.sqrt(1-np.sqrt(1-smax**2/4))

month_dict = {
0: 'Jan-22',
1: 'Feb-22',
2: 'Mar-22',
3: 'Apr-22',
4: 'May-22',
5: 'Jun-22',
6: 'Jul-22',
7: 'Aug-22',
8: 'Sep-22',
9: 'Oct-22',
10: 'Nov-22',
11: 'Dec-22',
12: 'Jan-23',
13: 'Feb-23',
14: 'Mar-23',
15: 'Apr-23',
}

noise_metrics_list_16 = [
    'q_0__readout_error_value',
    'q_1__readout_error_value',
    'q_2__readout_error_value',
    'q_3__readout_error_value',
    #'q_4__readout_error_value', # because we do not measure this
    'cx0_1_gate_error_value',
    'cx2_1_gate_error_value',
    'q_0__T2_value',
    'q_1__T2_value',
    'q_2__T2_value',
    'q_3__T2_value',
    'q_4__T2_value',
    'hadamard_0_gate_error_value',
    'hadamard_1_gate_error_value',
    'hadamard_2_gate_error_value',
    'hadamard_3_gate_error_value',
    'hadamard_4_gate_error_value',
  ]

header_dict = {
    0: 'Readout error (q0)',
    1: 'Readout error (q1)',
    2: 'Readout error (q2)',
    3: 'Readout error (q3)',
    #'Readout error (q4)',
    4: 'CNOT (0,1) error',
    5: 'CNOT (2,1) error',
    6: 'T2 time (q0)',
    7: 'T2 time (q1)',
    8: 'T2 time (q2)',
    9: 'T2 time (q3)',
    10: 'T2 time (q4)',
    11: 'Hadamard error (q0)',
    12: 'Hadamard error (q1)',
    13: 'Hadamard error (q2)',
    14: 'Hadamard error (q3)',
    15: 'Hadamard error (q4)',
  }