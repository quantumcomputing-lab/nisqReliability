#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:56:17 2023

@author: samudra
"""
from qiskit import *
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import ReadoutError
from qiskit.providers.aer.noise import phase_amplitude_damping_error, phase_damping_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import coherent_unitary_error

import qiskit.quantum_info as qi

from qiskit.visualization import array_to_latex
from qiskit.providers.ibmq.job import job_monitor


#IBMQ.load_account()
simulator = Aer.get_backend('qasm_simulator')
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc406')
# backend = provider.get_backend('ibmq_belem')
# #backend = provider.get_backend('ibm_washington')
# backend_config = backend.configuration()
# if backend_config.multi_meas_enabled == True:
#     print('ok')

# n_resets = 3
# dt = backend_config.dt
# print(f"Sampling time: {dt*1e9} ns")
# delay_in_dt = int(delay_in_sec/dt)

# backend_config = backend.configuration()
# if backend_config.multi_meas_enabled == True:
#     print('ok')
#nqubits = 27
#QHUB = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc406')
#device = QHUB.get_backend('ibmq_toronto')
#backend = provider.get_backend('ibm_washington')
#noise_model=NoiseModel.from_backend(device)
# dt = backend_config.dt
# print(f"Sampling time: {dt*1e9} ns")
# delay_in_dt = int(delay_in_sec/dt)

#### ROUTINES ####
def get_time_circuit():
    t_circuit = 2*TIME_U3 + TIME_CX + TIME_MEASURE #careful
    return t_circuit

def my_noise_model(hadamard_e_vec, cx_e_vec, readout_e_vec, 
                                            T1_vec, T2_vec):
    noise_model = NoiseModel()

    k=0
    for e in hadamard_e_vec:
        err_h = QuantumCircuit(1, 1)
        err_h.p(e, 0) # we are adding a phase gate
        u_err_h = qi.Operator(err_h)
        error = coherent_unitary_error(u_err_h)       
        noise_model.add_quantum_error(error, ['h'], [k])

        param_amp = 1-np.exp(-TIME_U3/T1_vec[k])
        param_phase = 1-np.exp(-TIME_U3/T2_vec[k])
        errors_phase_amplitude =  phase_amplitude_damping_error(param_amp, param_phase)
        noise_model.add_quantum_error(errors_phase_amplitude, ['h'], [k])
        k=k+1

    k=0
    for e in readout_e_vec:
        # Measurement miss-assignement probabilities
        p0given1 = e
        p1given0 = e
        error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        noise_model.add_readout_error(error, [k])
        k=k+1

    for k,v in cx_e_vec.items():
        noise_model.add_quantum_error(depolarizing_error(v[2]*4/3, 2), 'cx', [v[0],v[1]])

        param_amp1 = 1-np.exp(-TIME_CX/T1_vec[ int(v[0]) ])
        param_phase1 = 1-np.exp(-TIME_CX/T2_vec[int(v[0])])
        errors_phase_amplitude1 =  phase_amplitude_damping_error(param_amp1, param_phase1)

        param_amp2 = 1-np.exp(-TIME_CX/T1_vec[ int(v[1]) ])
        param_phase2 = 1-np.exp(-TIME_CX/T2_vec[int(v[1])])
        errors_phase_amplitude2 =  phase_amplitude_damping_error(param_amp2, param_phase2)

        error2 = errors_phase_amplitude1.tensor(errors_phase_amplitude2)
        

    # readout error will have the effect: avoid double counting
    # for k in range(len(T1_vec)):
    #     param_amp = 1-np.exp(-t_circuit/T1_vec[k])
    #     param_phase = 1-np.exp(-t_circuit/T2_vec[k])
    #     errors_phase_amplitude =  phase_amplitude_damping_error(param_amp, param_phase)
    #     noise_model.add_quantum_error(errors_phase_amplitude, ['measure'], [k])
        
    #print(noise_model)    
    return noise_model

def my_hadamard_cx_readout_T1T2_noise_model(hadamard_e_vec, cx_e_vec, 
                                            readout_e_vec, 
                                            T1_vec, T2_vec):
    noise_model = NoiseModel()

    k=0
    for e in hadamard_e_vec:
        err_h = QuantumCircuit(1, 1)
        err_h.p(e, 0) # we are adding a phase gate
        u_err_h = qi.Operator(err_h)
        error = coherent_unitary_error(u_err_h)       
        noise_model.add_quantum_error(error, ['h'], [k])
        k=k+1

    k=0
    for e in readout_e_vec:
        # Measurement miss-assignement probabilities
        p0given1 = e
        p1given0 = e
        error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        noise_model.add_readout_error(error, [k])
        k=k+1

    for k,v in cx_e_vec.items():
        noise_model.add_quantum_error(depolarizing_error(v[2]*4/3, 2), 'cx', [v[0],v[1]])

    for k in range(len(T1_vec)):
        param_amp = 1-np.exp(-t_circuit/T1_vec[k])
        param_phase = 1-np.exp(-t_circuit/T2_vec[k])
        error =  phase_amplitude_damping_error(param_amp, param_phase)
        #error = phase_damping_error(param_phase)
        # print(phase_damping_error )
        noise_model.add_quantum_error(error , ['id'], [k])
        
    #print(noise_model)    
    return noise_model

def make_hist_and_probability_of_success(e=0.4, L=1000):

    pr_success = np.zeros(L)*np.nan
    for L_iter in range(L):
        print(L_iter)
        resetted_results, pr_success[L_iter] = get_hist_and_pr_success(e)

    keys = get_labels()
    v = np.zeros(2**4)
    for i in np.arange(2**4):
       result_key = keys[i][1:-1]
       if result_key in resetted_results.keys():
           v[i] = resetted_results[keys[i][1:-1]]

    plt.bar(np.arange(2**4), v)
    plt.savefig('test0.png')

    plt.plot(np.arange(L), pr_success, linestyle='', marker='*')
    #plt.ylim(.97, 1.0)
    plt.savefig('test1.png')

    return 0

def get_labels():
    return ['{0000}',
    '{0001}',
    '{0010}',
    '{0011}',
    '{0100}',
    '{0101}',
    '{0110}',
    '{0111}',
    '{1000}',
    '{1001}',
    '{1010}',
    '{1011}',
    '{1100}',
    '{1101}',
    '{1110}',
    '{1111}']

# oracle solution
def get_word(n, soln=3):
    word=[]
    if soln==3:
        word = '0'*(n-2)+'1'*2
    return word

# BV circuit
def my_BV(n = 4, s = '0011', num_circuits=1):
    # n = number of qubits used to represent s
    # s = the hidden binary string
    if len(s)!=n:
        print('error')

    # We need a circuit with n qubits, plus one auxiliary qubit
    # Also need n classical bits to write the output to
    #bv_circuit = QuantumCircuit(n+1, n)
    # n = 4
    # s = '0011'
    
    bv_circuit = QuantumCircuit(n+1, n*num_circuits)
    for _ in np.arange(num_circuits): # 1 milli second variability
        # put auxiliary in state |->
        bv_circuit.h(n)
        bv_circuit.z(n)
        
        # Apply Hadamard gates before querying the oracle
        for i in range(n):
            bv_circuit.h(i)
            
        # Apply barrier 
        bv_circuit.barrier()
        
        # Apply the inner-product oracle
        s_rev = s[::-1] # reverse s to fit qiskit's qubit ordering
        for q in range(n):
            if s_rev[q] == '0':
                pass #bv_circuit.i(q)
            else:
                bv_circuit.cx(q, n)
                
        # Apply barrier 
        bv_circuit.barrier()
        
        #Apply Hadamard gates after querying the oracle
        for i in range(n):
            bv_circuit.h(i)
        
        # Identity layer (will come in handy for noise)
        for i in range(n):
            bv_circuit.id(i)

        # Measurement
        for i in range(n):
            bv_circuit.measure(i, i)

        # bv_circuit.barrier()

        # # all reset post measurement    
        # for temp_ in range(n+1):
        #     bv_circuit.reset([temp_]*n_resets)
    
        # bv_circuit.barrier()
    
    #bv_circuit.draw(output='mpl', filename='bv_qiskit_ckt.png')

    return bv_circuit

def get_param_vec_for_qiskit(data_set, param_name):
    if param_name == 'H':
        ans = np.zeros(5)
        for i in range(5):
            ans[i] = np.nanmedian( data_set['hadamard_'+str(i)+'_gate_error_value'] )  

    if param_name == 'readout':
        ans = np.zeros(5)
        for i in range(5):
            colname = 'q_'+str(i)+'__readout_error_value'
            if colname in data_set.keys():
                ans[i] = np.nanmedian( data_set[colname] )  

    if param_name == 'cx':
        ans = {}
        ans['2,4'] = [2,4, np.nanmedian( data_set['cx0_1_gate_error_value'] )]
        ans['3,4'] = [3,4, np.nanmedian( data_set['cx2_1_gate_error_value'] )]

    if param_name == 'T2':
        ans = np.ones(5)*np.inf
        for i in range(5):
            ans[i] = np.nanmedian( data_set['q_'+str(i)+'__T2_value'] )  

    if param_name == 'T1':
        ans = np.ones(5)*np.inf
        for i in range(5):
            colname = 'q_'+str(i)+'__T1_value'
            if colname in data_set.keys():
                ans[i] = np.nanmedian( data_set[colname] )  

    return ans

def get_pr_success_monthly(samples_df):
    #samples_df = mcmc_samples[1]
    
    samples_df  = samples_df.reset_index(drop=True)  # make sure indexes pair with number of rows

    s = np.zeros(len(samples_df))*np.nan
    k = -1
    for _, row in samples_df.iterrows():
        k=k+1
        print(k)
        one_error_sample_df = pd.DataFrame(columns = samples_df.columns)
        for name in one_error_sample_df.columns:
            one_error_sample_df.loc[0, name] = row[name] 
        #break
        hadamard_e_vec, cx_e_vec,  readout_e_vec, T1_vec, T2_vec = initialize_error_set(one_error_sample_df)
        num_qubits=4
        n_resets=3
        secret_word = '0'*(num_qubits-2)+'11'
        bv_circuit = my_BV(n = num_qubits, s = secret_word, num_circuits=1)
        num_shots=10_000
        t_circuit = get_time_circuit()
        simulator_results = execute(bv_circuit,\
                   #backend  = device,\
                   backend  = simulator,\
                  #initial_layout = layout,\
                  noise_model    = my_hadamard_cx_readout_T1T2_noise_model(hadamard_e_vec, 
                                                                           cx_e_vec, readout_e_vec, 
                                                                           T1_vec, 
                                                                           T2_vec),\
                  #coupling_map   = coupling_map,\
                  #basis_gates    = test_config['basis_gates'],\
                  #seed_simulator = seed,
                  optimization_level = 0,\
                  memory = False,\
                  shots          = num_shots).result()
            
        simulator_results = simulator_results.get_counts()    
        pr_success = 0
        if secret_word in simulator_results.keys():
            pr_success = simulator_results[secret_word]/num_shots
        #print("Obtained = ", pr_success )
        s[k] = pr_success
    
    return s

def initialize_error_set(one_error_sample_df=pd.DataFrame()):
    
    hadamard_e_vec = np.zeros(5)
    hadamard_e_vec[0] = 0
    hadamard_e_vec[1] = 0
    hadamard_e_vec[2] = 0
    hadamard_e_vec[3] = 0
    hadamard_e_vec[4] = 0
    
    cx_e_vec = {'2,4' : [2, 4, 0.0], '3,4' : [3, 4, 0.0],}
    
    readout_e_vec = np.zeros(5)
    readout_e_vec[0] = 0
    readout_e_vec[1] = 0
    readout_e_vec[2] = 0
    readout_e_vec[3] = 0
    readout_e_vec[4] = 0
    
    T1_vec = np.ones(5)*np.inf    
    T2_vec = np.ones(5)*np.inf
    
    if not one_error_sample_df.empty:
        for colname in one_error_sample_df.columns:
            #print(colname)
            val = float( one_error_sample_df[colname] )
    
            for i in range(5):
                str_i = str(i)
                if colname == 'hadamard_'+str_i+'_gate_error_value':
                    hadamard_e_vec[i] = np.pi/8 #val
        
                if colname == 'q_'+str_i+'__readout_error_value':
                    readout_e_vec[i] = .1 #val
                    
                if colname == 'q_'+str_i+'__T1_value':
                    T1_vec[i] = val
        
                if colname == 'q_'+str_i+'__T2_value':
                    T2_vec[i] = val
        
            val  = 0.01
            if colname == 'cx0_1_gate_error_value':
                cx_e_vec['2,4'] = [2,4, val]
        
            if colname == 'cx2_1_gate_error_value':
                cx_e_vec['3,4'] = [3,4, val]
        
    return hadamard_e_vec, cx_e_vec, readout_e_vec, T1_vec, T2_vec

