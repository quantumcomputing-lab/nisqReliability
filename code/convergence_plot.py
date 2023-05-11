#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 12:45:28 2023

@author: samudra

# read from parameters pkl
with open('/home/samudra/Desktop/QuantumReliability/data/mc1_may4_2023_9am_run.pkl', 'rb') as f:
    pickle_dict = pickle.load(f)
    
Input:
    list of variables
    mcmc samples from pdf at time t1
    parameters for pdf at time t1 and t2:
        variables in each cluster at time t
        corelation matrix for each cluster at time t
Output:
    BC = np.mean(root_of_ratio)
"""  

# data set full
filename_full = data_folder+'device_data.csv'
data_set_full = get_data_set(filename_full)

# data set by month
data_set_monthly={}
for i in range(len( data_set_monthly )):
    if i ==0:
        filename_this = data_folder+'device_data_12_2021.csv'        
    else:
        filename_this = data_folder+'device_data_'+str(i)+'_2022.csv'

    data_set_monthly[i] = get_data_set(filename_this, header_list_for_corr)
    print(filename_this)

# # get mcmc samples and parameters from pickle
# parameters = pickle_dict["parameters"] # exists
# mcmc_samples = pickle_dict["mcmc_samples"] # exists

monthly_correlation_matrix={}
for i in range(len( data_set_monthly )):
    monthly_correlation_matrix[i] = get_correlation_matrix(data_set_monthly[i])

num_mcmc_samples = 100_000

convergence_data = {}
for num_mcmc_samples in (100, 500, 1_000, 5_000, 10_000, 20_000, 40_000, 60_0000, 80_000, 100_000):
# for num_mcmc_samples in (500, 1000):
    convergence_data[num_mcmc_samples]={}
    for max_size in (2, 3, 4, 5, 6, 7, 8, 9, 10):
    # for max_size in (2,3):
        convergence_data[num_mcmc_samples][max_size]={}
        # marker = num_mcmc_samples
        # print("Doing MCMC size ", num_mcmc_samples )
    
        print("Doing max size ", max_size )    

        threshold_for_low_correlation, cluster_list_for_this_month = get_threshold(monthly_correlation_matrix, 
                                                                                   max_size = max_size)
        print("threshold_for_low_correlation=", threshold_for_low_correlation)
        
        #estimate the joint distribution parameters
        parameters={}
        for i in np.arange(1, len( data_set_monthly )):
            parameters[i]={}
            parameters[i]['marginal'] = get_marginal_parameters(data_set_monthly[i], header_list_for_corr)
            parameters[i]['correlationMatrix'] = get_correlation_matrix(data_set_monthly[i], 
                                                                        #correlation_assumption,
                                                                        threshold_for_low_correlation)        
        mcmc_samples = {}
        for i in np.arange(1, len( data_set_monthly )):
            print(i)
            marginals_monthly = parameters[i]['marginal']
            correlationMatrix_monthly = parameters[i]['correlationMatrix']
            mcmc_samples[i] = generate_copula_data(marginals_monthly, 
                                                   parameters[i]['correlationMatrix'].copy(), 
                                                   num_mcmc_samples)
    
    
        hellinger            = np.zeros(len( data_set_monthly ))*np.nan
        hellinger_normalized = np.zeros(len( data_set_monthly ))*np.nan
        hellinger_averaged   = np.zeros(len( data_set_monthly ))*np.nan
        # get distance w.r.t t=0
        for t in np.arange(1,len( data_set_monthly )):
            hellinger[t] = np.sqrt(1 - get_BC_16d_distribution(t)) # check if t=0 is zero
            hellinger_normalized[t-1] = np.sqrt(1 - (1-hellinger[t]**2)**(1/16) ) # check if t=0 is zero
            BC_marginals=get_BC_marginals(t)
            hellinger_averaged[t]  = np.sqrt(1-BC_marginals).mean()
    
        convergence_data[num_mcmc_samples][max_size]["hellinger"]=hellinger
        convergence_data[num_mcmc_samples][max_size]["hellinger_normalized"]=hellinger_normalized
        convergence_data[num_mcmc_samples][max_size]["hellinger_averaged"]=hellinger_averaged

# for num_mcmc_samples in (1_000, 10_000, 40_000, 100_000):
for max_size in (5, 6, 7, 8):
    # marker = num_mcmc_samples
    marker = max_size
    plt.plot( convergence_data[marker]["hellinger_normalized"], label=max_size)
plt.legend()


plt.clf()
for max_size in (2, 6, 8, 10):
    test=[convergence_data[i][max_size]["hellinger_normalized"][8] for i in 
          (100, 500, 1_000, 5_000, 10_000, 20_000, 40_000, 60_0000, 80_000, 100_000)]
    plt.plot(test, label="Max cluster size = "+str(max_size), marker="*", linestyle="-.")
plt.legend()
plt.gca().set_xticks(np.arange(1,11))
ticks = np.array(['100', '500', '1K', '5K', '10K', '20K', '40K', '60K', '80K', '100K'])
plt.gca().set_xticklabels(ticks)
plt.xlabel('Number of MCMC samples')
plt.ylabel('Hellinger distance')
sns.despine()
plt.savefig('convergence_and_deciding_cluster_size.png', bbox_inches = "tight", dpi=300)

