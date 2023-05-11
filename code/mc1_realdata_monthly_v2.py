#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:47:02 2023
@author: samudra

pickle_dict = {}
pickle_dict["monthly_correlation_matrix"]=monthly_correlation_matrix
pickle_dict["threshold_for_low_correlation"]=threshold_for_low_correlation
pickle_dict["max_size"]=max_size
pickle_dict["cluster_list"]=cluster_list
pickle_dict["parameters"]=parameters
pickle_dict["mcmc_samples"]=mcmc_samples
pickle_dict["t_circuit"]=t_circuit
pickle_dict["pr_success_monthly"]=pr_success_monthly
pickle_dict["current_time"]=current_time
pickle_dict["hellinger"]=hellinger
pickle_dict["hellinger_normalized"]=hellinger_normalized
pickle_dict["hellinger_averaged"]=hellinger_averaged
pickle_dict["hellinger_for_each_param"]=hellinger_for_each_param

# write to pickle
with open('/home/samudra/mc1_may9_2023_11am_run.pkl', 'wb') as f:
    pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# read from parameters pkl
with open('/home/samudra/Desktop/QuantumReliability/data/mc1_may9_2023_11am_run.pkl', 'rb') as f:
    pickle_dict = pickle.load(f)
    
# unpickle
monthly_correlation_matrix=pickle_dict["monthly_correlation_matrix"]
threshold_for_low_correlation=pickle_dict["threshold_for_low_correlation"]
cluster_list=pickle_dict["cluster_list"]
parameters=pickle_dict["parameters"]
mcmc_samples=pickle_dict["mcmc_samples"]
t_circuit=pickle_dict["t_circuit"]
pr_success_monthly=pickle_dict["pr_success_monthly"]
current_time=pickle_dict["current_time"]

"""
# data set full
# filename_full = data_folder+'device_data.csv'
# data_set_full = get_data_set(filename_full)

# data set by month
data_set_monthly={}
k=0
for i in np.arange(1, 13):
    filename_this = data_folder+'device_data_'+str(i)+'_2022.csv'
    data_set_monthly[k] = get_data_set(filename_this, noise_metrics_list_16); k=k+1
    print(filename_this)
for i in np.arange(1, 5):
    filename_this = data_folder+'device_data_'+str(i)+'_2023.csv'
    data_set_monthly[k] = get_data_set(filename_this, noise_metrics_list_16); k=k+1
    print(filename_this)

# go through all the months and get the largest cluster size
monthly_correlation_matrix={}
for i in range(len( data_set_monthly )):
    monthly_correlation_matrix[i] = get_correlation_matrix(data_set_monthly[i], 0)                

max_size=3
threshold_for_low_correlation, cluster_list = get_threshold(monthly_correlation_matrix, max_size=3)

#estimate the joint distribution parameters
parameters={}
for i in np.arange(0, len( data_set_monthly )):
    parameters[i]={}
    parameters[i]['marginal'] = get_marginal_parameters(data_set_monthly[i], noise_metrics_list_16)
    parameters[i]['correlationMatrix'] = get_correlation_matrix(data_set_monthly[i], 
                                                                threshold_for_low_correlation)

# find the number of samples required for the hellinger computation convergence
# plot_hellinger_convergence(parameters[0], parameters[1])

# generate mcmc samples
mcmc_samples = {}
for i in range(len( data_set_monthly )):
    print(i)
    marginals_monthly = parameters[i]['marginal']
    correlationMatrix_monthly = parameters[i]['correlationMatrix']
    mcmc_samples[i] = generate_copula_data(marginals_monthly, 
                                           parameters[i]['correlationMatrix'], 
                                           num_mcmc_samples)

# show why coupulas are important
# plot_copula_graphs(data_set_monthly, mcmc_samples) # why does this take such long time?
# plot_correlation_matrix(monthly_correlation_matrix[15])

### compute the three kinds of distances ###
hellinger            = np.zeros(len( data_set_monthly ))*np.nan
hellinger_normalized = np.zeros(len( data_set_monthly ))*np.nan
hellinger_averaged   = np.zeros(len( data_set_monthly ))*np.nan
hellinger_for_each_param = {}
# get distance w.r.t t=0
for t in np.arange(0,len( data_set_monthly )):
    print(t)
    hellinger[t] = np.sqrt(1 - get_BC_16d_distribution(t)) # check if t=0 is zero
    hellinger_normalized[t] = np.sqrt(1 - (1-hellinger[t]**2)**(1/16) ) # check if t=0 is zero
    BC_marginals=get_BC_marginals(t)
    hellinger_for_each_param[t] = np.sqrt(1-BC_marginals)
    hellinger_averaged[t]  = np.sqrt(1-BC_marginals).mean()

plot_distance_from_ref_3kinds(hellinger, hellinger_normalized, hellinger_averaged)

# get prob of success from qiskit
t_circuit = get_time_circuit()
pr_success_monthly={}
for i in range(len(parameters)):
    current_time = datetime.now()
    print("Current time is:", current_time)
    pr_success_monthly[i] = get_pr_success_monthly(mcmc_samples[i].loc[0:n_qiskit_samples-1, :])
    print("month done = ", i)
    current_time = datetime.now()
    print("Current time is:", current_time)


# attribution using hellinger_for_each_param and the monthly_h_avg
plot_hellinger_attribution(hellinger_for_each_param[15], header_dict)
# attribution, hellinger_for_each_param, monthly_h_avg = get_hellinger_attribution(parameters)

# Illustrating the problem of hellinger rising fast with dimension at higher dimensions
plt.clf()
x=np.arange(1,21); 
y = np.sqrt(1-np.exp( np.log(1-np.mean( [ np.mean( hellinger_for_each_param[i]) for i in np.arange(1,len( data_set_monthly ))] )**2)*x )); 
plt.plot(x,y)
ticks = x
plt.gca().set_xticks(ticks)
labels = [int(item) for item in x]
plt.gca().set_xticklabels(labels, rotation = 0, ha="right")
sns.despine(left=True, bottom=True)
plt.grid(True, which='both')    
plt.xlabel(r'Dimension of the joint distribution')
plt.ylabel('Distance')
plt.savefig('hellinger_rising_too_fast_with_dim.png', bbox_inches = "tight", dpi=300)

# s by smax
smax = 2* hellinger * np.sqrt(2-hellinger**2)
s = np.zeros((len(parameters)-1, n_qiskit_samples))
s_to_smax = np.zeros((len(parameters)-1, n_qiskit_samples))
for i in np.arange(1, len(parameters)):
    print(i)
    s[i-1,:] = np.abs( pr_success_monthly[i] - np.nanmean( pr_success_monthly[0] ) )
    s_to_smax[i-1,:] = s[i-1,:] / smax[i]

fig, ax = plt.subplots()
ax.boxplot(np.transpose(s_to_smax)*100, showfliers=False)
ax.set_xticklabels(['Feb-22','Mar-22','Apr-22','May-22','Jun-22','Jul-22',
                    'Aug-22','Sep-22','Oct-22','Nov-22','Dec-22',
                    'Jan-23','Feb-23','Mar-23','Apr-23'], rotation=45)
sns.despine()
plt.ylabel("100 times the ratio of s to s(max)")
plt.savefig("s_by_smax.png", bbox_inches = "tight", dpi=300)
