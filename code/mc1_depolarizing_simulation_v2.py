#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:44:58 2022
@author: samudra
"""
# config
rho = 0.80
t_tot = 16#1e-6 #1ms
time_vec = np.arange(0,16) #np.linspace(0, 1, 1000)*t_tot

e_mean = 0.3
e_std = 0.1
var_m = 20 #upto 20 is okay
e_vec = np.linspace(0, 1, 1000)
#time_vec = np.linspace(0, 1, 1000)*t_tot*100

k = t_tot / ( var_m/ ( 1+(var_m-1)/(1-e_mean*(1-e_mean)/e_std**2) ) - 1 )
alpha_0 = e_mean * k * ( (e_mean - e_mean**2)/e_std**2-1 )
beta_0 = (1-e_mean) * k * ( (e_mean - e_mean**2)/e_std**2 - 1 )
if k<0 or alpha_0<0 or beta_0<0:
    print("STOP! error")
    

ydata={}
for num_qubits in (4,8,12):
    print(num_qubits)
    # colnames
    colnames = []
    for i in range(num_qubits):
        colnames.append( 'e' + str(i) )
    
    # define correlationMatrix_monthly
    correlationMatrix_monthly = pd.DataFrame(columns = colnames, index = colnames)
    for i in range(num_qubits):
        colname =  'e' + str(i)
        correlationMatrix_monthly.loc[colname, colname] = 1.0
        for j in range(i+1, num_qubits):
            rowname =  'e' + str(j)
            #print(i,j)
            correlationMatrix_monthly.loc[rowname, colname] = rho
            correlationMatrix_monthly.loc[colname, rowname] = rho
    
    # define marginals_monthly
    num_mcmc_samples = 10_000 #this is okay
    parameters={}
    mcmc_samples={}
    for time_ind in np.arange(0,16):
        print(time_ind)
        #time_ind = 0
        t=time_vec[time_ind]
        a=alpha_0/(k+t)
        b=beta_0/(k+t)
        
        marginals_monthly = {}
        for e_ind in range(num_qubits):
            item =  'e' + str(e_ind)
            marginals_monthly[item] = {}    
            marginals_monthly[item]['type'] = 'beta'
            marginals_monthly[item]['alpha'] = a
            marginals_monthly[item]['beta'] = b
            marginals_monthly[item]['loc'] = 0
            marginals_monthly[item]['scale'] = 1
    
        parameters[time_ind]={}
        parameters[time_ind]['marginal'] = marginals_monthly
        parameters[time_ind]['correlationMatrix'] = correlationMatrix_monthly
            
        # get the mcmc samples for the 4 errors
        mcmc_samples[time_ind] = generate_copula_data(marginals_monthly, 
                                                      correlationMatrix_monthly, 
                                                      num_mcmc_samples)
    
    # compute the hellinger distance and hence smax
    hellinger=np.zeros(16)
    for time_ind in np.arange(0,16):
        hellinger[time_ind] = get_hellinger_using_mcmc(g_params=parameters[time_ind], 
                                                       f_params=parameters[0], 
                                                       f_samples=mcmc_samples[0])
    
    smax = np.sqrt(4* hellinger**2 * (2-hellinger**2))
    #plt.plot(smax)
    
    # compute s
    mean_obs=np.zeros(16)
    std_obs=np.zeros(16)
    for time_ind in np.arange(0,16):
        mean_obs[time_ind] = np.mean( np.prod( 1-mcmc_samples[time_ind]/2, axis=1 ) )
        std_obs[time_ind] = np.std( np.prod( 1-mcmc_samples[time_ind]/2, axis=1 ) )
        
    s_mean = mean_obs-mean_obs[0]
    s_std = std_obs/np.sqrt(num_mcmc_samples)
    ydata[num_qubits] = s_mean/smax # first value is zero for smax; you will get a warning; it is benign
    # ydata[0]=0 This is not correct to do because technically it is nan as the denominator is 0
    
# plot s by smax
#plt.plot(time_vec, s_mean)
#plt.plot(time_vec, smax)
ax = plt.gca()
ax.set_xticks(time_vec)
labels=[]
for i in range(0, 16):
    labels.append(f"month-{i}")
ax.set_xticklabels(labels, rotation = 45, ha="right")

for num_qubits in (4,8,12):
    plt.plot(time_vec[1:], ydata[num_qubits][1:], label = str(num_qubits) + ' qubits')

plt.legend()
sns.despine()
#plt.xlabel("Month")
plt.ylabel("Ratio of s to s(max)")
plt.savefig("depol_sbysmax.png", bbox_inches = "tight", dpi=300)
print( np.min( ydata[4][1:] ) , np.max( ydata[4][1:] ))
print( np.min( ydata[8][1:] ) , np.max( ydata[8][1:] ))
print( np.min( ydata[12][1:] ) , np.max( ydata[12][1:] ))

#plt.errorbar(time_vec[1:], s_mean[1:]/smax[1:], yerr = s_std[1:], label = 'TBD')
"""
# write to pickle
with open('/home/samudra/Desktop/QuantumReliability/data/depol_simulationdata_n12.pkl', 'wb') as f:
    pickle.dump([time_vec, s_mean, s_max, s_std], f, protocol=pickle.HIGHEST_PROTOCOL)
# read from pickle
for i in (4,8,12):
    with open('/home/samudra/Desktop/QuantumReliability/data/depol_simulationdata_n'+str(i)+'.pkl', 'rb') as f:
        time_vec, s_mean, s_max, s_std = pickle.load(f)
        plt.errorbar(time_vec[4:], s_mean[4:]/smax[4:], yerr = s_std[4:], label = str(i) + ' qubiits')
plt.legend()
sns.despine()
plt.xlabel("Month index (starts from April and ends in Dec)")
plt.ylabel("Ratio of sensitivity ($s$) to\n upper bound ($s_{max}$)")
"""

# convergence
"""
samples_vec = [int(item) for item in np.exp( np.linspace(2,14,20) )]
mcmc_samples={}
for num_mcmc_samples in samples_vec:
    print(np.log(num_mcmc_samples))
    mcmc_samples[num_mcmc_samples] = generate_copula_data(parameters[0]['marginal'], 
                                                  parameters[0]['correlationMatrix'], 
                                                  num_mcmc_samples)
h={}
for num_mcmc_samples in samples_vec:
    print(np.log(num_mcmc_samples))
    h[num_mcmc_samples] = get_hellinger_using_mcmc(g_params=parameters[12], 
                                                   f_params=parameters[0], 
                                                   f_samples=mcmc_samples[num_mcmc_samples])
plt.scatter(np.log(samples_vec), h.values())
sns.despine()
#plt.legend()
#plt.legend(loc='upper left')
plt.xlabel("Log of Number of Samples")
plt.ylabel("Ratio of sensitivity ($s$) to\n upper bound ($s_{max}$)")

# write to pickle
# with open('/home/samudra/Desktop/QuantumReliability/data/depol_simulation_convergence_data.pkl', 'wb') as f:
#     pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
# read from pickle
# with open('/home/samudra/Desktop/QuantumReliability/data/depol_simulation_convergence_data.pkl', 'rb') as f:
    # h = pickle.load(f)
"""



#################################################
# reproducibility vs Holder bound
#################################################
# plt.close()
# analytical_mean_dict = {}
# for n in (4, 8, 16, 40):
#     print(n)
#     analytical_mean = np.zeros(len(time_vec))
#     analytical_std = np.zeros(len(time_vec))
#     hellinger = np.zeros(len(time_vec))
#     for i in range(len(time_vec)):
#         t=time_vec[i]
#         a1=alpha_0/k
#         a2=alpha_0/(k+t)
#         b1=beta_0/k
#         b2=beta_0/(k+t)
#         num = scipy.special.beta( (a1+a2)/2, (b1+b2)/2 )
#         den1 = np.sqrt( scipy.special.beta( a1, b1 ))
#         den2 = np.sqrt( scipy.special.beta( a2, b2 ))
#         hellinger[i] =  np.sqrt(1 - num / (den1*den2) )
    
#         #mean_obs_ref, var_obs_ref = get_mean_and_var_of_observable(n, alpha_0, beta_0, k, t, method='analytical')
#         mean_obs_ref, var_obs_ref = get_mean_and_var_of_observable(n, alpha_0, beta_0, k, t, method='mcmc')
#         analytical_std[i] = np.sqrt(var_obs_ref)
#         analytical_mean[i] = mean_obs_ref
    
#     s_t = np.abs( ( analytical_mean- analytical_mean[0] ))    
#     c = 1
#     s_max = np.sqrt( 4 * c**2 * hellinger**2 * (2-hellinger**2) )
#     analytical_mean_dict[n] = s_t[1:]/s_max[1:]
#     plt.plot(time_vec[1:]/max(time_vec[1:]), s_t[1:]/s_max[1:], label = "Problem size = "+ str(n) + " bits")
#     #plt.plot(time_vec[1:]/max(time_vec[1:]), analytical_mean[1:], label = "Problem size = "+ str(n) + " bits")
#     #plt.plot(time_vec[1:]/max(time_vec[1:]), analytical_mean[1:] - analytical_mean[0], label = "Problem size = "+ str(n) + " bits")

# sns.despine()
# #plt.legend()
# plt.legend(loc='upper left')
# plt.ylim(0,.4)
# plt.xlabel("Time (a.u.)")
# plt.ylabel("Ratio of reproducibility ($s$) to\n upper bound ($s_{max}$)")
# plt.savefig("time-varying_reproducibility_vs_holder_bound.png", bbox_inches = "tight", dpi=300)

#################################################
# time-varying distributions
#################################################

# # Progressively worsening depolarizing quantum channel
# for t in np.linspace(0, .04, 8)*t_tot:
#     fx = scipy.stats.beta.pdf(e_vec, alpha_0/(k+t), beta_0/(k+t))
#     plt.plot(e_vec, fx, label = 'Time = ' + str(t/1e-6)[0:5]+ '$\mu$s', alpha=0.8, linestyle = '--')
# #plt.legend()
# sns.despine()
# plt.xlabel("Depolarizing error $x_i$ for the $i$-th single-qubit noise channel\n ($x_i \in [0, 1]$)")
# plt.ylabel("Probability distribution function")
# plt.savefig("time_varying_distributions.png", bbox_inches = "tight", dpi=300)

#################################################
#### ROUTINES ####
#################################################
"""
#################################################
# 2 samples from the time-varying distributions
#################################################
# Monte Carlo samples from the time-varying distributions
for t in np.linspace(0, 1, 3)*t_tot:
    samples = scipy.stats.beta.rvs(alpha_0/(k+t), beta_0/(k+t), size = 1_00_000)
    plt.hist(samples, 500, label = 'time = ' + str(t/1e-6)[0:5]+ '$\mu$s', alpha = 0.3)
plt.legend()
sns.despine()
plt.xlabel("Depolarizing error,  $e \in [0, 1]$)")
plt.ylabel("Histogram of error samples")
#plt.savefig("time_varying_histograms.png", bbox_inches = "tight", dpi=300)

#################################################
# 3 time-varying variance [Not needed - goto 4 directly]
#################################################
# Monte Carlo samples from the time-varying distributions
t = np.linspace(0, 1, 1000)*t_tot
my_mean, my_var = scipy.stats.beta.stats(alpha_0/(k+t), beta_0/(k+t), moments='mv')
plt.plot(t/1e-6, np.sqrt(my_var) )
#plt.legend()
sns.despine()
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Time-varying standard deviation\n of the depolarizing error distribution")
#plt.savefig("time_varying_std.png", bbox_inches = "tight", dpi=300)

#################################################
# 4 time-varying Hellinger w.r.t. t_0
#################################################
time_vec = np.linspace(0, 1, 1000)*t_tot
hellinger = np.zeros(len(time_vec))
for i in range(len(time_vec)):
    t=time_vec[i]
    a1=alpha_0/k
    a2=alpha_0/(k+t)
    b1=beta_0/k
    b2=beta_0/(k+t)
    num = scipy.special.beta( (a1+a2)/2, (b1+b2)/2 )
    den1 = np.sqrt( scipy.special.beta( a1, b1 ))
    den2 = np.sqrt( scipy.special.beta( a2, b2 ))
    hellinger[i] =  np.sqrt(1 - num / (den1*den2) )
my_mean, my_var = scipy.stats.beta.stats(alpha_0/(k+time_vec), beta_0/(k+time_vec), moments='mv')
plt.plot(time_vec/1e-6, hellinger, label = "Hellinger distance from t=0")
plt.plot(time_vec/1e-6, np.sqrt(my_var), label = "Standard Deviation at time t" )
plt.legend()
sns.despine()
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Time-varying distance and standard deviation\n of the depolarizing error distribution")
#plt.savefig("time_varying_std_and_hellinger.png", bbox_inches = "tight", dpi=300)

### Output set ###
#################################################
# 5 simulation vs analytical mean of the observable
#################################################
# Monte carlo sampling for mean of observable
L = 10 #set this to 100
time_vec = np.linspace(0, 1, L)*t_tot*2

# First analytical
analytical_mean = np.zeros(L)
i=-1
for t in time_vec:
    i=i+1
    print(i)
    t = time_vec[i]
    mean_obs_ref = 0
    for k1 in np.arange(0, n+1):
        nChooseK1 = math.comb(n,k1)
        p = np.arange(0, k1)
        # check the 1/2 factor again: I claim it is 2/3*3/4 following Mike and Ike
        mean_obs_ref += (-1)**k1 * nChooseK1 * (1/2)**k1 * np.prod( (alpha_0 + p*(k+t)) / (alpha_0+ beta_0 + p*(k+t)))
    analytical_mean[i] = mean_obs_ref

# # Next Monte Carlo: this takes too much time: save in csv file once satisfied
# sample_mean = np.zeros(L)
# i=-1
# for t in time_vec:
#     i=i+1
#     print(i)
#     t = time_vec[i]
#     samples = scipy.stats.beta.rvs(alpha_0/(k+t), beta_0/(k+t), size = 100)
#     sample_pr_of_success = runSim_pr_of_success(samples)
#     sample_mean[i] = np.mean( sample_pr_of_success )
# ax = plt.gca()
# plt.plot(sample_mean); plt.plot(analytical_mean)
# plt.plot(np.abs( sample_mean/ analytical_mean-1)*100)
# #plt.bar(np.arange(1, 21, 2), sample_mean[::10], width=0.4, label = 'analytical')
# #plt.bar(np.arange(1.5, 21.5, 2), analytical_mean[::10], width=0.4, label = 'simulation' )
# #ax.set_xticks(np.arange(1.25, 21.25, 2))
# #ax.set_xticklabels( [str(item)[0:5] for item in time_vec[::10]/1e-6] ) 
# #plt.ylim(0.4, )
# plt.legend()
# sns.despine()
# plt.xlabel("Time ($\mu$s)")
# plt.ylabel("Bernstein-Vazirani probability of success\n with constant mean (=0.3),\n time-varying depolarizing channel")
#plt.savefig("sim_vs_analytical_pr_success.png", bbox_inches = "tight", dpi=300)

#################################################
# 6 time-varying reproducibility
#################################################
s_t = ( sample_mean - sample_mean[0] )**2
plt.plot(time_vec/1e-6, s_t )
sns.despine()
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Time-varying reproducibility,\n $s(t) = [ <O>_t - <O>_0]^2$")
#plt.savefig("time-varying_reproducibility.png", bbox_inches = "tight", dpi=300)

#################################################
# 8 when bound recipe is followed, reproducibility stays within the tolerance
#################################################

time_vec = np.linspace(0, 1, 10_000)*t_tot*3 # note this has to be sufficiently granular otherwise you will get garbage

hellinger = np.zeros(len(time_vec))
for i in range(len(time_vec)):
    t=time_vec[i]
    a1=alpha_0/k
    a2=alpha_0/(k+t)
    b1=beta_0/k
    b2=beta_0/(k+t)
    num = scipy.special.beta( (a1+a2)/2, (b1+b2)/2 )
    den1 = np.sqrt( scipy.special.beta( a1, b1 ))
    den2 = np.sqrt( scipy.special.beta( a2, b2 ))
    hellinger[i] =  np.sqrt(1 - num / (den1*den2) )

#eps_sq_vec = np.zeros(8)
BV_nqubits_vec=np.arange(3, 25)

#for eps in np.arange(0.01, .99, 0.04):
for eps in (0.01, 0.04, 0.1, 0.40, 0.80, 0.95, 0.99):    
    counter = 0
    s_vec      = np.zeros(len(BV_nqubits_vec))
    for n in BV_nqubits_vec:
        print(n)
        #eps = 1/(1 + math.pow(2,-(n-1)) ) #0.5
        #eps = 0.1
        #eps_sq_vec[counter] = eps**2
        hmax = np.sqrt(1-np.sqrt( 1 - eps**2/4/c**2 ))
    
        t0 = time_vec[np.nanargmin(np.abs(hellinger-hmax))]
        s_vec[counter]   = ( get_mean_obs_t(t0, n)-get_mean_obs_t(0, n) )**2/eps**2
        counter+=1
    plt.plot(BV_nqubits_vec, np.sqrt(s_vec), marker = '*' ,label = "eps = "+str(eps)[0:5])

sns.despine()
plt.legend()
plt.xlabel("Problem size of Bernstein-Vazirani (number of qubits)")
plt.ylabel("Normalized reproducibility\n $\sqrt{s(t)/ \epsilon^2}$")
#plt.savefig("ratio_of_sqrt_s_to_eps", bbox_inches = "tight", dpi=300)

#### ROUTINES ####
def runBVdepol(e):
    #pr_of_success = (1-2/3*e)**n
    # this needs bv_simple_jan12_2023.py to be loaded first
    secret_word = '0011'
    bv_circuit = my_BV(n = n, s = secret_word, num_circuits=1)
    simulator_results = execute(bv_circuit,\
               #backend  = device,\
               backend  = simulator,\
              #initial_layout = layout,\
              noise_model    = my_noise_model(e),\
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
    edash = 3/4*e
    #print("Expected = ", (1-2*edash/3)**n)
    return pr_success 

def runSim_pr_of_success(samples):
    # this is the line that needs to be changed
    #sample_pr_of_success = (1-2/3*samples)**n
    sample_pr_of_success=[]
    for err in samples:
        val = runBVdepol(err)
        sample_pr_of_success.append(val)
    return np.array(sample_pr_of_success)

def get_mean_obs_t(t, n):
    mean_obs = 0
    for k1 in np.arange(0, n+1):
        nChooseK1 = math.comb(n,k1)
        p = np.arange(0, k1)
        # check the 1/2 factor again: I claim it is 2/3*3/4 following Mike and Ike
        mean_obs += (-1)**k1 * nChooseK1 * (1/2)**k1 * np.prod( (alpha_0 + p*(k+t)) / (alpha_0+ beta_0 + p*(k+t)) )
    return mean_obs

"""