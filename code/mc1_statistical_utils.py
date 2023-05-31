#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan len( data_set_monthly ) 09:56:17 2023

@author: samudra
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as la
import scipy.stats
import seaborn as sns
from statsmodels.stats.correlation_tools import cov_nearest
import copy
import math
import pickle
from datetime import datetime
import warnings

# #### ROUTINES ####
# def plot_ts_by_month(monthly_data, ylabel, figname):
#     plt.plot( monthly_data )
#     plt.plot(np.arange(0,len( data_set_monthly )), monthly_data)
#     #plt.bar(ticks, pvalue)
#     ticks = np.arange(0, len( data_set_monthly ))
#     plt.gca().set_xticks(ticks)
#     labels = ['Dec-21', 'Jan-22','Feb-22','Mar-22','Apr-22','May-22','Jun-22','Jul-22',
#               'Aug-22','Sep-22','Oct-22','Nov-22','Dec-22']
#     plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
#     sns.despine(left=True, bottom=True)
#     plt.grid(True, which='both')    
#     plt.xlabel(r'Month-Year')
#     plt.ylabel(ylabel)
#     plt.savefig(figname, bbox_inches = "tight", dpi=300)
    
# # print the parameters for each of parameters
# def print_parameters_by_month():    
#     for param_index in range(0,16):
#         #print(header_dict[param_index])
#         data = []#[('parameter', 'distribution type', 'month-year', 'alpha', 'beta')]
#         nice_param_name = header_dict[param_index]
#         param_name = noise_metrics_list_16[param_index]    
#         for mnth in range(0,len( data_set_monthly )):
#             dist_type = parameters[mnth]['marginal'][param_name]['type']
#             alpha = round( parameters[mnth]['marginal'][param_name]['alpha'],1)
#             if dist_type=='beta':
#                 beta = round( parameters[mnth]['marginal'][param_name]['beta'],0)
#             if dist_type=='gamma':
#                 scale = parameters[mnth]['marginal'][param_name]['scale']
#                 beta = round( 1/scale, 2)
#             data.append((nice_param_name, dist_type, month_dict[mnth], alpha, beta))
            
#         # Write CSV file
#         with open("test_"+str(param_index)+".csv", "wt") as fp:
#             writer = csv.writer(fp, delimiter=",")
#             # writer.writerow(["your", "header", "foo"])  # write header
#             writer.writerows(data)


#     for param_index in range(0,16):
#         # Read CSV file
#         with open("test_"+str(param_index)+".csv") as fp:
#             reader = csv.reader(fp, delimiter=",", quotechar='"')
#             # next(reader, None)  # skip the headers
#             s=""
#             for row in reader:
#                 #print(row)
#                 s = s + row[2] + " & " + row[3] + " & " + row[4] + "\\\\ \hline\n"
#             s1 = "\\begin{table}[htbp]\n"+\
#             "\\caption{" + row[0] + " [" + row[1] + " distribution]}\n"+\
#             "\\begin{center}\n"+\
#             "\\begin{tabular}{|l|c|c|}\hline\n"+\
#             "Month-year & alpha(t) & beta(t) \\\\ \hline\n"+\
#             s+\
#             "\\end{tabular}\\label{tab:parameter_"+str(param_index)+"}\n"+\
#             "\\end{center}\n"+\
#             "\\end{table}\n"
#             #print(s1)
#             with open("latex_table_parameter_"+str(param_index)+".txt", "w") as text_file:
#                 text_file.write(s1)




# hellinger attribution
def plot_hellinger_attribution(input_data, header_dict):
    # input_data = hellinger_for_each_param[15]
    total_amount = np.mean(input_data)
    hdata = input_data/np.sum(input_data)*np.mean(input_data)
    attribution={}
    for k in range(len(hdata)):
        attribution[header_dict[k]]=-hdata[k]
    attribution=dict(sorted(attribution.items(), key=lambda item: item[1]))

    data  = list(attribution.values())
    index = list(attribution.keys())

    data = {'amount': [total_amount] + data}
    index = ['Total']+index
    
    trans = pd.DataFrame(data=data,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)
    total = trans.sum().amount
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    my_plot = trans.plot(kind='bar', stacked=True, bottom=blank, legend=None, 
                          figsize=(10, 5), rot=90, 
                          #ylim = (0,0.65),
                          ylim = (None, None),
                          alpha = 0.6 )
    my_plot.plot(step.index, step.values,'b', linestyle = '--', color = 'k', alpha=0.4)
    my_plot.set_xlabel("\n Quantum error types", fontsize = 14)
    ylabel = 'Hellinger distance attribution'
    my_plot.set_ylabel(ylabel)
    # plt.tight_layout() #only for multiple subplot version
    sns.despine()
    fig = my_plot.get_figure()
    figname = 'distance_attribution_ind.png'
    fig.savefig(figname, bbox_inches = "tight", dpi=300)

# def get_beta_hellinger(alpha1, beta1, loc1, scale1, 
#                    alpha2, beta2, loc2, scale2):
#     avg_a = np.mean([alpha1, alpha2])
#     avg_b = np.mean([beta1, beta2])
#     log_1_minus_hsq_1d = scipy.special.betaln(avg_a, avg_b) \
#                             - 0.5*scipy.special.betaln(alpha1, beta1)\
#                                 - 0.5*scipy.special.betaln(alpha2, beta2)
#     h = np.sqrt( 1-np.exp( log_1_minus_hsq_1d ) )
#     return h, log_1_minus_hsq_1d

# def get_gamma_hellinger(alpha1, loc1, scale1, 
#                        alpha2, loc2, scale2):
#     beta1 = 1/scale1
#     beta2 = 1/scale2

#     avg_a = np.mean([alpha1, alpha2])
#     avg_b = np.mean([beta1, beta2])
    
#     log_1_minus_hsq_1d = scipy.special.gammaln(avg_a) - avg_a * np.log(avg_b)\
#                          + 0.5*(alpha1*np.log(beta1) + alpha2*np.log(beta2) \
#                          - scipy.special.gammaln(alpha1) - scipy.special.gammaln(alpha2))

#     h = np.sqrt( 1-np.exp( log_1_minus_hsq_1d ) )
#     return h, log_1_minus_hsq_1d 

# def plot_hellinger_convergence(params_full_dict_f, params_full_dict_g):
#     # params_full_dict_f = parameters[0]
#     # params_full_dict_g = parameters[1]
#     sample_size_vector = [int(item) for item in np.exp(np.arange(3, 12, 0.1))]
#     hellinger_vec = np.zeros(len(sample_size_vector))*np.nan
    
#     k=-1
#     for num_mcmc_samples in sample_size_vector:
#         k=k+1
#         print(k)
#         hellinger_vec[k] = get_hellinger_at_nsamples(params_full_dict_f, params_full_dict_g, num_mcmc_samples)

#     plt.plot( np.log(sample_size_vector), hellinger_vec, linestyle='', marker='*') # smaller range
#     #plt.plot( np.log(sample_size_vector), np.log(1-hellinger_vec**2), linestyle='', marker='*') # bigger range
#     #plt.ylim(.5, 1)
#     sns.despine()
#     plt.ylabel("Hellinger distance\n computed using MCMC")
#     plt.xlabel("Log of Number of Samples")
#     plt.savefig("test.png", bbox_inches = "tight", dpi=300)

# def get_hellinger_at_nsamples(params_full_dict_f, params_full_dict_g, num_mcmc_samples, f_samples_full):
#     #params_full_dict_f = parameters[0]
#     #params_full_dict_g = parameters[1]
#     #f_samples_full=mcmc_samples[0]
#     log_1_minus_hsq_1dsum = 0
#     log_1_minus_hsq_by_cluster=[]
#     for item in correlation_assumption['ind_list']:
#         colname = noise_metrics_list_16[item]
#         print(colname)
#         dist_type = params_full_dict_f['marginal'][colname]['type']
#         print(dist_type)
 
#         if dist_type == 'beta':
#             alpha1 = params_full_dict_f['marginal'][colname]['alpha']
#             beta1 = params_full_dict_f['marginal'][colname]['beta']
#             loc1 = params_full_dict_f['marginal'][colname]['loc']
#             scale1 = params_full_dict_f['marginal'][colname]['scale']
#             #print(alpha1, beta1, loc1, scale1)

#             alpha2 = params_full_dict_g['marginal'][colname]['alpha']
#             beta2 = params_full_dict_g['marginal'][colname]['beta']
#             loc2 = params_full_dict_g['marginal'][colname]['loc']
#             scale2 = params_full_dict_g['marginal'][colname]['scale']
#             print(alpha2, beta2, loc2, scale2)

#             _, log_1_minus_hsq_1d = get_beta_hellinger(alpha1, beta1, loc1, scale1, 
#                                                        alpha2, beta2, loc2, scale2)
        
#         if dist_type == 'gamma':
#             alpha1 = params_full_dict_f['marginal'][colname]['alpha']
#             loc1 = params_full_dict_f['marginal'][colname]['loc']
#             scale1 = params_full_dict_f['marginal'][colname]['scale']

#             alpha2 = params_full_dict_g['marginal'][colname]['alpha']
#             loc2 = params_full_dict_g['marginal'][colname]['loc']
#             scale2 = params_full_dict_g['marginal'][colname]['scale']

#             _, log_1_minus_hsq_1d = get_gamma_hellinger(alpha1, loc1, scale1, 
#                                                         alpha2, loc2, scale2)       
        
#         log_1_minus_hsq_by_cluster.append(log_1_minus_hsq_1d)
#         log_1_minus_hsq_1dsum += log_1_minus_hsq_1d

#     log_1_minus_hsq=log_1_minus_hsq_1dsum
#     for restricted_dim in ('3','7'):
#         cols = []
#         f_params = {}
#         g_params = {}
#         f_params['marginal']={}
#         g_params['marginal']={}
#         for item in correlation_assumption[restricted_dim+'d_list']:
#             colname = noise_metrics_list_16[item]
#             f_params['marginal'][colname] = params_full_dict_f['marginal'][colname]
#             g_params['marginal'][colname] = params_full_dict_g['marginal'][colname]
#             #print(colname)
#             cols.append(colname)
        
#         f_params['correlationMatrix'] = params_full_dict_f['correlationMatrix'][cols].loc[cols, :]
#         g_params['correlationMatrix'] = params_full_dict_g['correlationMatrix'][cols].loc[cols, :]
        
#         marginals_monthly = f_params['marginal']
#         correlationMatrix_monthly = f_params['correlationMatrix']
        
#         if len( f_samples_full ) != 0:
#             f_samples = f_samples_full[cols]
#         else:
#             f_samples = generate_copula_data(marginals_monthly, 
#                                              correlationMatrix_monthly, 
#                                              num_mcmc_samples)
#         temp_val = np.log( 1 - get_hellinger_using_mcmc(g_params, f_params, f_samples)**2)
#         log_1_minus_hsq_by_cluster.append(temp_val)
#         log_1_minus_hsq += temp_val
    
#     return np.sqrt( 1 - np.exp( log_1_minus_hsq ) ), log_1_minus_hsq_by_cluster

# # with correlation
def plot_copula_graphs(data_set_monthly, mcmc_samples):
    col1='hadamard_0_gate_error_value'
    col2='hadamard_3_gate_error_value'
    month_num = 15

    # with correlation
    x1 = mcmc_samples[month_num][col1]
    x2 = mcmc_samples[month_num][col2]   
        
    # no correlation
    fit_data = data_set_monthly[month_num][col1]
    fit_data = fit_data[~np.isnan(fit_data)]
    (fit_alpha, fit_beta, fit_loc, fit_scale) = scipy.stats.beta.fit(fit_data, floc=0, fscale=1)
    x1_nocorr = scipy.stats.beta.rvs(fit_alpha, fit_beta, size = (10_000,1))
    
    fit_data = data_set_monthly[month_num][col2]
    fit_data = fit_data[~np.isnan(fit_data)]
    (fit_alpha, fit_beta, fit_loc, fit_scale) = scipy.stats.beta.fit(fit_data, floc=0, fscale=1)
    x2_nocorr = scipy.stats.beta.rvs(fit_alpha, fit_beta, size = (10_000,1))

    # boundaries for the graph
    xmin = np.min([ np.percentile(x1, 1),  np.percentile(x1_nocorr, 1)])
    xmax = np.max([ np.percentile(x1, 90), np.percentile(x1_nocorr, 95)])
    ymin = np.min([ np.percentile(x2, 1),  np.percentile(x2_nocorr, 1)])
    ymax = np.max([ np.percentile(x2, 90), np.percentile(x2_nocorr, 95)])

    # plot
    plt.close()    
    h = sns.jointplot(x1*100, x2*100, kind='kde', ylim=(ymin*100, ymax*100), cbar=True, 
                      xlim=(xmin*100, xmax*100), stat_func=None);
    h.set_axis_labels("Hadamard gate error (%) for qubit 0", "Hadamard gate error (%) for qubit 3", fontsize=16)
    # h.set_axis_labels("Hadamard gate error for qubit 1", "T2 time for qubit 2\n($\mu$s)", fontsize=16)
    # h.set_axis_labels("Hadamard gate error for qubit 1", "T2 time for qubit 2", fontsize=16)
    plt.savefig('dist_with_copula.png', bbox_inches = "tight", dpi=300)

    # if you choose a gamma variable    
    # fit_data = data_set_monthly[1][col2]
    # fit_data = fit_data[~np.isnan(fit_data)]
    # (fit_alpha, fit_loc, fit_scale) = scipy.stats.gamma.fit(fit_data)
    # x2_nocorr = scipy.stats.gamma.rvs(fit_alpha, fit_loc, fit_scale, size = (10_000, 1))
    
    plt.close()
    h = sns.jointplot(x1_nocorr*100, x2_nocorr*100, kind='kde', cbar=True, 
                      ylim=(ymin*100, ymax*100), xlim=(xmin*100, xmax*100), stat_func=None);
    h.set_axis_labels("Hadamard gate error (%) for qubit 0", "Hadamard gate error (%) for qubit 3", fontsize=16)
    # h.set_axis_labels("Hadamard gate error for qubit 1", "T2 time for qubit 2\n($\mu$s)", fontsize=16)
    plt.savefig('dist_without_copula.png', bbox_inches = "tight", dpi=300)

def get_correlation_matrix(data_set_this, threshold_for_low_correlation = 0):
    #data_set_this=data_set_monthly[15]
    n_header_list_for_corr = len(noise_metrics_list_16)
    my_corr = data_set_this.dropna().corr()
    for i in range(len(noise_metrics_list_16)):
        my_corr.iloc[i, i] = 1.0
        for j in np.arange(i+1, len(noise_metrics_list_16)):
            colname1 = noise_metrics_list_16[i]
            colname2 = noise_metrics_list_16[j]
            #print(colname1, colname2)
            data1 = data_set_this[colname1]
            data2 = data_set_this[colname2]
            temp = pd.DataFrame({colname1: data1,
                                  colname2: data2,
                                  })
            val = temp.corr()[colname1][colname2]
            if np.abs(val) < threshold_for_low_correlation:
                val = 0
            my_corr.iloc[j,i] = val
            my_corr.iloc[i,j] = val

    my_corr.loc[:,:] = cov_nearest(my_corr, threshold=1e-5)
    # if ~np.all(np.linalg.eigvals(my_corr) > 0):
    #     print("error")

    return my_corr

# def decide_corr_struct(data_set_full, threshold_for_low_correlation):
#     import copy
#     n_header_list_for_corr = len(noise_metrics_list_16)
#     my_corr = np.zeros((n_header_list_for_corr, n_header_list_for_corr))*np.nan
#     for i in range(len(noise_metrics_list_16)):
#         my_corr[i, i] = 1.0
#         for j in np.arange(i+1, len(noise_metrics_list_16)):
#             colname1 = noise_metrics_list_16[i]
#             colname2 = noise_metrics_list_16[j]
#             #print(colname1, colname2)
    
#             month_index = 0
#             # data1 = data_set_monthly[month_index][colname1]
#             # data2 = data_set_monthly[month_index][colname2]
#             data1 = data_set_full[colname1]
#             data2 = data_set_full[colname2]
#             temp = pd.DataFrame({colname1: data1,
#                                  colname2: data2,
#                                  })
#             my_corr[j,i] = temp.corr()[colname1][colname2]
#             my_corr[i,j] = my_corr[j, i]
            
#     my_corr_masked = copy.deepcopy(my_corr)
#     my_corr_masked[np.abs(my_corr)< threshold_for_low_correlation]=0
#     # np.sum( np.abs( my_corr_masked - my_corr) )
#     return my_corr, my_corr_masked

for key, my_corr in monthly_correlation_matrix.items():
    # print(val)
    plot_correlation_matrix(my_corr, figname = 'correlation_matrix_'+str(key)+'.png')

def plot_correlation_matrix(my_corr, figname = 'correlation_matrix.png'):        
    f, ax = plt.subplots(figsize=(11, 9))
    #cmap = sns.diverging_palette(10, 250, as_cmap=True)
    x_labels = ['x' + str(i) for i in range(16)]
    # y_labels = ['x' + str(i) for i in range(16)]
    sns.heatmap(my_corr, 
                #annot=True, 
                #cmap=cmap, 
                xticklabels=x_labels,
                yticklabels=x_labels,
                vmin=-1,vmax=1,center=0,cmap="RdBu",
                square=True, fmt='.02f',
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.savefig(figname, bbox_inches = "tight", dpi=300)


def get_hellinger_using_mcmc(g_params, f_params, f_samples):
    # THESE MUST BE EVALUATED AT SAME SAMPLES!!!
    log_copula_pdf_of_f_at_fsamples = get_log_copula(f_params, f_samples)
    log_copula_pdf_of_g_at_fsamples = get_log_copula(g_params, f_samples)
        
    copula_pdf_of_f_at_fsamples = np.exp( log_copula_pdf_of_f_at_fsamples)
    copula_pdf_of_g_at_fsamples = np.exp( log_copula_pdf_of_g_at_fsamples)
    
    ratio = np.sqrt( copula_pdf_of_g_at_fsamples / copula_pdf_of_f_at_fsamples  )
    
    ind = copula_pdf_of_f_at_fsamples > 1e-5
    BC = np.mean( ratio[ind] )
    hellinger = np.sqrt(1-BC)
    return hellinger

def get_log_copula(g_params, f_samples):
    # g_params=f_params; f_samples=sample1
    F={}
    logf={}
    for colname in f_samples.columns:
        #print(colname)
        data = f_samples[colname]

        if g_params['marginal'][colname]['type']=='gamma':
            F[colname] = scipy.stats.gamma.cdf(data, 
                                                g_params['marginal'][colname]['alpha'], 
                                                g_params['marginal'][colname]['loc'],
                                                g_params['marginal'][colname]['scale'])

            logf[colname] = scipy.stats.gamma.logpdf(data, 
                                                      g_params['marginal'][colname]['alpha'], 
                                                      g_params['marginal'][colname]['loc'],
                                                      g_params['marginal'][colname]['scale'])
        
        if g_params['marginal'][colname]['type']=='beta':
            F[colname] = scipy.stats.beta.cdf(data, 
                                                g_params['marginal'][colname]['alpha'], 
                                                g_params['marginal'][colname]['beta'],
                                                g_params['marginal'][colname]['loc'],
                                                g_params['marginal'][colname]['scale'])

            logf[colname] = scipy.stats.beta.logpdf(data, 
                                                      g_params['marginal'][colname]['alpha'], 
                                                      g_params['marginal'][colname]['beta'],
                                                      g_params['marginal'][colname]['loc'],
                                                      g_params['marginal'][colname]['scale'])

    logf_df = pd.DataFrame.from_dict(logf)
    F_df = pd.DataFrame.from_dict(F)
    
    rv = scipy.stats.multivariate_normal(mean=np.zeros( len(f_samples.columns) ), 
                                          cov=g_params['correlationMatrix'])

    log_copula_pdf = rv.logpdf(F_df) + np.sum( logf_df, axis=1 )

    return log_copula_pdf

def generate_copula_data(marginals_monthly, correlationMatrix_monthly, num_mcmc_samples):
    # correlationMatrix_monthly = cov_nearest(correlationMatrix_monthly_sampleEstimate, threshold=1e-5)   
    colnames = [item for item in marginals_monthly.keys()]
    mcmc_samples = pd.DataFrame(columns = colnames)
    mvnorm = scipy.stats.multivariate_normal(mean=np.zeros(len( marginals_monthly.keys() )), 
                                              cov=correlationMatrix_monthly)
    x = mvnorm.rvs(num_mcmc_samples)
    x_unif = scipy.stats.norm.cdf(x) #this must stay the same always

    k=0
    for colname in colnames : #marginals_monthly.keys():
        #print(colname)
        if marginals_monthly[colname]['type'] == 'gamma':
            m = scipy.stats.gamma(a= marginals_monthly[colname]['alpha'], 
                                  loc= marginals_monthly[colname]['loc'], 
                                  scale= marginals_monthly[colname]['scale'])            
            mcmc_samples[colname] = m.ppf(x_unif[:, k])
            

        if marginals_monthly[colname]['type'] == 'beta':
            m = scipy.stats.beta(a=marginals_monthly[colname]['alpha'], 
                                  b=marginals_monthly[colname]['beta'],
                                  loc= marginals_monthly[colname]['loc'],
                                  scale= marginals_monthly[colname]['scale'])
            mcmc_samples[colname] = m.ppf(x_unif[:, k])
        k=k+1

    return mcmc_samples

def get_marginal_parameters(data_set, noise_metrics_list_16):
    marginal_parameters={}
    for colname in noise_metrics_list_16:
        marginal_parameters[colname]={}
        x=data_set[colname]
        if 'error' in colname:
            #data_set.loc[np.logical_or(x<=np.percentile(x,0.05), x>=np.percentile(x,99.5)), colname] = np.nan
            data_set.loc[np.logical_or(x<=0, x>=1), colname] = np.nan
            #data_set.loc[np.logical_or(x<=np.percentile(x,1), x>=np.percentile(x,99)), colname] = np.nan
        fit_data = data_set[colname]
        fit_data = fit_data[~np.isnan(fit_data)]
    
        if 'T2' in colname:
            (fit_alpha, fit_loc, fit_scale) = scipy.stats.gamma.fit(fit_data, floc=0) #scale is 1/beta            
            marginal_parameters[colname] = {'type': 'gamma', 
                                            'alpha': fit_alpha, 
                                            'loc': fit_loc, 
                                            'scale': fit_scale}           
        else:
            (fit_alpha, fit_beta, fit_loc, fit_scale) = scipy.stats.beta.fit(fit_data, floc=0, fscale=1)
            marginal_parameters[colname] = {'type': 'beta', 
                                            'alpha': fit_alpha, 
                                            'beta': fit_beta,
                                            'loc': fit_loc, 
                                            'scale': fit_scale}    
    return marginal_parameters

def get_data_set(filename, noise_metrics_list_16):
    df = pd.read_csv(filename)
    
    dates = df['query_date'].to_numpy().ravel()
    
    df['time_label_datetime'] = pd.to_datetime(df['query_date'].str.strip('+'), format='%Y - %m - %d')
    df = df.sort_values(by='time_label_datetime').reset_index(drop=True)
    
    # create the csv subset
    header_list = ['query_date', 'last_update_date', 'time_label_datetime']
    # 5 SPAM errors
    for i in range(5):
        header_list.append( 'q_'+str(i)+'__readout_error_value')
    
    # 2 CNOT errors
    for item in ['cx0_1_gate_error_value', 'cx1_0_gate_error_value', 'cx1_2_gate_error_value','cx2_1_gate_error_value']:
        header_list.append(item)
    
    # 4 Hadamard errors (maps to = 1-(1-rz0_gate_error_value)*(1-sx0_gate_error_value) [0,4,1,2])
    # RZ errors
    for i in range(5):
        header_list.append( 'rz'+str(i)+'_gate_error_value')
        header_list.append( 'sx'+str(i)+'_gate_error_value')
    
    # 5 T2 times (maps to = )
    for i in range(5):
        header_list.append( 'q_'+str(i)+'__T2_value')
    
    for header in header_list:
        if header not in df.columns:
            print(header)
    for i in range(5):
        key = 'hadamard_'+str(i)+'_gate_error_value'
        val = 1 - ( 1 - df[ 'rz'+str(i)+'_gate_error_value'] )**2 * ( 1-df['sx'+str(i)+'_gate_error_value'] )
        df[key] = val
        header_list.append(key)
       
    data_set = df[noise_metrics_list_16].copy(deep=True)
    for colname in noise_metrics_list_16:
        x=data_set[colname]
        if 'error' in colname:
            #data_set.loc[np.logical_or(x<=np.percentile(x,0.05), x>=np.percentile(x,99.5)), colname] = np.nan
            data_set.loc[np.logical_or(x<=0, x>=1), colname] = np.nan
    return data_set

def latex_19col_H_comparison(hellinger, ):
    numOfMonths = len( hellinger )
    latex_table = {}
    for j in range(numOfMonths):
        latex_table[ "x_" + str(j)] = [round(float(hellinger_for_each_param[i][j]),2)  for i in range(numOfMonths)]
    latex_table["hellinger_normalized"] = [round(float(item),2) for item in hellinger_normalized]
    latex_table["hellinger_avg"]        = [round(float(item),2) for item in hellinger_averaged]
    latex_table["hellinger"]            = [round(float(item),6) for item in hellinger]

    cols = ["x_" + str(i) for i in range(numOfMonths)]
    cols.append('hellinger_normalized')
    cols.append('hellinger_avg')
    cols.append('hellinger')

    rows  = []
    for j in range(numOfMonths):
        s=month_dict[j] + "&"
        for col in cols:
            s+= str( latex_table[col][j] ) + "&"
        s+="\\"
        rows.append(s)
    print(rows)

# def get_hellinger_attribution(parameters):
#     monthly_distance = np.zeros(len( data_set_monthly ))
#     h4each_param = {}
#     h4each_param[0]=np.zeros(16) 
#     for i in range(1,len( data_set_monthly )):
#         print(i)
#         log_1_minus_hsq_tot = []
#         for colname in noise_metrics_list_16:
#             #print(colname)
#             dist_type = parameters[0]['marginal'][colname]['type']
#             #print(dist_type)
        
#             if dist_type == 'beta':
#                 alpha1 = parameters[0]['marginal'][colname]['alpha']
#                 beta1 = parameters[0]['marginal'][colname]['beta']
#                 loc1 = parameters[0]['marginal'][colname]['loc']
#                 scale1 = parameters[0]['marginal'][colname]['scale']
            
#                 alpha2 = parameters[i]['marginal'][colname]['alpha']
#                 beta2 = parameters[i]['marginal'][colname]['beta']
#                 loc2 = parameters[i]['marginal'][colname]['loc']
#                 scale2 = parameters[i]['marginal'][colname]['scale']
            
#                 _, log_1_minus_hsq_1d = get_beta_hellinger(alpha1, beta1, loc1, scale1, 
#                                                            alpha2, beta2, loc2, scale2)
            
#             if dist_type == 'gamma':
#                 alpha1 = parameters[0]['marginal'][colname]['alpha']
#                 loc1 = parameters[0]['marginal'][colname]['loc']
#                 scale1 = parameters[0]['marginal'][colname]['scale']
            
#                 alpha2 = parameters[i]['marginal'][colname]['alpha']
#                 loc2 = parameters[i]['marginal'][colname]['loc']
#                 scale2 = parameters[i]['marginal'][colname]['scale']
            
#                 _, log_1_minus_hsq_1d = get_gamma_hellinger(alpha1, loc1, scale1, 
#                                                             alpha2, loc2, scale2)       
       
#             log_1_minus_hsq_tot.append( log_1_minus_hsq_1d )
#             h4each_param[i] = np.array( np.sqrt(1-np.exp(log_1_minus_hsq_tot)) )
#         #print( i, np.mean(h4each_param ))
#         monthly_distance[i] = np.mean( h4each_param[i] )
    
#     # attribution (this is dec21 vs dec22)
#     total_amount = np.mean(h4each_param[12])
#     hdata = h4each_param[12]/np.sum(h4each_param[12])*np.mean(h4each_param[12])
#     attribution={}
#     for k in range(len(hdata)):
#         attribution[header_dict[k]]=-hdata[k]
#     attribution=dict(sorted(attribution.items(), key=lambda item: item[1]))
#     ylabel = 'Hellinger distance attribution'
#     figname = 'distance_attribution_ind.png'
#     plot_hellinger_attribution(total_amount, attribution, ylabel, figname)
#     return attribution, h4each_param, monthly_distance

def get_threshold(monthly_correlation_matrix, max_size):
    for threshold in np.arange(.99,.01, -0.01):
        ### get the largest cluster size for one month at this threshold
        # set all values of correlation less than threshold to 0
        max_cluster_size = []
        for i in range(len( data_set_monthly )):
            non_disjoint_clusters = get_non_disjoint_clusters( monthly_correlation_matrix[i], threshold )
            clusters_for_month_i = get_merged_sets(non_disjoint_clusters)
            max_cluster_size.append( np.max( [len(item) for item in clusters_for_month_i] ) )
        
        # now get the largest cluster size amongst all the months at this threshold
        max_cluster_size_across_months = np.max(max_cluster_size)
        # when you hit max size break out and add 0.01: this is your threshold_for_low_correlation
        print(threshold, max_cluster_size_across_months)
        if max_cluster_size_across_months > max_size:
            threshold_for_low_correlation = threshold + 0.01
            break
        
    # rerun at this threshold_for_low_correlation and get cluster_list
    cluster_list={}
    for i in range(len( data_set_monthly )):
        non_disjoint_clusters = get_non_disjoint_clusters( monthly_correlation_matrix[i], threshold_for_low_correlation )
        clusters_for_month_i = get_merged_sets(non_disjoint_clusters)
        cluster_list[i] = clusters_for_month_i
        
    return threshold_for_low_correlation, cluster_list

def get_non_disjoint_clusters( sanple_correlation_matrix, threshold ):
    non_disjoint_clusters=[]
    for counter in range(16):
        non_disjoint_clusters.append( set( [item for item in sanple_correlation_matrix.columns 
                                             if np.abs( sanple_correlation_matrix.iloc[counter,:][item] ) > threshold ] ))
    return non_disjoint_clusters

def get_merged_sets(non_disjoint_clusters):
    # Create a list to hold the merged sets
    clusters_for_month_i = []            
    # Iterate over the sets in the list
    for s in non_disjoint_clusters:
        # Check if the set intersects with any existing merged sets
        intersecting_sets = [existing_set for existing_set in clusters_for_month_i if s & existing_set]
        if intersecting_sets:
            # If there are intersecting sets, merge them with the current set
            merged_set = s.union(*intersecting_sets)
            # Remove the intersecting sets from the list of merged sets
            clusters_for_month_i = [existing_set for existing_set in clusters_for_month_i if not (existing_set & s)]
        else:
            # If there are no intersecting sets, add the current set to the list of merged sets
            merged_set = s
        # Add the merged set to the list of merged sets
        clusters_for_month_i.append(merged_set)
    return clusters_for_month_i

def get_BC_marginals(t):
    temp, logf_df_t = get_log_marginal_pdfs(g_params=parameters[t], f_samples = mcmc_samples[0])
    logRoot_pdf_t = 0.5*logf_df_t 

    temp, logf_df_0 = get_log_marginal_pdfs(g_params=parameters[0], f_samples = mcmc_samples[0])
    logRoot_pdf_t0 = 0.5*logf_df_0

    BC_matrix = np.mean(np.exp(logRoot_pdf_t- logRoot_pdf_t0), axis=0)
    
    return BC_matrix

def get_BC_16d_distribution(t):
    logRootcopulas_at_t0   = get_logRootcopulas_at_t(0)
    logRootcopulas_at_t   = get_logRootcopulas_at_t(t)
    temp1 = logRootcopulas_at_t - logRootcopulas_at_t0

    logRootMarginals_at_t0 = get_logRootMarginals_at_t(0)
    logRootMarginals_at_t = get_logRootMarginals_at_t(t)
    temp2 = logRootMarginals_at_t - logRootMarginals_at_t0

    # good_indices = np.logical_and( np.logical_and(temp1 > -87, temp1 < 87), 
                                      # np.logical_and(temp2 > -87, temp2 < 87))    
    # np.sum(good_indices)
    # temp2 = temp2[good_indices]
    # temp1 = temp1[good_indices]

    rootMarginalRatio = np.exp( temp2)     
    rootCopulaRatio   = np.exp( temp1)    
    # rootCopulaRatio[ np.isinf(rootCopulaRatio) ] = np.nan
    # rootCopulaRatio[ np.logical_or(rootCopulaRatio>1e38, rootCopulaRatio<1e-38) ] = np.nan

    # BC = np.nanmean( rootCopulaRatio * rootMarginalRatio )
    BC = np.mean( rootCopulaRatio * rootMarginalRatio )
    return BC

def get_logRootMarginals_at_t(t):    
    # marginals
    g_params  = parameters[t]
    f_samples = mcmc_samples[0]
    logRootMarginals_at_t, temp = get_log_marginal_pdfs(g_params, f_samples)
    logRootMarginals_at_t = 0.5*logRootMarginals_at_t
    return logRootMarginals_at_t

def get_logRootcopulas_at_t(t):
    logRootcopula_clusterVals_at_t=[]
    for kthCluster in range(len(cluster_list[t])):
        clusterKvars = list(cluster_list[t][kthCluster])
        if len(clusterKvars)==1:
            logRootcopula_clusterVals_at_t.append(np.zeros(num_mcmc_samples)) # if there is only 1 variable in the cluster, then gaussian copula = 1 always
        else:            
            # cluster k parameters 
            parameters_clusterK={}
            parameters_clusterK['marginal']={}
            for key in clusterKvars:
                parameters_clusterK['marginal'][key] = parameters[t]['marginal'][key]
            parameters_clusterK['correlationMatrix'] = parameters[t]['correlationMatrix'].loc[clusterKvars,clusterKvars]
        
            f_samples = mcmc_samples[0][clusterKvars]
            g_params  = parameters_clusterK
        
            logCopula_clusterK_at_t = get_log_gaussian_copula_only(
                g_params  = parameters_clusterK, 
                f_samples = mcmc_samples[0][clusterKvars] # 0 because the samples need to same for num and den
                )
            logRootcopula_clusterK_at_t = 0.5*logCopula_clusterK_at_t
            logRootcopula_clusterVals_at_t.append(logRootcopula_clusterK_at_t)
    
    # sum over the clusters
    logRootcopulas_at_t = np.sum( logRootcopula_clusterVals_at_t, axis=0) # sum over time t clusters
    return logRootcopulas_at_t

def get_log_marginal_pdfs(g_params, f_samples):
    logf={}
    for colname in f_samples.columns:
        #print(colname)
        data = f_samples[colname]

        if g_params['marginal'][colname]['type']=='gamma':
            logf[colname] = scipy.stats.gamma.logpdf(data, 
                                                      g_params['marginal'][colname]['alpha'], 
                                                      g_params['marginal'][colname]['loc'],
                                                      g_params['marginal'][colname]['scale'])
        
        if g_params['marginal'][colname]['type']=='beta':
            logf[colname] = scipy.stats.beta.logpdf(data, 
                                                      g_params['marginal'][colname]['alpha'], 
                                                      g_params['marginal'][colname]['beta'],
                                                      g_params['marginal'][colname]['loc'],
                                                      g_params['marginal'][colname]['scale'])

    logf_df = pd.DataFrame.from_dict(logf)
    logRootMarginals_at_t = 0.5 * np.sum( logf_df, axis=1 )
    return logRootMarginals_at_t, logf_df # scalar, dataframe

def get_log_gaussian_copula_only( g_params, f_samples):
    #g_params  = parameters_clusterK
    # f_samples = mcmc_samples[0][clusterKvars] # 0 because the samples need to same for num and den
    F={}
    if len(f_samples.columns) == 1:
        logCopula_clusterK_at_t = np.zeros(len(f_samples))
    else:
        for colname in f_samples.columns:
            #print(colname)
            data = f_samples[colname]
    
            if g_params['marginal'][colname]['type']=='gamma':
                F[colname] = scipy.stats.gamma.cdf(data, 
                                                    g_params['marginal'][colname]['alpha'], 
                                                    g_params['marginal'][colname]['loc'],
                                                    g_params['marginal'][colname]['scale'])
            
            if g_params['marginal'][colname]['type']=='beta':
                F[colname] = scipy.stats.beta.cdf(data, 
                                                    g_params['marginal'][colname]['alpha'], 
                                                    g_params['marginal'][colname]['beta'],
                                                    g_params['marginal'][colname]['loc'],
                                                    g_params['marginal'][colname]['scale'])
    
        F_df = pd.DataFrame.from_dict(F)
        # correlationMatrix_temp = cov_nearest(g_params['correlationMatrix'], threshold=1e-5) 
        rv = scipy.stats.multivariate_normal(mean=np.zeros( len(f_samples.columns) ), 
                                              cov=g_params['correlationMatrix'])
        logCopula_clusterK_at_t = rv.logpdf(F_df)

    return logCopula_clusterK_at_t # vector over mcmc samples

def plot_distance_from_ref_3kinds(hellinger, hellinger_normalized, hellinger_averaged):
    plt.plot(np.arange(0,len( hellinger)), hellinger, marker='*', linestyle='-.', alpha=0.8, label = 'Hellinger (H)')
    plt.plot(np.arange(0,len( hellinger)), hellinger_normalized, marker='*', linestyle='--', alpha=0.8, label = 'Hellinger normalized ($H_{normalized}$)')
    plt.plot(np.arange(0,len( hellinger)), hellinger_averaged, marker='*', linestyle='-', alpha=0.8, label = 'Hellinger  average ($H_{avg}$)')
    plt.legend()
    ticks = np.arange(0, len( data_set_monthly ))
    plt.gca().set_xticks(ticks)
    labels = ['Jan-22','Feb-22','Mar-22','Apr-22','May-22','Jun-22','Jul-22',
              'Aug-22','Sep-22','Oct-22','Nov-22','Dec-22',
              'Jan-23','Feb-23','Mar-23','Apr-23']
    plt.gca().set_xticklabels(labels, rotation = 45, ha="right")
    sns.despine()
    plt.grid(True, which='both')    
    plt.xlabel(r'Month-Year')
    plt.ylabel('Distance from Jan-2022')
    # plt.savefig('distance_from_dec_2021.png', bbox_inches = "tight", dpi=300)
    plt.savefig('distance_from_ref.png', bbox_inches = "tight", dpi=300)

def get_mean_and_var_of_observable(n, alpha_0, beta_0, k, t, method='analytical', size=4_000):
    if method=='mcmc':    
        a=alpha_0/(k+t)
        b=beta_0/(k+t)
        e_samples = scipy.stats.beta.rvs(a,b, size=size)
        mean_obs_ref = np.mean( (1-e_samples/2)**n )
        var_obs_ref  = np.var( (1-e_samples/2)**n )

    if method=='analytical':    
        mean_obs_ref = 0
        for k1 in np.arange(0, n+1):
            nChooseK1 = math.comb(n,k1)
            mean_obs_ref += nChooseK1 * (-1/2)**k1 * scipy.special.beta( a+k1,b )/scipy.special.beta( a,b )
    
        var_obs_ref = 0
        for k1 in np.arange(0, 2*n+1):
            nChooseK1 = math.comb(2*n,k1)
            var_obs_ref += nChooseK1 * (-1/2)**k1 * scipy.special.beta( a+k1,b )/scipy.special.beta( a,b )
    
        var_obs_ref = var_obs_ref-mean_obs_ref**2

    return mean_obs_ref, var_obs_ref 
