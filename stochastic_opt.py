import math
import numpy as np
import scipy.stats as sct
from scipy.optimize import minimize

from scipy.stats import norm
from scipy.stats import bernoulli
from scipy.stats import geom

from matplotlib import pyplot as plt

def u(x):
    return -1*np.log(max(0,(x[0]*F1 + x[1]*S1 - (x[1]**2)*rho*S1)))

def opt(N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b):
    obj, opt = np.zeros(pow(2,N+n0+1)), np.zeros(pow(2,N+n0+1))
        
    J = bernoulli.rvs(p, size=pow(2,N+n0+1))
    Z = norm.rvs(0, 1, size=pow(2,N+n0+1))
    R = np.exp((mu0 + Z*sigma0*J) + (mu1 + Z*sigma1*(1-J)))
    
    F0 = 1
    S1 = S0*R
    F1 = np.tile(np.exp(r)*F0,pow(2,N+n0+1))
    
    for i in range(len(obj)):
        def u(x):
            return -1*np.log(max(0,(x[0]*F1[i] + x[1]*S1[i] - (x[1]**2)*rho*S1[i])))
        def cons1(x):
            return 1*(x[0]*F0 + x[1]*S0 + (x[1]**2)*rho*S0 - b)
        cons = ({'type':'eq', 'fun': cons1})
        bnds = ((.000001, np.inf), (.000001, np.inf))
    
        res = minimize(u, [0,0], method='SLSQP', bounds = bnds, constraints = cons)
        obj[i] = -u(res.x)
        opt[i] = res.x[1]
    return obj, opt

def Q2(S0, p, mu0, sigma0, mu1, sigma1, rho, r, b, gamma, delta, n0, N0):
    burn_in = n0
    num_sample = N0
    z = sct.norm.ppf(1 - delta/2) # 1 - delta/2 quantile of N(0, 1)
    r_star = 1 - pow(2, -1.5) # optimal success rate for the geometric of N

    obj_confidence_interval = float('inf')
    opt_confidence_interval = float('inf')
    obj_running_mean = 0
    opt_running_mean = 0
    obj_running_2moment = 0
    opt_running_2moment = 0

    num_estimator = 0 #count of number of estimators generated
    CIs1 = np.zeros((1, num_sample))
    CIs2 = np.zeros((1, num_sample))
    obj_estimation = np.zeros((1, num_sample))
    opt_estimation = np.zeros((1, num_sample))

    while (num_estimator < num_sample or (obj_confidence_interval >= delta and opt_confidence_interval >= delta)):
        N = np.random.geometric(p=r_star)
        obj_samples, opt_samples = opt(N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b)
    
        obj_samples_odd = obj_samples[0::2]
        obj_samples_even = obj_samples[1::2]
        obj_samples_n_0 = obj_samples[0:pow(2,n0)]
    
        obj_theta_N = np.mean(obj_samples)
        obj_theta_N_odd = np.mean(obj_samples_odd)
        obj_theta_N_even = np.mean(obj_samples_even)
        obj_theta_n_0 = np.mean(obj_samples_n_0)
    
        opt_samples_odd = opt_samples[0::2]
        opt_samples_even = opt_samples[1::2]
        opt_samples_n_0 = opt_samples[0:pow(2,n0)]
    
        opt_theta_N = np.mean(opt_samples)
        opt_theta_N_odd = np.mean(opt_samples_odd)
        opt_theta_N_even = np.mean(opt_samples_even)
        opt_theta_n_0 = np.mean(opt_samples_n_0)
    
        obj_X_star = (obj_theta_N - (obj_theta_N_odd + obj_theta_N_even) / 2) / sct.geom(r_star).pmf(N+1) + obj_theta_n_0
        opt_X_star = (opt_theta_N - (opt_theta_N_odd + opt_theta_N_even) / 2) / sct.geom(r_star).pmf(N+1) + opt_theta_n_0
    
        obj_running_mean = (obj_running_mean * num_estimator + obj_X_star) / (num_estimator + 1)
        opt_running_mean = (opt_running_mean * num_estimator + opt_X_star) / (num_estimator + 1)
    
        obj_running_2moment = (obj_running_2moment * num_estimator + pow(obj_X_star, 2)) / (num_estimator + 1)
        opt_running_2moment = (opt_running_2moment * num_estimator + pow(opt_X_star, 2)) / (num_estimator + 1)
    
        obj_sample_std = math.sqrt(obj_running_2moment - pow(obj_running_mean, 2))
        opt_sample_std = math.sqrt(opt_running_2moment - pow(opt_running_mean, 2))
    
        num_estimator = num_estimator + 1
    
        obj_confidence_interval = z * obj_sample_std / (math.sqrt(num_estimator))
        opt_confidence_interval = z * opt_sample_std / (math.sqrt(num_estimator))
    
        obj_estimation[:,num_estimator-1] = obj_running_mean
        opt_estimation[:,num_estimator-1] = opt_running_mean
    
        CIs1[:,num_estimator-1] = obj_confidence_interval
        CIs2[:,num_estimator-1] = opt_confidence_interval
    
    obj_lower = obj_estimation - CIs1
    opt_lower = opt_estimation - CIs2
    
    obj_upper = obj_estimation + CIs1
    opt_upper = opt_estimation + CIs2
    
    print('Generate', num_estimator , 'samples \n')
    
    fig, (ax1, ax2) = plt.subplots(2)
                                                                        
    n_range = np.arange(burn_in-1, num_sample)
    ax1.plot(n_range, obj_estimation[0,n_range], label='objective function estimation')
    ax1.plot(n_range, obj_lower[0,n_range], label='lower CI')
    ax1.plot(n_range, obj_upper[0,n_range], label='upper CI')
    
    ax2.plot(n_range, opt_estimation[0,n_range], label='optimal parameter estimation')
    ax2.plot(n_range, opt_lower[0,n_range], label='lower CI')
    ax2.plot(n_range, opt_upper[0,n_range], label='upper CI')
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    plt.show()
    return obj_running_mean, obj_confidence_interval, opt_running_mean, opt_confidence_interval

def sampler(riskless, risky, N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b):
    obj = np.zeros(pow(2,N+n0+1))
        
    J = bernoulli.rvs(p, size=pow(2,N+n0+1))
    Z = norm.rvs(0, 1, size=pow(2,N+n0+1))
    R = np.exp((mu0 + Z*sigma0*J) + (mu1 + Z*sigma1*(1-J)))
    
    F0 = 1
    S1 = S0*R
    F1 = np.tile(np.exp(r)*F0,pow(2,N+n0+1))
    
    for i in range(len(obj)):
        obj[i] = max(0,b-(riskless*F1[i] + risky*S1[i] - (risky**2)*rho*S1[i]))
    
    return obj

def other_Q2(S0, p, mu0, sigma0, mu1, sigma1, rho, r, b, gamma, delta, n0, N0):

    burn_in = n0
    num_sample = N0
    
    z = sct.norm.ppf(1 - delta/2) # 1 - delta/2 quantile of N(0, 1)
    r_star = 1 - pow(2, -1.5) # optimal success rate for the geometric of N
    
    confidence_interval = float('inf')
    running_mean = 0
    running_2moment = 0
    num_estimator = 0 #count of number of estimators generated
    
    CIs = np.zeros((1, num_sample))
    estimation = np.zeros((1, num_sample))
    
    while (num_estimator < num_sample or confidence_interval >= delta):
        N = np.random.geometric(p=r_star)
        samples = sampler(riskless, risky, N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b)
        samples_odd = samples[0::2]
        samples_even = samples[1::2]
        samples_n_0 = samples[0:pow(2,n0)]
        
        theta_N = np.mean(samples)
        theta_N_odd = np.mean(samples_odd)
        theta_N_even = np.mean(samples_even)
        theta_n_0 = np.mean(samples_n_0)
        
        X_star = (theta_N - (theta_N_odd + theta_N_even) / 2) / sct.geom(r_star).pmf(N+1) + theta_n_0
        running_mean = (running_mean * num_estimator + X_star) / (num_estimator + 1)
        running_2moment = (running_2moment * num_estimator + pow(X_star, 2)) / (num_estimator + 1)
       
        sample_std = math.sqrt(running_2moment - pow(running_mean, 2))
        num_estimator = num_estimator + 1
        confidence_interval = z * sample_std / (math.sqrt(num_estimator))
        estimation[:,num_estimator-1] = running_mean
        CIs[:,num_estimator-1] = confidence_interval
        
    lower = estimation - CIs
    upper = estimation + CIs
    print('Generate', num_estimator , 'samples \n')
    
    n_range = np.arange(burn_in-1, num_sample)
    plt.plot(n_range, estimation[0,n_range], label='estimation')
    plt.plot(n_range, lower[0,n_range], label='lower CI')
    plt.plot(n_range, upper[0,n_range], label='upper CI')
    plt.legend(loc='upper right')
    plt.show()
    
    return running_mean, confidence_interval

def Q3_opt(N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b):
    obj, opt, GZ = np.zeros(pow(2,N+n0+1)), np.zeros(pow(2,N+n0+1)), np.zeros(pow(2,N+n0+1))
        
    J = bernoulli.rvs(p, size=pow(2,N+n0+1))
    Z = norm.rvs(0, 1, size=pow(2,N+n0+1))
    R = np.exp((mu0 + Z*sigma0*J) + (mu1 + Z*sigma1*(1-J)))
    
    F0 = 1
    S1 = S0*R
    F1 = np.tile(np.exp(r)*F0,pow(2,N+n0+1))
    
    for i in range(len(obj)):
        def u(x):
            return -1*np.log(max(0,(x[0]*F1[i] + x[1]*S1[i] - (x[1]**2)*rho*S1[i])))
        def cons1(x):
            return 1*(x[0]*F0 + x[1]*S0 + (x[1]**2)*rho*S0 - b)
        cons = ({'type':'eq', 'fun': cons1})
        bnds = ((.000001, np.inf), (.000001, np.inf))
    
        res = minimize(u, [0,0], method='SLSQP', bounds = bnds, constraints = cons)
        obj[i] = -u(res.x)
        opt[i] = res.x[1]
        GZ[i] = res.x[0]
        
    return obj, opt, GZ

def first_CV_Q3(S0, p, mu0, sigma0, mu1, sigma1, rho, r, b, gamma, delta, n0, N0):
    
    EZ = riskless*F0

    def g(Z):
        return Z*F0
    
    burn_in = n0
    num_sample = N0
    z = sct.norm.ppf(1 - delta/2) # 1 - delta/2 quantile of N(0, 1)
    r_star = 1 - pow(2, -1.5) # optimal success rate for the geometric of N

    obj_confidence_interval = float('inf')
    opt_confidence_interval = float('inf')
    obj_running_mean = 0
    opt_running_mean = 0
    obj_running_2moment = 0
    opt_running_2moment = 0

    num_estimator = 0 #count of number of estimators generated
    CIs1 = np.zeros((1, num_sample))
    CIs2 = np.zeros((1, num_sample))
    obj_estimation = np.zeros((1, num_sample))
    opt_estimation = np.zeros((1, num_sample))

    while (num_estimator < num_sample):
        N = np.random.geometric(p=r_star)
        obj_samples, opt_samples, Z = Q3_opt(N, S0, n0, p, mu0, sigma0, mu1, sigma1, rho, r, b)
        
        ######################
        ###CONTROL VARIATES###
        ######################
        
        opt_samples = opt_samples - (opt_beta * (g(Z) - EZ))
    
        obj_samples_odd = obj_samples[0::2]
        obj_samples_even = obj_samples[1::2]
        obj_samples_n_0 = obj_samples[0:pow(2,n0)]
    
        obj_theta_N = np.mean(obj_samples)
        obj_theta_N_odd = np.mean(obj_samples_odd)
        obj_theta_N_even = np.mean(obj_samples_even)
        obj_theta_n_0 = np.mean(obj_samples_n_0)
    
        opt_samples_odd = opt_samples[0::2]
        opt_samples_even = opt_samples[1::2]
        opt_samples_n_0 = opt_samples[0:pow(2,n0)]
    
        opt_theta_N = np.mean(opt_samples)
        opt_theta_N_odd = np.mean(opt_samples_odd)
        opt_theta_N_even = np.mean(opt_samples_even)
        opt_theta_n_0 = np.mean(opt_samples_n_0)
    
        obj_X_star = (obj_theta_N - (obj_theta_N_odd + obj_theta_N_even) / 2) / sct.geom(r_star).pmf(N+1) + obj_theta_n_0
        opt_X_star = (opt_theta_N - (opt_theta_N_odd + opt_theta_N_even) / 2) / sct.geom(r_star).pmf(N+1) + opt_theta_n_0
    
        obj_running_mean = (obj_running_mean * num_estimator + obj_X_star) / (num_estimator + 1)
        opt_running_mean = (opt_running_mean * num_estimator + opt_X_star) / (num_estimator + 1)
    
        obj_running_2moment = (obj_running_2moment * num_estimator + pow(obj_X_star, 2)) / (num_estimator + 1)
        opt_running_2moment = (opt_running_2moment * num_estimator + pow(opt_X_star, 2)) / (num_estimator + 1)
    
        obj_sample_std = math.sqrt(obj_running_2moment - pow(obj_running_mean, 2))
        opt_sample_std = math.sqrt(opt_running_2moment - pow(opt_running_mean, 2))
    
        num_estimator = num_estimator + 1
    
        obj_confidence_interval = z * obj_sample_std / (math.sqrt(num_estimator))
        opt_confidence_interval = z * opt_sample_std / (math.sqrt(num_estimator))
    
        obj_estimation[:,num_estimator-1] = obj_running_mean
        opt_estimation[:,num_estimator-1] = opt_running_mean
    
        CIs1[:,num_estimator-1] = obj_confidence_interval
        CIs2[:,num_estimator-1] = opt_confidence_interval
    
    obj_lower = obj_estimation - CIs1
    opt_lower = opt_estimation - CIs2
    
    obj_upper = obj_estimation + CIs1
    opt_upper = opt_estimation + CIs2
    
    print('Generate', num_estimator , 'samples \n')
    
    fig, (ax1, ax2) = plt.subplots(2)
                                                                        
    n_range = np.arange(burn_in-1, num_sample)
    ax1.plot(n_range, obj_estimation[0,n_range], label='objective function estimation')
    ax1.plot(n_range, obj_lower[0,n_range], label='lower CI')
    ax1.plot(n_range, obj_upper[0,n_range], label='upper CI')
    
    ax2.plot(n_range, opt_estimation[0,n_range], label='optimal parameter estimation')
    ax2.plot(n_range, opt_lower[0,n_range], label='lower CI')
    ax2.plot(n_range, opt_upper[0,n_range], label='upper CI')
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    plt.show()
    return obj_running_mean, obj_confidence_interval, opt_running_mean, opt_confidence_interval

