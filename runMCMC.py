#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:00:02 2021

@author: Michi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:52:04 2021

@author: Michi
"""


import time
import os
#import h5py
import sys
#import array as arr
import emcee
import corner
from multiprocessing import Pool, cpu_count

#from scipy.stats import norm as norm1
import matplotlib.pyplot as plt
import numpy as np
#from astropy.cosmology import FlatLambdaCDM
#from astropy.cosmology import Planck15
#import astropy.units as u
#from scipy.optimize import fsolve
#import multiprocessing as multi
#import scipy.integrate as si
#from scipy.integrate import cumtrapz
#from scipy.interpolate import interp1d, RectBivariateSpline
#import scipy.stats as ss
#import seaborn as sns
#from tqdm import tqdm, tqdm_notebook
#from scipy.integrate import quad
#import h5py
#import scipy.integrate as si
#from scipy.integrate import cumtrapz
#from scipy.interpolate import interp1d, RectBivariateSpline
#import scipy.stats as ss
#import seaborn as sns
#import sys
#from tqdm import tqdm, tqdm_notebook
#from scipy.stats import loguniform
import shutil


#from data import *
#from config import *
#from models import *
#from params import PriorLimits
#from glob import *
from utils import Logger
import config
import models
import Globals


os.environ["OMP_NUM_THREADS"] = "1"


###############################################
#  GLOBAL VARIABLES
###############################################



# Nobs=100
#allMyPriors = PriorLimits()
#allMyPriors.set_priors(priors_types=config.priors_types, priors_params=config.priors_params)





labels_param = [ config.myParams.names[param] for param in config.params_inference ] #.sort()
Lambda_ntest = np.array([config.myParams.trueValues[param] for param in config.params_n_inference])


exp_values= np.array(config.myParams.get_expected_values(config.params_inference)) #[70, 1, 45]
eps = [val if val!=0 else 1 for val in exp_values ]

lowLims = [ max( [val-eps[i]*config.perc_variation_init/100, config.myPriorLimits[i][0] ] )  for i,val in enumerate(exp_values) ] 
upLims = [ min( [ val+eps[i]*config.perc_variation_init/100, config.myPriorLimits[i][1] ]) for i,val in enumerate(exp_values) ] 
#upLims[exp_values==0]=perc_variation_init/100
for i in range(len(lowLims)):
    if lowLims[i]>upLims[i]:
        lowLim=upLims[i]
        upLim=lowLims[i]
        upLims[i]=upLim
        lowLims[i]=lowLim
        
print('lowLims: %s' %lowLims)
print('upLims: %s' %upLims)
Delta = [upLims[i]-lowLims[i] for i in range(len(exp_values))] #[140-20, 10-0, 150-20] 

print('Delta: %s' %Delta)

labels_param= config.myParams.get_labels(config.params_inference)#[r"$H_0$", r"$\Xi_0$", r"$m_h$"]
trueValues = config.myParams.get_true_values(config.params_inference)#[67.74, 1, 45]




########################


def main():
    
    in_time=time.time()
    
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--observingRun", default='O2', type=str, required=True)
    #parser.add_argument("--wf_model", default='', type=str, required=True)
    #FLAGS = parser.parse_args()
    

    
    #####
    # Out file
    ####
    
    # St out path and create out directory
    out_path=os.path.join(Globals.dirName, 'results', config.fout)
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    else:
       print('Using directory %s for output' %out_path)
       
       
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog
    
    shutil.copy('config.py', os.path.join(out_path, 'config_original.py'))
    
    #####
    # Load data
    #####
    
    

    #####
    # Setup MCMC
    #####

    ndim = len(Delta)
    
    if  (('R0' in config.params_inference) and (config.marginalise_rate) ):
        raise ValueError('You cannot marfinalise on R0 and run inference on R0 at the same time. Change marginalise_rate to False if running inference on R0.' )
    
    print('Running inference for parameters: %s' %str(config.params_inference))
    print('Prior range: %s' %config.myPriorLimits)
    print('Fixing parameters: %s' %str(config.params_n_inference))
    print('Values: %s' %str(Lambda_ntest))
    if config.marginalise_rate:
        print('Marginalising over total rate R0')
    
     
    #print('Initial balls for initialization of the walkers: %s' %str(Delta))
    print(' Initial intervals for initialization of the walkers have an amplitude of +-%s percent around the expeced values of %s'%(config.perc_variation_init, str(exp_values)) )
    pos = Delta*np.random.rand(config.nChains,  ndim)+lowLims
    nwalkers = pos.shape[0]
    #print('Initial positions of the walkers: %s' %str(pos))
    
    #scriptname = __file__
    #filenameT = scriptname.replace("_", "\_")
    #filenameT = scriptname
    #if telegramAlert:
    #    telegram_bot_sendtext("Start of: %s" %filenameT)
    #    telegram_bot_sendtext('%s: nwalkers = %d, ndim = %d' %(filenameT,nwalkers, ndim) )
    #    telegram_bot_sendtext("%s: labels =%s" %(filenameT,labels_param))
    #    telegram_bot_sendtext("%s: max step =%s" %(filenameT,max_n))

    print('nwalkers=%s, ndim=%s' %(nwalkers, ndim))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "chains.h5"
    backend = emcee.backends.HDFBackend(os.path.join(out_path,filename))
    backend.reset(nwalkers, ndim)

    print('starting MCMC. Max number of steps: %s' %config.max_n)

    # Initialize the sampler
    ncpu = cpu_count()
    print('Parallelizing on %s CPUs ' %config.nPools)
    print('Number of availabel cores:  %s ' %ncpu)
    
    #if marginalise_rate:
    #    logpost_function  = log_posterior_marg
    #else:
    #    logpost_function  = log_posterior_unmarg
    
    
    
    with Pool(config.nPools) as pool:
    	
        sampler = emcee.EnsembleSampler(nwalkers, ndim, models.log_posterior, backend=backend, args=(Lambda_ntest, config.myPriorLimits, config.params_inference , config.allMyPriors.pnames, config.allMyPriors.prior_params ), pool=pool)
        #sampler.run_mcmc( pos, max_n, progress=True)  
        
        
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(config.max_n)
        
        old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(pos, iterations=config.max_n, progress=True, skip_initial_state_check=False):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
            else:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index +=1
                print('Step n %s. Check autocorrelation: ' %(index*100))
                print(tau)
        
                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    print('Chain has converged. Stopping. ')
                    break
                else:
                    print('Chain has not converged yet. ')
                    old_tau = tau
    
    print('Plotting chains... ')
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = labels_param
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");  
    
    plt.savefig(os.path.join(out_path,'chains.pdf'))  

    tau = np.zeros(ndim) #sampler.get_autocorr_time()
    burnin = int(4 * np.max(tau)) # I try with 4 times instead of 2
    thin = 1#int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    #flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    
    print('Plotting cornerplot... ')
    fig1 = corner.corner(
    samples, labels=labels, truths=trueValues,quantiles=[0.16, 0.84],show_titles=True, title_kwargs={"fontsize": 12}
    );

    fig1.savefig(os.path.join(out_path, 'corner.pdf'))
    #if telegramAlert:
    #    telegram_bot_sendtext("End of: %s " %filenameT)
    
    
    
    
    ######
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close() 




#######################################################

if __name__=='__main__':
    main()

    
