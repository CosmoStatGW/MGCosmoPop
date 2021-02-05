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
import h5py
import sys
import array as arr
import emcee
import corner
from multiprocessing import Pool, cpu_count

from scipy.stats import norm as norm1
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.optimize import fsolve
import multiprocessing as multi
import scipy.integrate as si
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, RectBivariateSpline
import scipy.stats as ss
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
from scipy.integrate import quad
import h5py
import scipy.integrate as si
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, RectBivariateSpline
import scipy.stats as ss
import seaborn as sns
import sys
from tqdm import tqdm, tqdm_notebook
from scipy.stats import loguniform
import shutil


#from dataFarr import *
from models import *
from params import Params, PriorLimits



os.environ["OMP_NUM_THREADS"] = "1"


###############################################
#  GLOBAL VARIABLES
###############################################

dirName  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
dataPath=os.path.join(dirName, 'data')

fout='run2'

dataset_name = 'mock'

telegramAlert = False

nChains=8
max_n=10000

maxNtaus = 150
checkTauStep = 100
# Nobs=100

myParams = Params(dataset_name)

params_inference = [ 'H0', 'Xi0','mh' ]

params_n_inference = [param for param in myParams.allParams if param not in params_inference]


labels_param = [ myParams.names[param] for param in params_inference ] #.sort()
Lambda_ntest = np.array([myParams.trueValues[param] for param in params_n_inference])


exp_values= myParams.get_expected_values(params_inference) #[70, 1, 45]
perc_variation = 10
lowLims = [val-val*perc_variation/100 for val in exp_values] #[20, 0, 20]
upLims = [val+val*perc_variation/100 for val in exp_values] 
Delta = [upLims[i]-lowLims[i] for i in range(len(exp_values))] #[140-20, 10-0, 150-20] 

labels_param= myParams.get_labels(params_inference)#[r"$H_0$", r"$\Xi_0$", r"$m_h$"]
trueValues = myParams.get_true_values(params_inference)#[67.74, 1, 45]



allMyPriors = PriorLimits()

priorLimits  = [ (allMyPriors.limInf[param],allMyPriors.limSup[param] ) for param in params_inference ]


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
    out_path=os.path.join(dirName, 'results', fout)
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    else:
       print('Using directory %s for output' %out_path)
       
       
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog
    
    shutil.copy('runMCMC.py', os.path.join(out_path, 'runMCMC_original.py'))
    
    #####
    # Load data
    #####
    
    

    #####
    # Setup MCMC
    #####

    ndim = len(Delta)
    
    print('Running inference for parameters: %s' %str(params_inference))
    print('Prior range: %s' %priorLimits)
    print('Fixing parameters: %s' %str(params_n_inference))
    print('Values: %s' %str(Lambda_ntest))
     
    #print('Initial balls for initialization of the walkers: %s' %str(Delta))
    print(' Initial intervals for initialization of the walkers have an amplitude of +-%s percent around the expeced values of %s'%(perc_variation, str(exp_values)) )
    pos = Delta*np.random.rand(nChains,  ndim)+lowLims
    nwalkers = pos.shape[0]
    print('Initial positions of the walkers: %s' %str(pos))
    
    scriptname = __file__
    filenameT = scriptname.replace("_", "\_")
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

    print('starting MCMC. Max number of steps: %s' %max_n)

    # Initialize the sampler
    ncpu = cpu_count()
    print('Parallelizing on %s CPUs ' %ncpu)
    with Pool() as pool:
    	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, backend=backend, args=(Lambda_ntest,priorLimits), pool=pool)
    	
    # We'll track how the average autocorrelation time estimate changes
    	index = 0
    	autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    	old_tau = np.inf

        # Now we'll sample for up to max_n steps
    	for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        	if sampler.iteration % 100:
            	   continue
            
            #if telegramAlert:
            # 	telegram_bot_sendtext("%s: step No.  %s" %(filenameT,sampler.iteration))
        
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
        	tau = sampler.get_autocorr_time(tol=0)
        	autocorr[index] = np.mean(tau)
        	index += 1
 

    
        
        # Check convergence
        	converged = np.all(tau * 100 < sampler.iteration)
        	converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        	if converged:
         	   break
        	old_tau = tau
        
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

    

    fig1 = corner.corner(
    samples, labels=labels, truths=trueValues,quantiles=[0.16, 0.84],show_titles=True, title_kwargs={"fontsize": 12}
    );

    fig1.savefig(os.path.join(out_path, 'corner.pdf'))
    #if telegramAlert:
    #    telegram_bot_sendtext("End of: %s " %filenameT)
    
    
    # Example: save plot 
    #plt.plot(a,b )
    #plt.savefig(os.path.join(out_path,'my_plot.png'))
    #plt.close();
    
    
    ## Save corner plot
    # fig1 = corner.corner( flat_samples, labels=labels,  #truths=[]);
    # fig1.savefig(os.path.join(out_path, 'corner.pdf'))
    
    
    ######
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close() 




#######################################################

if __name__=='__main__':
    main()

    
