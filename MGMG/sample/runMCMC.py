#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:55:07 2021

@author: Michi
"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import argparse   
import importlib
import shutil
import time

from posteriors.prior import Prior
from posteriors.likelihood import HyperLikelihood
from posteriors.selectionBias import SelectionBiasInjections
from posteriors.posterior import Posterior

import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
import astropy.units as u

import Globals
import utils

from sample.models import build_model, load_data, setup_chain



units = {'Mpc': u.Mpc, 'Gpc': u.Gpc}


params_O3 = {   'R0': 24. ,  # Gpc^-3 yr^-1 
                        #'Xi0': 1. , 
                        #'n' : 0.5,  
                'lambdaRedshift':2.7,
                'mh':87.,
                'ml':2., 
                'beta'  :1.4                                    
    }



def notify_start(telegram_id, bot_token, filenameT, fname, nwalkers, ndim, params_inference, max_steps):
    #scriptname = __file__
    #filenameT = scriptname.replace("_", "\_")
    #filenameT = scriptname
    text=[]
    text.append( "Start of: %s" %filenameT )
    text.append( '%s: nwalkers = %d, ndim = %d'%(filenameT,nwalkers, ndim) )
    text.append( "%s: parameters :%s" %(filenameT,params_inference))
    text.append( "%s: max step =%s" %(filenameT,max_steps))
        
    utils.telegram_bot_sendtext( "\n".join(text), telegram_id, bot_token)
 

def find_duplicates(a):
    seen = {}
    dupes = []

    for x in a:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
                seen[x] += 1
    return dupes


def check_params(params_inference, params_fixed, params_all):
    
    allparams = params_inference+list(params_fixed.keys())
    dupes =  find_duplicates(allparams)
    if dupes:
        raise ValueError('The following parameters are included both in params_inference and parametes_fixed. Check your choices! \n%s' %str(dupes))
    



def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='', type=str, required=True) # config file
    parser.add_argument("--fout", default='', type=str, required=True) # output folder 
    FLAGS = parser.parse_args()
    
    config = importlib.import_module(FLAGS.config, package=None)
    
    fout = FLAGS.fout
    
    out_path=os.path.join(Globals.dirName, 'results', fout)
    
    out_path=os.path.join(Globals.dirName, 'results', fout)
    #if not os.path.exists(out_path):
    try:
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    #else:
    except FileExistsError:
        print('Using directory %s for output' %out_path)
    
    shutil.copy(FLAGS.config+'.py', os.path.join(out_path, 'config_original.py'))
    
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = utils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog
        
    if config.telegram_notifications:
        scriptname = __file__
        filenameT = scriptname.replace("_", "\_")
        filenameT = scriptname+'(run %s)'%fout
    
    #ncpu = cpu_count()
    #print('Parallelizing on %s CPUs ' %config.nPools)
    #print('Number of availabel cores:  %s ' %ncpu)
    
    ndim = len(config.params_inference)
    
   ############################################################## 
   ##############################################################
    
    def run():
        ##############################################################
        # POPULATION MODELS
        print('\nCreating populations...')
    
        # Pass correct unit to rate nosmalization
        rate_args=config.rate_args
        rate_args['unit'] = units[config.dist_unit]
    
        allPops = build_model( config.populations, 
                              mass_args=config.mass_args, 
                              spin_args=config.spin_args, 
                              rate_args=rate_args)
        
        # Check that params inference and not inference are correctly specified
        # (No duplicates)
        check_params(config.params_inference, config.params_fixed, allPops.params )
        
        
        # If using O3 data, set base values to expected ones from LVC
        if config.dataset_name=='O3a':
            allPops.set_values( params_O3)
        
        # Fix values of the parameters not included in the MCMC
        allPops.set_values( config.params_fixed)
        
        # Fix values of the parameters not included in the MCMC
        allPops.set_values( config.params_fixed)
        
        
        
        # Check that the order of arguments for MCMC in the config match the order
        # of arguments in population
        allPops.check_params_order(config.params_inference)
        
        ############################################################
        # DATA
        
        if config.dataset_name=='O3a':
            O3_use=config.O3_use
        elif config.dataset_name=='mock':
            O3_use=None
        
        print('\nLoading data from %s catalogue...' %config.dataset_name) 
        Data, injData = load_data(config.dataset_name, nObsUse=config.nObsUse, nSamplesUse=config.nSamplesUse, nInjUse=config.nInjUse, dist_unit=units[config.dist_unit], events_use=O3_use, )
        
        
        ############################################################
        # STATISTICAL MODEL
        
        print('\nSetting up inference for %s ' %config.params_inference)
        
        
        print('Fixed parameters and their values: ')
        print(allPops.get_fixed_values(config.params_inference))
        
        myPrior = Prior(config.priorLimits, config.params_inference, config.priorNames, config.priorParams)
        
        myLik = HyperLikelihood(allPops, Data, config.params_inference )
        
        selBias = SelectionBiasInjections( allPops, injData, config.params_inference, get_uncertainty=config.include_sel_uncertainty )
        
        myPost = Posterior(myLik, myPrior, selBias)
    

        ############################################################
        # SETUP MCMC
        
        # Initial position of the walkers
        exp_values = allPops.get_base_values(config.params_inference)
        pos = setup_chain(config.nwalkers, exp_values, config.priorNames, config.priorLimits, config.priorParams, config.params_inference, perc_variation_init=config.perc_variation_init, seed=config.seed)
        print('nwalkers=%s, ndim=%s' %(pos.shape[0], pos.shape[1]))
        #if (ndim<5) & (config.nwalkers<5):
        #print('Initial positions of the walkers: %s' %str(pos))
        
    
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = "chains.h5"
        backend = emcee.backends.HDFBackend(os.path.join(out_path,filename))
        backend.reset(config.nwalkers, ndim)
    
        autocorr_fname = os.path.join(out_path, "autocorr.txt")
    
        if config.telegram_notifications:
            notify_start(config.telegram_id, config.telegram_bot_token, filenameT, fout, config.nwalkers, ndim, config.params_inference, config.max_steps)
    
    
    
        ############################################################
        # RUN MCMC
    
    
        sampler = emcee.EnsembleSampler(config.nwalkers, ndim, myPost.logPosterior, backend=backend, pool=pool)
    
    
        # Track autocorrelation time
        index = 0
        autocorr = np.empty(config.max_steps)
        
        old_tau = np.inf

        # Sample
        for sample in sampler.sample(pos, iterations=config.max_steps, progress=True, skip_initial_state_check=False):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
            else:
                tau = sampler.get_autocorr_time(tol=0) # tol=0 is in irder to continue the chain
                if np.any(np.isnan(tau)):
                    burnin=0
                    converged=False
                else:
                    burnin = int(2 * np.max(tau))
                    # Check convergence
                    converged = np.all(tau * config.convergence_ntaus < sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < config.convergence_percVariation)
                    
                autocorr[index] = np.mean(tau)
                index +=1
                print('Step n %s. Check autocorrelation: ' %(index*100))
                print(tau)
                np.savetxt(autocorr_fname, autocorr[:index])
        
                
                
                if config.telegram_notifications:
                    N=len(sampler.get_chain())
                    utils.telegram_bot_sendtext( "%s: step No.  %s, converged=%s, N/%s=%s, burnin=%s" %(filenameT, sampler.iteration, converged, config.convergence_ntaus, N/config.convergence_ntaus, burnin ), config.telegram_id, config.telegram_bot_token)
                
                if converged:
                    print('Chain has converged. Stopping. ')
                    break
                else:
                    print('Chain has not converged yet. ')
                    if np.any(np.isnan(tau)):
                        old_tau = np.inf
                    else:
                        old_tau = tau
        return allPops, sampler, index, autocorr
    
    ##############################################################
    ##############################################################
    
    if config.parallelization == 'mpi':
        from schwimmbad import MPIPool
        myPool=MPIPool()
    elif config.parallelization == 'pool':
        from multiprocessing import Pool, cpu_count
        ncpu = cpu_count()
        print('Number of availabel cores:  %s ' %ncpu)
        npools = min(config.nPools,ncpu )
        print('Parallelizing on %s CPUs ' %npools)
        myPool=Pool(npools)
    
    with myPool as pool:
        if config.parallelization == 'mpi':
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            allPops, sampler, index, autocorr  = run()
        else:
            allPops, sampler,  index, autocorr = run()
    
              
    
    
    
    ############################################################
    # SUMMARY PLOTS           
    
    print('Plotting chains... ')
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = allPops.get_labels(config.params_inference)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");  
    
    plt.savefig(os.path.join(out_path,'chains.pdf'))  

    tau = sampler.get_autocorr_time(quiet=True)
    if np.any(np.isnan(tau)):
        burnin=0
        thin=1
    else:
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    plt.close()

    
    print('Plotting cornerplot... ')
    trueValues=None
    if config.dataset_name=='mock':
        trueValues = allPops.get_base_values(config.params_inference)
    
    fig1 = corner.corner(
    samples, labels=labels, truths=trueValues, quantiles=[0.16, 0.84],show_titles=True, title_kwargs={"fontsize": 12}
    );

    fig1.savefig(os.path.join(out_path, 'corner.pdf'))
    plt.close()
    
    if index>2:
        print('Plotting autocorrelation... ')
        n = 100 * np.arange(1, index + 1)
        y = autocorr[:index]
        plt.plot(n, n / config.convergence_ntaus, "--k", label='N/%s'%config.convergence_ntaus )
        plt.plot(n, y)
        plt.xlim(n.min(), n.max())
        plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$");
        plt.legend();
        plt.savefig(os.path.join(out_path,'autocorr.pdf'))
    
    ############################################################
    # END
    
    if config.telegram_notifications:
        utils.telegram_bot_sendtext("End of %s. Total execution time: %.2fs" %(filenameT,  (time.time() - in_time)) , config.telegram_id, config.telegram_bot_token)
    
    print('\nDone. Total execution time: %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close() 
    
    if config.parallelization == 'mpi':
        pool.close()
        sys.exit(0)
    
    
    
    
if __name__=='__main__':
    main()