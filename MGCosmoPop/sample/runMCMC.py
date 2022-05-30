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


params_O3 = {   'R0': 25. ,  # Gpc^-3 yr^-1 

                 'Xi0': 1. , 
                 'n' : 0.,  
                'lambdaRedshift':1.8,
                'alpha1':1.5, 'alpha2':5.5, 'beta':1.5, 'deltam':5.0, 'ml':4.,  'mh':87.0, 'b':0.4                         
    }


params_O3_GW190814 = {   'R0': 24. ,  # Gpc^-3 yr^-1 
                        #'Xi0': 1. , 
                        #'n' : 0.5,  
                        'alpha1':1., 'alpha2':5.17, 'beta':0.28, 
                'lambdaRedshift':1.8,
                'mh':87.,
                'ml':2., 
                'deltam':0.4 ,
                'b':0.4                                  
    }




params_mock_BPL_5yr_aLIGOdesignSensitivity = {'H0':67.74, 'Om':0.3075, 'w0':-1., 'Xi0':1., 'n':1.91, 'R0':25.0,
                  'lambdaRedshift':2., 'alpha1':1.6, 'alpha2':5.6, 'beta':1.4, 'deltam':5.0, 'ml':4., 'mh':90.0, 'b':0.4}


#params_mock_BPL_5yr_aLIGOdesignSensitivity_MG = {'H0':67.74, 'Om':0.3075, 'w0':-1., 'Xi0':1.2, 'n':2., 'R0':25.0,
#                  'lambdaRedshift':2., 'alpha1':1.6, 'alpha2':5.6, 'beta':1.4, 'deltam':5.0, 'ml':4., 'mh':90.0, 'b':0.4}

params_mock_BPL_5yr_aLIGOdesignSensitivity_MG = {'H0':67.74, 'Om':0.3075, 'w0':-1.,
                                                 'Xi0':1.8, 'n':1.91, 'R0':50.0,
                                                 'alphaRedshift':3.,'betaRedshift':2, 'zp':2,

                                                 'alpha1':1.6, 'alpha2':5.6, 'beta':1.4,
                                                 'deltam':5.0, 'ml':4., 'mh':70.0, 'b':0.5}


params_mock_BPL_5yr_GR = {'H0':67.74, 'Om':0.3075, 'w0':-1, 'Xi0':1.0,
                          'n':2,'R0': 50.0,'alphaRedshift': 3.0, 'betaRedshift':2.0,'zp': 2.0,
                          'alpha1':1.6, 'alpha2':5.6,'beta': 1.4, 'deltam':5.0, 'ml':4.0,'mh': 70.0, 'b':0.5 }



params_mock = {'H0':67.74, 'Om':0.3075, 'w0':-1., 'Xi0':1., 'n':2., 'R0':60.0,
                  'lambdaRedshift':3., 'alpha':0.75, 'beta': 0., 'ml':5, 'mh':45, 'sh':0.1, 'sl':0.1

}

params_mock_BPL_5yr_GR_1410 = {'H0':67.74, 'Om':0.3075, 'w0':-1, 'Xi0':1.,
                               'n':2,'R0': 30.0,'alphaRedshift': 2.7, 'betaRedshift':2.0,'zp': 2.0,
                               'alpha1':1.6, 'alpha2':5.6,'beta': 1.4, 'deltam':5.0, 'ml':4.0,'mh': 70.0, 'b':0.5 }

params_mock_BPL_5yr_MG_1410 = {'H0':67.74, 'Om':0.3075, 'w0':-1, 'Xi0':1.8,
                               'n':2,'R0': 30.0,'alphaRedshift': 2.7, 'betaRedshift':2.0,'zp': 2.0,
                               'alpha1':1.6, 'alpha2':5.6,'beta': 1.4, 'deltam':5.0, 'ml':4.0,'mh': 70.0, 'b':0.5 }




which_spins={ 'gauss':'chiEff',
             'skip':'skip'
    
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


def check_params(params_inference, params_fixed, params_all, normalized):
    
    allparams = params_inference+list(params_fixed.keys())
    dupes =  find_duplicates(allparams)
    if dupes:
        raise ValueError('The following parameters are included both in params_inference and parametes_fixed. Check your choices! \n%s' %str(dupes))
    if normalized and 'R0' in params_inference:
        raise ValueError('Remove R0 from parameters to infer, or use normalized=False ! ')



def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='', type=str, required=False) # config file (do not put if resuming)
    parser.add_argument("--fout", default='', type=str, required=True) # output folder 
    parser.add_argument("--resume", default=0, type=int, required=False) # restart chain or not
    parser.add_argument("--parallelization", default='mpi', type=str, required=False) # mpi or pool 
    parser.add_argument("--nPools", default=1, type=int, required=False) # if using multiprocessing, specify number of pools
    FLAGS = parser.parse_args()

    
    fout = FLAGS.fout
    out_path=os.path.join(Globals.dirName, 'results', fout)
    
    def run():
    
    
    
        if FLAGS.resume==0:
            resume=False
        else: resume=True
    
        if resume and FLAGS.config!='':
            raise ValueError('If continuing a pre-existing chain, you do not have to specify the config file.')
    
        if not resume:
            configname = FLAGS.config
            configspath = 'configs/'
        #config = importlib.import_module(FLAGS.config, package=None)
        else:
            try:
                configspath=os.path.join(out_path, "configs")
                print('Creating directory %s' %configspath)
                os.makedirs(configspath)
            except FileExistsError:
                print('Using directory %s for configs files' %configspath)
            configname='config_tmp'
            confignameFull = os.path.join(out_path, configname)
        
            shutil.copy(os.path.join(out_path, 'config_original.py'), os.path.join(configspath,'config_tmp.py') )
        #configname = os.path.join(configspath, 'config_tmp')
            print('Reading config from %s...' %confignameFull)
            
            
        sys.path.append(configspath)
        config = importlib.import_module( configname, package=None)
        if resume:
            from inspect import getmembers, ismodule
            config_items = {item[0]: item[1] for item in getmembers(config) if '__' not in item[0]}
            print('Config items:')
            print(config_items)

    #if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
            print('Creating directory %s' %out_path)

    #else:
        except FileExistsError:
            print('Using directory %s for output' %out_path)
    
        if not resume:
            shutil.copy(os.path.join( 'configs/', FLAGS.config+'.py'), os.path.join(out_path, 'config_original.py'))
    
        baselogfileName= 'logfile'
        if resume:
            logfile=os.path.join(out_path, baselogfileName+'_run1.txt')
            nResume=1
            while os.path.exists(logfile):
                logfileName=baselogfileName+'_run'+str(nResume)+'.txt'
                logfile=os.path.join(out_path, logfileName)
                nResume += 1
        else:   
            logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    
        myLog = utils.Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog
        filenameT=None
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
    
    
        ##############################################################
        # POPULATION MODELS
        print('\nCreating populations...')
    
        # Pass correct unit to rate nosmalization
        rate_args=config.rate_args
        rate_args['unit'] = units[config.dist_unit]
    
        allPops = build_model( config.populations, 
                              cosmo_args ={'dist_unit':units[config.dist_unit]},  
                              mass_args=config.mass_args, 
                              spin_args=config.spin_args, 
                              rate_args=rate_args, 
                              )
        
        # Check that params inference and not inference are correctly specified
        # (No duplicates)
        check_params(config.params_inference, config.params_fixed, allPops.params, config.normalized )
        
        
        # If using O3 data, set base values to expected ones from LVC
        if 'O3a' in config.dataset_names or 'O1O2' in config.dataset_names or 'O3b' in config.dataset_names:
            
            try:
                if ('GW190814' in config.O3_use['not_use']):
                    # Exclude 'GW190814' . Standard choice
                    params_MCMC_start=params_O3
                else:
                    print('GW190814 is included in the analysis. Setting expected values to %s' %str(params_O3_GW190814))
                    params_MCMC_start= params_O3_GW190814
            except:
                params_MCMC_start=params_O3
        else:
            assert len(config.dataset_names)==1
            if 'mock_BPL_5yr_aLIGOdesignSensitivity'==config.dataset_names[0]:
                params_MCMC_start=params_mock_BPL_5yr_aLIGOdesignSensitivity
            elif 'mock_BPL_5yr_aLIGOdesignSensitivity_MG'==config.dataset_names[0]:
                params_MCMC_start=params_mock_BPL_5yr_aLIGOdesignSensitivity_MG
            elif 'mock'==config.dataset_names[0]:
                params_MCMC_start= params_mock
            elif ('mock_BPL_5yr_GR'==config.dataset_names[0]) or ('mock_BPL_5yr_GR_1511'==config.dataset_names[0]):
                params_MCMC_start= params_mock_BPL_5yr_GR

        if config.normalized and 'R0' in params_MCMC_start.values():
            print('Removing R0 from parameters')
            _ = params_MCMC_start.pop('R0')
            
            
        allPops.set_values( params_MCMC_start)
        
        # Fix values of the parameters not included in the MCMC
        allPops.set_values( config.params_fixed)
        
        
        # Check that the order of arguments for MCMC in the config matches the order
        # of arguments in population
        allPops.check_params_order(config.params_inference)
        
        
        if units[config.dist_unit]==u.Mpc and not config.normalized:
            print('Converting expected value of rate to yr Mpc^-3')
            R0base = allPops.get_base_values('R0')[0]
            #print(R0base)
            new_rate = {   
                'R0': R0base*1e-09,}
            allPops.set_values( new_rate)   
        
        ############################################################
        # DATA
        
        allData=[]
        allInjData=[]
        for i,dataset_name in enumerate(config.dataset_names):
            
            if dataset_name in ('O3a', 'O1O2', 'O3b'):
                O3_use=config.O3_use
            elif 'mock' in dataset_name:
                O3_use=None
        
            print('\nLoading data from %s catalogue...' %dataset_name) 
            
            # This is a hack because the code does not yes support the option of different populations with different spin models
            # To be fixed in case is needed
            spindist = config.populations[list(config.populations.keys())[0]]['spin_distribution']
            
            try:
                SNR_th = config.SNR_th
            except:
                SNR_th = 8.

            try:
                FAR_th = config.FAR_th
            except:
                FAR_th = 1.
                print('FAR_th not found in config. Using 1/yr ')

            Data, injData = load_data(dataset_name, injections_name=config.injections_names[i],
                                      nObsUse=config.nObsUse, nSamplesUse=config.nSamplesUse, percSamplesUse=config.percSamplesUse, nInjUse=config.nInjUse, 
                                      dist_unit=units[config.dist_unit], 
                                      data_args={'events_use':O3_use, 'which_spins':which_spins[spindist], 'SNR_th':SNR_th, 'FAR_th': FAR_th }, 
                                      inj_args={'which_spins':which_spins[spindist], 'snr_th':SNR_th },
                                      Tobs=config.Tobs)
            allData.append(Data)
            allInjData.append(injData)
        
        
        #Data, injData = load_data(config.dataset_name, nObsUse=config.nObsUse, nSamplesUse=config.nSamplesUse, nInjUse=config.nInjUse, dist_unit=units[config.dist_unit], events_use=O3_use, )
        
        
        ############################################################
        # STATISTICAL MODEL
        
        print('\nSetting up inference for %s ' %config.params_inference)
        
        
        print('Fixed parameters and their values: ')
        print(allPops.get_fixed_values(config.params_inference))
        
        myPrior = Prior(config.priorLimits, config.params_inference, config.priorNames, config.priorParams)
        
        myLik = HyperLikelihood(allPops, allData, config.params_inference, verbose=config.verbose_lik, safety_factor=config.safety_factor )
        
        selBias = SelectionBiasInjections( allPops, allInjData, config.params_inference, get_uncertainty=config.include_sel_uncertainty, normalized=config.normalized ) 
        
        myPost = Posterior(myLik, myPrior, selBias, verbose=config.verbose_inj, normalized=config.normalized)
    




        ############################################################
        # SETUP MCMC
        
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = "chains.h5"
        backend = emcee.backends.HDFBackend(os.path.join(out_path,filename))
        if not resume:
            backend.reset(config.nwalkers, ndim)
        
        
        # Initial position of the walkers
        
        if not resume:
            exp_values = allPops.get_base_values(config.params_inference)
            pos = setup_chain(config.nwalkers, exp_values, config.priorNames, config.priorLimits, config.priorParams, config.params_inference, perc_variation_init=config.perc_variation_init, seed=config.seed)
        else:
            print('Restarting chain from last point of the previous run')
            all_samples = backend.get_chain(discard=int(0), flat=False, thin=1)
            pos = all_samples[-1]
        
        print('nwalkers=%s, ndim=%s' %(pos.shape[0], pos.shape[1]))
        #if (ndim<5) & (config.nwalkers<5):
        #print('Initial positions of the walkers: %s' %str(pos))
        
    
        autocorr_fname = os.path.join(out_path, "autocorr.txt")
        accF_fname  = os.path.join(out_path, "acceptance_fraction.txt")

        if config.telegram_notifications:
            notify_start(config.telegram_id, config.telegram_bot_token, filenameT, fout, config.nwalkers, ndim, config.params_inference, config.max_steps)
    
    
    
        ############################################################
        # RUN MCMC
    
    
        sampler = emcee.EnsembleSampler(config.nwalkers, ndim, myPost.logPosterior, backend=backend, pool=pool)
    
    
        # Track autocorrelation time
        index = 0
        autocorr = np.empty(config.max_steps)
        accF = np.empty(config.max_steps)
        if resume:
            try:
                autocorr_old=np.loadtxt(autocorr_fname)
                index  =len(autocorr_old)
                autocorr[:index] = autocorr_old
            except TypeError:
                print('No points in older autocorrelation file')
            try:
                accF_old = np.loadtxt(accF_fname)
                index_1 = len(accF_old)
                #assert index_1 ==index
                accF[:index_1] = accF_old
            except TypeError:
                print('No points in older acceptance fraction file file')
        old_tau = np.inf

        # Sample
        for sample in sampler.sample(pos, iterations=config.max_steps, progress=True, skip_initial_state_check=False):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
            else:
                tau = sampler.get_autocorr_time(tol=0) # tol=0 is in order to continue the chain
                if np.any(np.isnan(tau)):
                    burnin=0
                    converged=False
                else:
                    burnin = int(2 * np.max(tau))
                    # Check convergence
                    converged = np.all(tau * config.convergence_ntaus < sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < config.convergence_percVariation)
                    
                autocorr[index] = np.mean(tau)
                accF[index] = np.mean(sampler.acceptance_fraction)
                index +=1
                print('Step n %s. Check autocorrelation: ' %(index*100))
                print(tau)
                np.savetxt(autocorr_fname, autocorr[:index])
                np.savetxt(accF_fname, accF[:index])
                print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
                
                
                
                if config.telegram_notifications:
                    N=len(sampler.get_chain())
                    utils.telegram_bot_sendtext( "%s: step No.  %s, converged=%s, N/%s=%s, burnin=%s" %(filenameT, sampler.iteration, converged, config.convergence_ntaus, N/config.convergence_ntaus, burnin ), config.telegram_id, config.telegram_bot_token)
                    utils.telegram_bot_sendtext("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), config.telegram_id, config.telegram_bot_token)
                
                if converged:
                    print('Chain has converged. Stopping. ')
                    break
                else:
                    print('Chain has not converged yet. ')
                    if np.any(np.isnan(tau)):
                        old_tau = np.inf
                    else:
                        old_tau = tau
        return allPops, sampler, index, autocorr, ndim, config, myLog, filenameT
    
    ##############################################################
    ##############################################################
    
    if FLAGS.parallelization == 'mpi':
        from schwimmbad import MPIPool
        myPool=MPIPool()
    elif FLAGS.parallelization == 'pool':
        from multiprocessing import Pool, cpu_count
        ncpu = cpu_count()
        print('Number of availabel cores:  %s ' %ncpu)
        npools = min(FLAGS.nPools,ncpu )
        print('Parallelizing on %s CPUs ' %npools)
        myPool=Pool(npools)
    
    with myPool as pool:
        if FLAGS.parallelization == 'mpi':
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            allPops, sampler, index, autocorr, ndim, config, myLog, filenameT  = run()
        else:
            allPops, sampler,  index, autocorr, ndim, config, myLog, filenameT = run()
    
              
    
    
    
    ############################################################
    # SUMMARY PLOTS           
    reader = emcee.backends.HDFBackend(os.path.join(out_path,'chains.h5'))
    
    
    print('Plotting chains... ')
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = reader.get_chain()
    print("all chains shape : {0}".format(samples.shape))
    labels = allPops.get_labels(config.params_inference)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");  
    
    plt.savefig(os.path.join(out_path,'chains.pdf'))  
    plt.close()
    
    tau = reader.get_autocorr_time(quiet=True)
    if np.any(np.isnan(tau)):
        burnin=0
        thin=1
    else:
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    print("flat chains shape (after discarding burnin phase and thin): {0}".format(samples.shape))
    

    
    print('Plotting cornerplot... ')
    trueValues=None
    #if 'mock' in config.dataset_names:
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
    
    #if resume:
        # rm config_tmp 'config_tmp.py'
    #    os.remove('config_tmp.py')
    
    
if __name__=='__main__':
    main()
