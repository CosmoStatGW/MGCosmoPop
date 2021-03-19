#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:42:13 2021

@author: Michi
"""
import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import argparse   
import importlib
import time

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"

from sample.models import build_model, load_data


import astropy.units as u

from posteriors.prior import Prior
from posteriors.likelihood import HyperLikelihood
from posteriors.selectionBias import SelectionBiasInjections
from posteriors.posterior import Posterior

import Globals
import utils


perc_variation = 10

skip=['n',  ]

AllpriorLimits =      {   'H0': {    'H0': (20., 140.)}, 
                        'Xi0':{   'Xi0': (0.1, 10.) }, 
                         'Om': { 'Om': (0.05, 1)},
                          'w0': {'w0': (-2, -0.5)},
                          'n': {'n': (0., 10) }, 
                          
                          'R0': {'R0': (1e-01 , 1e03)}, # Gpc^-3 yr^-1
                         'lambdaRedshift': { 'lambdaRedshift': (-15., 10.) },
                          
                         'alpha': {'alpha': (-5, 10 )},
                          'beta': {'beta': (-4, 12 ) }, 
                          'ml': {'ml': (2., 10.)}, 
                          'sl' :{'sl':( 0.01 , 1)}, 
                           'mh':{'mh':( 30., 150.)},
                          
                            'sh' :{'sh':(0.01, 1 )}, 
                          
                         'alpha1': {'alpha1': (-4., 12.)},                          
                         'alpha2': {'alpha2': (-4., 12.)}, 
                         #'beta': {}, 
                         'deltam': {'deltam': (0., 10.)},
                         #'ml': {}, 
                         #'mh': {}, 
                         'b': {'b': (0., 1.)} ,
                         
                          
                          'muEff': {'muEff': (-1, 1)},
                          'muP': {'muP': (0.01, 1)},
                          'sigmaEff': {'sigmaEff': (0.01, 1)},
                          'sigmaP': {'sigmaP': (0.01, 1)},
                          
                          
                          }
               



params_O3 = {   'R0': 24. ,  # Gpc^-3 yr^-1 
                        #'Xi0': 1. , 
                        #'n' : 0.5,  
                'lambdaRedshift':2.7,
                'mh':87.,
                'ml':4., 
                'beta'  :1.4                                    
    }




which_spins={ 'gauss':'chiEff',
             'skip':'skip'
    
    }








def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='', type=str, required=True) # config time
    FLAGS = parser.parse_args()
    
    config = importlib.import_module(FLAGS.config, package=None)
    
    
    if config.param in skip:
        print('Skipping %s' %config.param)
        exit
    else:
    
        out_path=os.path.join(Globals.dirName, 'results', config.fout)
        if not os.path.exists(out_path):
            print('Creating directory %s' %out_path)
            os.makedirs(out_path)
        else:
            print('Using directory %s for output' %out_path)
    
        logfile = os.path.join(out_path, 'logfile_'+config.param+'.txt') #out_path+'logfile.txt'
        myLog = utils.Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog
        
        ##############################################################
        # POPULATION MODELS
        
        myPopulations = { 'astro' : { 'mass_function': config.massf,
                                     'spin_distribution': config.spindist,
                                     'rate': 'simple_pow_law'
    
    
                            }
    
                         }
        
        allPops = build_model(myPopulations, rate_args={'unit':u.Gpc} )
        
        if 'O3a' in config.data or 'O1O2' in config.data:
            allPops.set_values( params_O3)
        
        ############################################################
        # DATA
        
        #if 'O3a' in config.data or 'O1O2' in config.data:
        #    events_use=config.events_use
        #elif config.data=='mock':
        #    events_use=None
        allData=[]
        allInjData=[]
        for data_ in config.data:
            if data_ in ('O1O2', 'O3a'):
                events_use=config.events_use
            elif data_=='mock':
                events_use=None
            Data, injData = load_data(data_, nObsUse=config.nObsUse, nSamplesUse=config.nSamplesUse, nInjUse=config.nInjUse, dist_unit=u.Gpc, data_args={'events_use':events_use, 'which_spins':which_spins[config.spindist]}, inj_args={'which_spins':which_spins[config.spindist] })
            allData.append(Data)
            allInjData.append(injData)
         
        
        ############################################################
        # MODEL
        
        params_inference= [config.param,]
        priorLimits=AllpriorLimits[config.param]
        priorNames = {config.param : 'flat'}
        priorParams = None
        
        myPrior = Prior(priorLimits, params_inference, priorNames, priorParams)
        
        myLik = HyperLikelihood(allPops, allData, params_inference )
        
        selBias = SelectionBiasInjections( allPops, allInjData, params_inference, get_uncertainty=True )
        
        myPost = Posterior(myLik, myPrior, selBias)
        
        ############################################################
        # COMPUTE ALL QUANTITIES
        
        truth=allPops.get_base_values(config.param)[0]
        print('True value: %s' %truth)
        eps=truth
        if truth==0:
            eps=1
            
            
        limInf, limSup =  priorLimits[config.param]
        grid = np.sort(np.concatenate( [np.array([truth,]) , np.linspace( limInf+0.001, truth-(eps*perc_variation/100)-0.01, 5), np.linspace(truth-(eps*perc_variation/100), truth+(eps*perc_variation/100), config.npoints) , np.linspace( truth+(eps*perc_variation/100)+0.01, limSup-0.001, 5)]) )
        grid=np.unique(grid, axis=0)
        
        print('Grid values: %s' %str(grid) )
        print('Computing posterior with log_posterior %s in range (%s, %s) on %s points... ' %(config.param, grid.min(), grid.max(), grid.shape[0] ) )
    
   
        #print( myPost.logPosterior(truth, return_all=True) )
        
        logPosteriorAll = np.array( [ myPost.logPosterior(val, return_all=True) for val in grid ])
        # logPost, lp, ll, mu, err
        #print(logPosteriorAll)
        logPosterior=logPosteriorAll[:,0]
        logPrior=logPosteriorAll[:,1]
        logLik=logPosteriorAll[:,2]
        MuVals=logPosteriorAll[:,3]
        #ErrVals=logPosteriorAll[:,4]
                
        
        print('logPrior:')
        print(logPrior)
        print('logLik:')
        print(logLik)
        print('mu:')
        print(MuVals)
        print('logPosterior:')
        print(logPosterior)
        
        ############################################################
        # PLOT N_det
        idx_truth = np.argwhere(grid==truth)
        mutot=np.zeros(MuVals[0].shape)
        ndet_tot=0
        NobsTot=0
        for i in range(len(allData)):
            
            data=allData[i]
            mu=MuVals[i]
            
            ndet_t = mu[idx_truth][0][0]
            print('N_det for %s at true value of %s: %s '%(config.data[i], truth, ndet_t ) )
        
            lab = r'$N_{\rm det}$, %s'%config.data[i]+'('+config.param +')'+'\n'+r'$N_{\rm det}$ (%s)=%s' %(truth, np.round(ndet_t, 1) )   
            plt.plot(grid, mu, label=lab)
            if config.nObsUse is None:
                plt.axhline(data.Nobs, ls=':', color='k', lw=1.5);
                
            mutot+=mu
            ndet_tot+=ndet_t
            NobsTot+=data.Nobs
        
        print('N_det total at true value of %s: %s '%( truth, ndet_tot ) )
        
        lab = r'$N_{\rm det}$ tot'+'('+config.param +')'+'\n'+r'$N_{\rm det}$ (%s)=%s' %(truth, np.round(ndet_tot, 1) )   
        plt.plot(grid, mutot, label=lab)
        if config.nObsUse is None:
                plt.axhline(NobsTot, ls=':', color='k', lw=1.5);
        
        
        plt.xlabel(config.param );
        plt.ylabel(r'$N_{det}$');
        
        plt.axvline(truth, ls='--', color='k', lw=2);
        
        if config.param=='R0':
            plt.xscale('log')
            plt.yscale('log')
        plt.legend(fontsize=16);
        plt.savefig( os.path.join(out_path, config.param+'_Ndet.pdf'))
        plt.close()
   
        ############################################################
        # PLOT posterior
        mymax = logPosterior.max().astype('float128')
        print('max of log posterior:')
        print(mymax)
        posterior = np.exp(logPosterior-mymax)
        posterior /=np.trapz(posterior, grid)
        print('normalized posterior:')
        print(posterior)
        logPosterior_noSel = logLik  + logPrior
        posterior_noSel = np.exp(logPosterior_noSel-logPosterior_noSel.max())
        posterior_noSel /=np.trapz(posterior_noSel, grid) 
        
        np.savetxt( os.path.join(out_path, config.param+'_values.txt') , np.stack([grid, logPosterior, posterior], axis=1) )
        
        
        
        plt.plot(grid, logPosterior, label='With sel effects')
        
        plt.plot(grid, logPosterior_noSel, label='No sel effects')
        plt.xlabel(config.param);
        plt.ylabel(r'$p$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.legend()
        if config.param=='R0':
            plt.xscale('log')
        plt.savefig( os.path.join(out_path, config.param+'_logpost.pdf'))
        plt.close()
        
        
        plt.plot(grid, posterior, label='With sel effects')
        
        plt.plot(grid, posterior_noSel, label='No sel effects')
        plt.xlabel(config.param);
        plt.ylabel(r'$p$');
        plt.legend()
        if config.param=='R0':
            plt.xscale('log')
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.savefig( os.path.join(out_path, config.param+'_post.pdf'))
        plt.close()
            

        ######
        print('\nDone for '+config.param+'.Total execution time: %.2fs' %(time.time() - in_time))
    
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        myLog.close() 

    
   
if __name__=="__main__":
    
    main()