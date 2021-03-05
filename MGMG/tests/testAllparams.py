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


perc_variation = 15

skip=['n',  ]

AllpriorLimits =      {   'H0': {    'H0': (20, 140)}, 
                        'Xi0':{   'Xi0': (0.1, 10) }, 
                         'Om': { 'Om': (0.05, 1)},
                          'w0': {'w0': (-2, -0.5)},
                          'n': {'n': (0., 10) }, 
                          'R0': {'R0': (1e-01 , 1e03)}, # Gpc^-3 yr^-1
                         'lambdaRedshift': { 'lambdaRedshift': (-15, 10) },
                          'alpha': {'alpha': (-5, 10 )},
                          'beta': {'beta': (-5, 10 ) }, 
                          'ml': {'ml': (2, 20)}, 
                          'sl' :{'sl':( 0.01 , 1)}, 
                           'mh':{'mh':( 20, 100)},
                          'sh' :{'sh':(0.01, 1 )} }
               


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
        
        myPopulations = { 'astro' : { 'mass_function': 'smooth_pow_law',
                                     'spin_distribution': 'skip',
                                     'rate': 'simple_pow_law'
    
    
                            }
    
                         }
        
        allPops = build_model(myPopulations, rate_args={'unit':config.dist_unit} )
        
        ############################################################
        # DATA
        
        mockData, injData = load_data('mock', nObsUse=config.nObsUse, nSamplesUse=config.nSamplesUse, nInjUse=config.nInjUse, dist_unit=u.Gpc)
        
         
        ############################################################
        # MODEL
        
        params_inference= [config.param,]
        priorLimits=AllpriorLimits[config.param]
        priorNames = {config.param : 'flat'}
        priorParams = None
        
        myPrior = Prior(priorLimits, params_inference, priorNames, priorParams)
        
        myLik = HyperLikelihood(allPops, mockData, params_inference )
        
        selBias = SelectionBiasInjections( allPops, injData, params_inference, get_uncertainty=True )
        
        myPost = Posterior(myLik, myPrior, selBias)
        
        ############################################################
        # COMPUTE ALL QUANTITIES
        
        truth=allPops.get_baseValue(config.param)
        print('True value: %s' %truth)
        eps=truth
        if truth==0:
            eps=1
            
            
        limInf, limSup =  priorLimits[config.param]
        grid = np.sort(np.concatenate( [np.array([truth,]) , np.linspace( limInf+0.001, truth-(eps*perc_variation/100)-0.01, 5), np.linspace(truth-(eps*perc_variation/100), truth+(eps*perc_variation/100), config.npoints) , np.linspace( truth+(eps*perc_variation/100)+0.01, limSup-0.001, 5)]) )
        grid=np.unique(grid, axis=0)
        
        print('Grid values: %s' %str(grid) )
        print('Computing posterior with log_posterior %s in range (%s, %s) on %s points... ' %(config.param, grid.min(), grid.max(), grid.shape[0] ) )
    
   
        logPosteriorAll = np.array( [ myPost.logPosterior(val, return_all=True) for val in grid ] )
        # logPost, lp, ll, mu, err
        
        logPosterior=logPosteriorAll[:,0]
        logLik=logPosteriorAll[:,2]
        logPrior=logPosteriorAll[:,1]
        MuVals=logPosteriorAll[:,3]
        ErrVals=logPosteriorAll[:,4]
                
        
        print('lik:')
        print(logLik)
        
        ############################################################
        # PLOT N_det
        
        idx_truth = np.argwhere(grid==truth)
        ndet_t = MuVals[idx_truth]
        print('N_det at true value of %s: %s '%(truth, ndet_t ) )
        
        lab = r'$N_{\rm det}$'+'('+config.param +')'+'\n'+r'$N_{\rm det}$ (%s)=%s' %(truth, np.round(ndet_t, 1))   
        plt.plot(grid, MuVals, label=lab)
        plt.xlabel(config.param );
        plt.ylabel(r'$N_{det}$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.axhline(5267, ls=':', color='k', lw=1.5);
        if config.param=='R0':
            plt.xscale('log')
            plt.yscale('log')
        plt.legend(fontsize=16);
        plt.savefig( os.path.join(out_path, config.param+'_Ndet.pdf'))
        plt.close()
   
        ############################################################
        # PLOT posterior
        mymax = logPosterior.max()
        posterior = np.exp(logPosterior-mymax)
        posterior /=np.trapz(posterior, grid)
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