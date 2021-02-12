#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:59:58 2021

@author: Michi
"""
import time
import sys
import os
from params import Params, PriorLimits
import argparse
import utils
import cosmo
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
import importlib

in_time=time.time()



perc_variation=15
npoints=5
dataset_name='mock'

nObsUse=50

with open('config.py', 'w') as f:
    f.write("dataset_name='%s'" %dataset_name)
    f.write("\ndataset_name_injections='%s'" %dataset_name)
    f.write("\nnObsUse=None " ) #%nObsUse)
    f.write("\nnSamplesUse=None  " )
    f.write("\nnInjUse=None  " )
    f.write("\nmarginalise_rate=False")
    f.write("\nselection_integral_uncertainty=True")


parser = argparse.ArgumentParser()
parser.add_argument("--param", default='H0', type=str, required=True)
FLAGS = parser.parse_args()
param = FLAGS.param



fout = 'test_oneVar_withNdet_Farr_wCDM'
out_path=os.path.join(utils.dirName, 'results', fout)
if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
else:
       print('Using directory %s for output' %out_path)
    
logfile = os.path.join(out_path, 'logfile_'+param+'.txt') #out_path+'logfile.txt'
myLog = utils.Logger(logfile)
sys.stdout = myLog
sys.stderr = myLog



#def main():
    
    
    
    

    
myParams = Params(dataset_name)
myPriorLims = PriorLimits()
    
print('Parameter: %s' %param)
    #for param in myParams.allParams:
        
if param=='n':
        print('Skipping n')
        exit
else:
        truth = myParams.trueValues[param]
        print('True value: %s' %truth)
        eps=truth
        if truth==0:
            eps=1e-01
        grid = np.sort(np.concatenate( [np.array([truth,]) , np.linspace( myPriorLims.limInf[param], truth-(eps*perc_variation/100)-0.01, 5), np.linspace(truth-(eps*perc_variation/100), truth+(eps*perc_variation/100), npoints) , np.linspace( truth+(eps*perc_variation/100)+0.01, myPriorLims.limSup[param], 5)]) )
        grid=np.unique(grid, axis=0)
        params_n_inference = [nparam for nparam in myParams.allParams if nparam!= param]

        Lambda_ntest = np.array([myParams.trueValues[param] for param in params_n_inference])
        priorLimits  = [ (myPriorLims.limInf[param],myPriorLims.limSup[param] ), ]
        
        print('Writing get_Lambda:' )
        lines=[]
        lines.append('def get_Lambda(Lambda_test, Lambda_ntest):')
        lines.append('    '+ param + ' = Lambda_test')
        lines.append('    '+ (', ').join(params_n_inference) + ' = Lambda_ntest')
        lines.append('    Lambda = ['+ (', ').join(myParams.allParams) +']')
        lines.append('    return Lambda' )
        print(('\n').join(lines) )
        
        with open('getLambda'+param+'.py', 'w') as f:
            f.write(('\n').join(lines))
        
        fread= open('models.py', 'r')
        fwrite=open('models'+param+'.py', 'w')
        for line in fread.readlines():
            ll=line
            if 'getLambda' in line:
                ll = 'from getLambda'+param+' import *'
            fwrite.write(ll+"\n")
        fwrite.close()
        fread.close()
        
        #from models import log_posterior
        mymodels = importlib.import_module('models'+param, package=None)
        
        
        print('Computing Ndet for %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        
        
        NdetRes = np.array( [mymodels.selectionBias(val, Lambda_ntest) for val in grid  ] )
        muVals=NdetRes[:, 0]*1000
        NeffVals=NdetRes[:, 1]
        
        if param=='R0':
            R0Vals=grid*1e-09
        else:
            R0Vals=np.repeat(myParams.trueValues['R0'], NeffVals.shape)*1e-09
        
        plt.plot(grid, R0Vals*muVals)
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$N_{det}$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        #plt.axhline(5267, ls=':', color='k', lw=1.5);
        plt.savefig( os.path.join(out_path, param+'_Ndet.pdf'))
        plt.close()
        
        print('N_det at true value of %s: %s '%(truth, R0Vals[np.argwhere(grid==truth)]*muVals[np.argwhere(grid==truth)] ) )
        
        print('Computing likelihood for %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        logLik = np.array( [mymodels.logLik(val, Lambda_ntest) for val in grid ] )
        print('Likelihood done for %s.' %param)
        
        logPrior = np.array([mymodels.log_prior(val, priorLimits) for val in grid ] )
        
        #logPosterior = np.array( [mymodels.log_posterior(val, Lambda_ntest, priorLimits) for val in grid ] )
        logPosterior_noSel = logLik  + logPrior
        
        
        
        logPosterior = logPosterior_noSel + mymodels.Nobs*np.log(R0Vals) + R0Vals*muVals*(R0Vals*muVals-2*NeffVals)/2/NeffVals #- muVals + (3 * mymodels.Nobs + mymodels.Nobs ** 2) / (2 * NeffVals)
        
        posterior = np.exp(logPosterior-logPosterior.max())
        posterior /=np.trapz(posterior, grid)
        
        posterior_noSel = np.exp(logPosterior_noSel-logPosterior_noSel.max())
        posterior_noSel /=np.trapz(posterior_noSel, grid) 
        print('Done.')
        np.savetxt( os.path.join(out_path, param+'_values.txt') , np.stack([grid, logPosterior, posterior], axis=1) )
        
        
        plt.plot(grid, logPosterior, label='With sel effects')
        plt.plot(grid, logPosterior_noSel, label='No sel effects')
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$p$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.legend()
        plt.savefig( os.path.join(out_path, param+'_logpost.pdf'))
        plt.close()
        
        
        plt.plot(grid, posterior, label='With sel effects')
        plt.plot(grid, posterior_noSel, label='No sel effects')
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$p$');
        plt.legend()
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.savefig( os.path.join(out_path, param+'_post.pdf'))
        plt.close()
            

######
print('\nDone for '+param+' in %.2fs' %(time.time() - in_time))
    
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
myLog.close() 

            
#if __name__=='__main__':
#    main()
