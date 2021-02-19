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
#from glob import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
import importlib
import Globals
from scipy.integrate import quad
import scipy.stats as ss

fout = 'test_plotAlphaWithR'

marginalise_R0=False
selection_integral_uncertainty=True
skip=['n',  ]

perc_variation=15
npoints=5
dataset_name='mock'


priors_types = {'R0': 'flatLog'} #, 'Om0':'gauss'}
priors_params = None#{'Om0': {'mu':0.3, 'sigma':0.01} }

in_time=time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--param", default='H0', type=str, required=True)
FLAGS = parser.parse_args()
param = FLAGS.param


with open('config.py', 'w') as f:
    f.write("from params import Params, PriorLimits")
    f.write("\ndataset_name='%s'" %dataset_name)
    f.write("\ndataset_name_injections='%s'" %dataset_name)
    f.write("\nnObsUse=None " ) #%nObsUse)
    f.write("\nnSamplesUse=None  " )
    f.write("\nnInjUse=None  " )
    f.write("\nmarginalise_rate=%s"%marginalise_R0)
    f.write("\nselection_integral_uncertainty=%s"%selection_integral_uncertainty)
    f.write("\nverbose_bias=True")
    f.write("\nmass_model='Farr' ")
    f.write("\nmass_normalization='integral' ")
    f.write("\npivot_val=64.4")
    
    #f.write("\nallMyPriors = PriorLimits()")
    #f.write("\nparams_inference = ['%s',]"%param)
    #f.write("\nmyPriorLims = allMyPriors.priorLimits(params_inference)" )
    
    
config = importlib.import_module('config', package=None)

##############################

if marginalise_R0 and 'R0' not in skip:
    skip.append('R0')




out_path=os.path.join(Globals.dirName, 'results', fout)
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

allMyPriors = PriorLimits()
allMyPriors.set_priors(priors_types=priors_types, priors_params=priors_params)

params_inference = [param,]
myPriorLims = allMyPriors.priorLimits(params_inference)


print('Parameter: %s' %param)
    #for param in myParams.allParams:
        
if param in skip:
        print('Skipping %s' %param)
        exit
else:
        truth = myParams.trueValues[param]
        print('True value: %s' %truth)
        eps=truth
        if truth==0:
            eps=1

        #limInf = config.myPriorLims.limInf[param]
        #limSup=config.myPriorLims.limSup[param]
        limInf, limSup =  myPriorLims[0]
        grid = np.sort(np.concatenate( [np.array([truth,]) , np.linspace( limInf, truth-(eps*perc_variation/100)-0.01, 5), np.linspace(truth-(eps*perc_variation/100), truth+(eps*perc_variation/100), npoints) , np.linspace( truth+(eps*perc_variation/100)+0.01, limSup, 5)]) )
        grid=np.unique(grid, axis=0)

        
        params_n_inference = [nparam for nparam in myParams.allParams if nparam!= param]

        Lambda_ntest = np.array([myParams.trueValues[param] for param in params_n_inference])
        #priorLimits  = [ (myPriorLims.limInf[param],myPriorLims.limSup[param] ), ]
        
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
        
        #print('Computing precomputed quantities for %s... ' %(param ) )
        
        myLambda = importlib.import_module('getLambda'+param, package=None)
        #Lambda_func = globals()['getLambda'+param]
        print('Grid values: %s' %str(grid) )
        print('Computing priorfor %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        logPrior=np.zeros(grid.shape[0] )
        for i,val in enumerate(grid):
             #Lambda = myLambda.get_Lambda(val, Lambda_ntest)
             logPrior[i] = mymodels.log_prior(val, myPriorLims, param, allMyPriors.get_logVals )
        print('log prior: %s ' %logPrior)
        
        
        
        print('Computing selection bias for %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        
        NdetRes=np.zeros( (grid.shape[0] , 2 ) )
        Vols=np.zeros( grid.shape[0]  )
        for i,val in enumerate(grid):
            Lambda = myLambda.get_Lambda(val, Lambda_ntest)
            m1_inj, m2_inj, z_inj = mymodels.get_mass_redshift(Lambda, which_data='inj')
            #print(precomputed_inj['m1'].shape)
            NdetRes[i] = mymodels.logSelectionBias(Lambda, m1_inj, m2_inj, z_inj, get_neff = selection_integral_uncertainty)
            
            H0val=myParams.trueValues['H0']
            Om0val=myParams.trueValues['Om0']
            w0val=myParams.trueValues['w0']
            gamma=myParams.trueValues['lambdaRedshift']
            if param=='H0':
                H0val=val
            if param=='lambdaRedshift':
                gamma=val
            if param=='Om0':
                Om0val=val
            if param=='w0':
                w0val=val
            Vols[i] = quad(  lambda z: np.exp(mymodels.logTobs+cosmo.log_dV_dz(z, H0val, Om0val, w0val)+mymodels.log_dtobsdtdet(z) + mymodels.log_dNdVdt(z, gamma) ), 0, mymodels.zmax  )[0]
       # NdetRes = np.array( [mymodels.selectionBias(val, precomputed['source_frame_mass1_injections'], precomputed['source_frame_mass2_injections'], precomputed['z_injections']) for val in grid  ] )
        
        
        logMuVals=NdetRes[:, 0].astype('float128')
        muVals= np.exp(logMuVals)#*1000
        NeffVals=NdetRes[:, 1]
        
        t1=time.time()
        print('\nSelection bias done for '+param+' in %.2fs' %(t1 - in_time))
        
        # This fixes the error I don't understand
        #if not marginalise_R0:
        #    muVals*=1000
        
        if param=='R0':
            R0Vals=grid #np.exp(grid)#*1e-09
        #    logR0vals=grid
        #    grid_plot = R0Vals
        #    truth_vline = np.exp(truth)
        else:
            R0Vals=np.repeat(myParams.trueValues['R0'], muVals.shape)#*1e-09
        #    logR0vals=np.repeat(myParams.trueValues['logR0'], muVals.shape)
        #    grid_plot= grid
        #    truth_vline = truth
        logR0vals=np.log(R0Vals)
        grid_plot= grid
        truth_vline = truth
        
        alphas = R0Vals*muVals/Vols
        ndet_t = (R0Vals[np.argwhere(grid==truth)]*muVals[np.argwhere(grid==truth)])[0,0]
        al_t = alphas[np.argwhere(grid==truth)][0,0]
        
        idx_ruth = np.argwhere(grid_plot==truth)
        lab = r'$N_{\rm det}$'+'('+myParams.names[param] +')'+'\n'+r'$N_{\rm det}$ (%s)=%s' %(truth, np.round(ndet_t, 1))   
        plt.plot(grid_plot, R0Vals*muVals, label=lab)
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$N_{det}$');
        plt.axvline(truth_vline, ls='--', color='k', lw=2);
        plt.axhline(5267, ls=':', color='k', lw=1.5);
        if param=='R0':
            plt.xscale('log')
        plt.legend(fontsize=16);
        plt.savefig( os.path.join(out_path, param+'_Ndet.pdf'))
        plt.close()
        
        
        
    
        
        idx_ruth = np.argwhere(grid_plot==truth)
        lab = r'$\alpha$'+'('+myParams.names[param] +')'+'\n'+r'$\alpha$ (%s)=%s' %(truth, np.round(al_t, 4))   
        plt.plot(grid_plot, alphas, label=lab)
        
        if param=='H0':
            n2 = grid_plot[-1]**3/(alphas[-1])
            plt.plot(grid_plot, grid_plot**3/n2, label=r'$\propto H_0^3$')
        
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$\alpha$');
        plt.axvline(truth_vline, ls='--', color='k', lw=2);
        #plt.axhline(5267, ls=':', color='k', lw=1.5);
        if param=='R0':
            plt.xscale('log')
        plt.legend(fontsize=16);
        plt.savefig( os.path.join(out_path, param+'_alpha.pdf'))
        plt.close()
        
        
        
        #print('log(N_det) at true value of %s: %s '%(truth, logR0vals[np.argwhere(grid==truth)]+logMuVals[np.argwhere(grid==truth)] ) )
        print('N_det at true value of %s: %s '%(truth, ndet_t ) )
        
        #print('log(alpha) at true value of %s: %s '%(truth, logMuVals[np.argwhere(grid==truth)] ) )
        print('alpha at true value of %s: %s '%(truth, al_t) )
        
        print('Computing likelihood for %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        logLik=np.zeros(grid.shape[0] )
        for i,val in enumerate(grid):
            Lambda=myLambda.get_Lambda(val, Lambda_ntest)
            m1_obs, m2_obs, z_obs = mymodels.get_mass_redshift(Lambda, which_data='obs')
            logLik[i] = mymodels.logLik(Lambda, m1_obs, m2_obs, z_obs)
        #logLik = np.array( [mymodels.logLik(Lambda, precomputed['source_frame_mass1_observations'],precomputed['source_frame_mass2_observations'],precomputed['z_observations'] ) for val in grid ] )
        print('\nLikelihood done for '+param+' in %.2fs' %(time.time() - t1))
        
        
        #logPosterior = np.array( [mymodels.log_posterior(val, Lambda_ntest, priorLimits) for val in grid ] )
        logPosterior_noSel = logLik  + logPrior
        
        
        if not marginalise_R0:
            logPosterior = logPosterior_noSel + mymodels.Nobs*logR0vals - R0Vals*muVals 
            if selection_integral_uncertainty:
                logPosterior+= (R0Vals*muVals*R0Vals*muVals)/2/NeffVals-ss.norm(loc=muVals, scale=muVals**2/NeffVals).logsf(0)+ss.norm(loc=R0Vals*muVals**2/NeffVals-muVals, scale=muVals**2/NeffVals).logsf(0)
        else:
            logPosterior = logPosterior_noSel  - mymodels.Nobs*logMuVals 
            if selection_integral_uncertainty:
                logPosterior+=(3 * mymodels.Nobs + mymodels.Nobs ** 2) / (2 * NeffVals)
                
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
print('\nDone for '+param+'.Total execution time: %.2fs' %(time.time() - in_time))
    
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
myLog.close() 

            
#if __name__=='__main__':
#    main()
