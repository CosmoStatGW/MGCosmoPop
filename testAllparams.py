#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:59:58 2021

@author: Michi
"""
import time
import os
from params import Params, PriorLimits
import argparse
from utils import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
import importlib

in_time=time.time()



perc_variation=20
npoints=20
dataset_name='mock'

nObsUse=50

with open('config.py', 'w') as f:
    f.write("dataset_name='%s'" %dataset_name)
    f.write("\ndataset_name_injections='%s'" %dataset_name)
    f.write("\nnObsUse=None " ) #%nObsUse)
    f.write("\nnSamplesUse=None  " )
    f.write("\nnInjUse=None  " )
    


parser = argparse.ArgumentParser()
parser.add_argument("--param", default='H0', type=str, required=True)
FLAGS = parser.parse_args()
param = FLAGS.param



fout = 'test_oneVar_Full'
out_path=os.path.join(dirName, 'results', fout)
if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
else:
       print('Using directory %s for output' %out_path)
    
logfile = os.path.join(out_path, 'logfile_'+param+'.txt') #out_path+'logfile.txt'
myLog = Logger(logfile)
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
        grid = np.sort(np.concatenate( [np.array([truth,]) , np.linspace( myPriorLims.limInf[param], truth-(truth*perc_variation/100)-0.01, 5), np.linspace(truth-(truth*perc_variation/100), truth+(truth*perc_variation/100), npoints) , np.linspace( truth+(truth*perc_variation/100)+0.01, myPriorLims.limSup[param], 5)]) )
            
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
        
        
        print('Computing posterior for %s in range (%s, %s) on %s points... ' %(param, grid.min(), grid.max(), grid.shape[0] ) )
        logPosterior = np.array( [mymodels.log_posterior(val, Lambda_ntest, priorLimits) for val in grid ] )
        posterior = np.exp(logPosterior)
        posterior_norm =np.trapz(posterior, grid) 
        print('Done.')
        np.savetxt( os.path.join(out_path, param+'_values.txt') , np.stack([grid, posterior, posterior_norm], axis=1) )
        
        
        plt.plot(grid, posterior)
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$p$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.savefig( os.path.join(out_path, param+'_post.pdf'))
        plt.close()
        
        
        plt.plot(grid, posterior_norm)
        plt.xlabel(myParams.names[param]);
        plt.ylabel(r'$p$');
        plt.axvline(truth, ls='--', color='k', lw=2);
        plt.savefig( os.path.join(out_path, param+'_post_norm.pdf'))
        plt.close()
            

######
print('\nDone for '+param+' in %.2fs' %(time.time() - in_time))
    
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
myLog.close() 

            
#if __name__=='__main__':
#    main()