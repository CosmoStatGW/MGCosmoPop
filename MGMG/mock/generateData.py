#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:35:05 2021

@author: Michi
"""

import os
import sys
import argparse   
import importlib
import shutil
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import astropy.units as u
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"


import Globals
import utils
from sample.models import build_model
from observePopulation import Observations

import seaborn as sns



##########################################################################
##########################################################################


def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", default='data', type=str, required=True) # "inj" or "data"
    parser.add_argument("--config", default='', type=str, required=True) # config file
    parser.add_argument("--fout", default='', type=str, required=True) # output folder 
    FLAGS = parser.parse_args()
    
    config = importlib.import_module(FLAGS.config, package=None)
    fout = FLAGS.fout
    
    
    out_path=os.path.join(Globals.dataPath,  fout)
    
    out_path=os.path.join(Globals.dataPath,  fout)
    #if not os.path.exists(out_path):
    try:
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    #else:
    except FileExistsError:
        print('Using directory %s for output' %out_path)
    
    if FLAGS.type=='inj':
        logname='logfile_inj.txt'
    else:
        logname='logfile.txt'

    logfile = os.path.join(out_path, logname) #out_path+'logfile.txt'
    myLog = utils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog 
      
    
    ##########################################################################
    
        
    allPops = build_model( config.populations, 
                              cosmo_args ={'dist_unit':u.Gpc},  
                              mass_args=config.mass_args, 
                              spin_args=config.spin_args, 
                              rate_args=config.rate_args)
    
    if config.lambdaBase is not None:
        # Fix values of the parameters to the fiducials chosen
        allPops.set_values( config.lambdaBase)
    print('Parameters used to generate data:')
    print(allPops.params)
    print(allPops.get_base_values(allPops.params))
    
    ### Sample masses and redshift
    
    myObs = Observations( allPops, zmax=config.zmax, out_dir=out_path,
                    rho_th=config.rho_th, 
                    psd_base_path=Globals.detectorPath, 
                    from_file=config.from_file, psd_path=config.psd_path, psd_name =config.psd_name, 
                    approximant=config.approximant, verbose=True)
    
    
    if FLAGS.type=='data':
        m1s, m2s, zs, thetas, mc_obs, eta_obs, rho_obs, theta_obs, sigma_mc, sigma_eta, sigma_rho, sigma_theta, allm1Gen, allm2Gen, allzGen = myObs.generate_dataset(duty_cycle=config.duty_cycle, 
                                                                                                                                    tot_time_yrs=config.tot_time_yrs, 
                                                                                                                                    chunks = config.time_steps,
                                                                                                                                    seed=config.seed, 
                                                                                                                                    save=True, return_vals=True, return_generated=True)


    
        _ = plt.hist(zs, bins=20, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allzGen, bins=20, density=True, label='True', alpha=0.5)
        plt.xlabel('z')
        plt.savefig(os.path.join(out_path, 'zs_dist.pdf') )
        plt.close()
    
        _ = plt.hist(m1s, bins=20, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allm1Gen, bins=20, density=True, label='True', alpha=0.5)
        plt.xlabel('m1')
        plt.savefig(os.path.join(out_path, 'm1s_dist.pdf') )
        plt.close()
    
        _ = plt.hist(m2s, bins=20, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allm2Gen, bins=20, density=True, label='Observed', alpha=0.5)
        plt.xlabel('m2')
        plt.savefig(os.path.join(out_path, 'm2s_dist.pdf') )
        plt.close()
    
        print('Getting samples from the likelihood for all observations....')
        myObs.get_likelihood_samples(Nsamples=config.Nsamples)
    
    
    elif FLAGS.type=='inj':
        
        myObs.generate_injections(config.N_desired, chunk_size=int(config.chunk_size))
    
    
    ##########################################################################
    
    print('\nDone. Total execution time: %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close()
    
    
    
    
if __name__=='__main__':
    main()