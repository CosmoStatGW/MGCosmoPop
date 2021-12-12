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
#import shutil
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
from SNRtools import NetworkSNR, Detector

#import seaborn as sns



##########################################################################
##########################################################################


def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    
    #parser.add_argument("--type", default='data', type=str, required=True) # "inj" or "data"
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
    
    #if FLAGS.type=='inj':
    logname='logfile_inj.txt'
    #else:
    #    logname='logfile.txt'

    logfile = os.path.join(out_path, logname) #out_path+'logfile.txt'
    myLog = utils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog 
      
    
    #########################################################################
    # Define reference population
        
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
    
    #########################################################################
    ## Define detector network
    
    myDetNet = NetworkSNR() 
    for dname in config.detectors.keys():
            d_ = Detector( lat=config.detectors[dname]['lat'], 
                                    long=config.detectors[dname]['long'], 
                                    xax=config.detectors[dname]['xax'], 
                                    shape=config.detectors[dname]['shape'],
                           duty_cycle=config.detectors[dname]['duty_cycle'],
                                    detector_args=config.detectors[dname]['detector_args'],
                                    interpolator_args=config.detectors[dname]['interpolator_args'])
            myDetNet.add_det(dname, d_)
    
    
    #########################################################################
    ## Instantiate observations object
    myObs = Observations( allPops, myDetNet,
                         zmax=config.zmax, out_dir=out_path,
                    rho_th=config.rho_th, 
#                    psd_base_path=Globals.detectorPath, 
#                    from_file=config.from_file, psd_path=config.psd_path, psd_name =config.psd_name, 
#                    approximant=config.approximant, verbose=True)
 )   

    #populations,detNet,zmax, out_dir,rho_th=8., eta_scatter=5e-03, mc_scatter = 3e-02, theta_scatter = 5e-02,
     
#########################################################################
    ## Generate injections 
    myObs.generate_injections(config.N_desired, chunk_size=int(config.chunk_size))
    
    
    ##########################################################################
    
    print('\nDone. Total execution time: %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close()
    
    
    
    
if __name__=='__main__':
    main()
