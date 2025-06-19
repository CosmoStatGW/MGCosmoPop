#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:23:59 2022

@author: Michi
"""


import os
import sys
import argparse   
import importlib
import shutil
import time
import copy

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
from observePopulation_GWFAST import Observations
#from SNRtools import NetworkSNR, Detector

import seaborn as sns
import numpy as np



##########################################################################
##########################################################################

def get_net(FLAGS):
    if FLAGS.netfile is not None:
            print('Custom detector file passed. Loading network specifications from %s...' %FLAGS.netfile)
            import json
            with open(FLAGS.netfile, 'r') as j:
                Net = json.loads(j.read())
    else:
        import gwfast.gwfastGlobals as glob
        if isinstance(FLAGS.net, list):
            from gwfast.gwfastGlobals import detectors as base_dets
            Net = { k: copy.deepcopy(base_dets).pop(k) for k in FLAGS.net } 
            for i,psd in enumerate(FLAGS.psds): 
                Net[FLAGS.net[i]]['psd_path'] = os.path.join(glob.detPath, psd)
        elif isinstance(FLAGS.net, dict):
            print('Custom detector passed specifications:' )
            Net = FLAGS.net
            for k in FLAGS.net.keys(): 
                psd = FLAGS.psds[k]
                Net[k]['psd_path'] = os.path.join(glob.detPath, psd)
            print(Net)
        else:
            print()
            raise ValueError()
        
    return Net


##########################################################################
##########################################################################


def main():
    
    in_time=time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", default='data', type=str, required=True) # "inj" or "data"
    parser.add_argument("--config", default='', type=str, required=True) # config file
    parser.add_argument("--fout", default='', type=str, required=True) # output folder (complete path)
    FLAGS_IN = parser.parse_args()
    
    shutil.copy(os.path.join( FLAGS_IN.config+'.py'), 'config_tmp.py')
    
    FLAGS = importlib.import_module('config_tmp', package=None)
    fout = FLAGS_IN.fout
    os.remove('config_tmp.py')    

    duty_cycle = FLAGS.duty_factor
    if FLAGS.duty_factor == 1.:
        FLAGS.duty_factor=None

    
    
    print('\n------------------------ Importing gwfast from %s... ------------------------------------\n'%FLAGS.gwastpath )
    sys.path.append(FLAGS.gwastpath)
    
    #########################################################################
    # gwfast imports
    
    import gwfast.gwfastGlobals as glob
    from gwfast.gwfastGlobals import detectors as base_dets
    from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD, IMRPhenomHM, IMRPhenomD_NRTidalv2, IMRPhenomNSBH
    from gwfast.signal import GWSignal
    from gwfast.network import DetNet
    from gwfast.fisherTools import compute_localization_region, fixParams, CheckFisher, CovMatr, compute_inversion_error
    from gwfast.gwfastUtils import  get_events_subset, save_detectors, load_population, save_data
    try:
        import lal
        from gwfast.waveforms import LAL_WF
    except ModuleNotFoundError:
        print('LSC Algorithm Library (LAL) is not installed, only the GWFAST waveform models are available, namely: TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM and IMRPhenomNSBH')

    
    # shortcuts for wf model names; have to be used in input
    wf_models_dict = {'IMRPhenomD':IMRPhenomD(), 
                      'IMRPhenomHM':IMRPhenomHM(), 
                      'tf2':TaylorF2_RestrictedPN(is_tidal=False, use_3p5PN_SpinHO=True),
                      'IMRPhenomD_NRTidalv2': IMRPhenomD_NRTidalv2(),
                      'tf2_tidal':TaylorF2_RestrictedPN(is_tidal=True, use_3p5PN_SpinHO=True),
                      'IMRPhenomNSBH':IMRPhenomNSBH(),
                      'tf2_ecc':TaylorF2_RestrictedPN(is_tidal=False, use_3p5PN_SpinHO=True, is_eccentric=True),
                      }
    
    
    #########################################################################
    # Out folder
    
    print('\n\n------------------------ Making output folder and log... ------------------------------------\n' )
    
    out_path = fout
    #if not os.path.exists(out_path):
    try:
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    #else:
    except FileExistsError:
        print('Using directory %s for output' %out_path)
    
    if FLAGS_IN.type=='inj':
        logname='logfile_inj.txt'
    else:
        logname='logfile_data.txt'

    logfile = os.path.join(out_path, logname) #out_path+'logfile.txt'
    myLog = utils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog 
      

    
    shutil.copy(os.path.join( FLAGS_IN.config+'.py'), os.path.join(out_path, 'config_original.py'))
    
    
    #########################################################################
    # Define reference population
        
    print('\n\n------------------------ Building population objects... ------------------------------------\n' )
    
    allPops = build_model( FLAGS.populations, 
                              cosmo_args ={'dist_unit':u.Gpc},  
                              mass_args=FLAGS.mass_args, 
                              spin_args=FLAGS.spin_args, 
                              rate_args=FLAGS.rate_args)
    
    if FLAGS.lambdaBase is not None:
        # Fix values of the parameters to the fiducials chosen
        allPops.set_values( FLAGS.lambdaBase)
    print('Parameters used to generate data:')
    print(allPops.params)
    print(allPops.get_base_values(allPops.params))
    
    
    
    #########################################################################
    ## Define detector network
    
    print('\n\n------------------------ Building network of detectors... ------------------------------------\n' )
    
    Net = get_net(FLAGS)
    
    if FLAGS.wf_model.split('-')[0] !=  'LAL':
        wf_model = wf_models_dict[ FLAGS.wf_model]
        wf_model_name =  type(wf_model).__name__
    else:
        is_tidal, is_prec, is_HM, is_ecc = False, False, False, False
        if 'tidal' in FLAGS.lalargs:
            is_tidal = True
        if 'precessing' in FLAGS.lalargs:
            is_prec = True
        if 'HM' in FLAGS.lalargs:
            is_HM = True
        if 'eccentric' in FLAGS.lalargs:
            is_ecc = True
        wf_model = LAL_WF(FLAGS.wf_model.split('-')[1], is_tidal=is_tidal, is_HigherModes=is_HM, is_Precessing=is_prec, is_eccentric=is_ecc)
        wf_model_name = FLAGS.wf_model
    

    
    print('\n------------ Network used:  ------------\n%s' %str(Net))
    if FLAGS.netfile is not None:
        print('(Custom detector file was passed. Loaded network specifications from %s.)' %FLAGS.netfile)
    print('------------------------\n')
    
    print('------ Waveform:------\n%s' %wf_model_name)
    print('------\n')
    
    
    
    mySignals = {}
    
    for d in Net.keys():
    
        mySignals[d] = GWSignal( wf_model, 
                psd_path= Net[d]['psd_path'],
                detector_shape = Net[d]['shape'],
                det_lat= Net[d]['lat'],
                det_long=Net[d]['long'],
                det_xax=Net[d]['xax'], 
                verbose=True,
                useEarthMotion = FLAGS.rot,
                fmin=FLAGS.fmin, fmax=FLAGS.fmax,
                IntTablePath=None, 
                DutyFactor=FLAGS.duty_factor, #[d], 
                               ) 
        
    
    myNet = DetNet(mySignals) 
    
    #myNet._update_all_seeds(seeds=FLAGS.seeds, verbose=True)
    
    fname_det_new = os.path.join(fout, 'detectors.json')
    save_detectors(fname_det_new, Net)
    
   
    #########################################################################
    ## Instantiate observations object
    print('\n\n------------------------ Building observations object and generating injections... ------------------------------------\n' )
    
    
    myObs = Observations( allPops, 
                         myNet,
                         zmax=FLAGS.zmax, 
                         out_dir=out_path,
                        #snr_th=FLAGS.snr_th, 
                        snr_th_dets = FLAGS.snr_th_dets, 
                         add_noise =  FLAGS.add_noise,
                         condition=FLAGS.condition,
                         eta_scatter=FLAGS.eta_scatter , mc_scatter = FLAGS.mc_scatter , theta_scatter =FLAGS.theta_scatter,
                        #seed=FLAGS.seed
     )   
    
         

    if FLAGS_IN.type=='inj':
        #########################################################################
        ## Generate injections 
        myObs.generate_injections(FLAGS.N_desired, chunk_size=int(FLAGS.chunk_size))
    elif FLAGS_IN.type=='data':
        #########################################################################
        ## Generate data
        m1s, m2s, zs, thetas, mc_obs, eta_obs, rho_obs, theta_obs, sigma_mc, sigma_eta, sigma_rho, sigma_theta, allm1Gen, allm2Gen, allzGen = myObs.generate_dataset(duty_cycle = duty_cycle, 
                                                                                                                                    tot_time_yrs=FLAGS.tot_time_yrs, 
                                                                                                                                    chunks = FLAGS.time_steps,
                                                                                                                                    seed=FLAGS.seed, 
                                                                                                                                    save=True, 
                                                                                                                                                                     return_vals=True, return_generated=True)

        print('Max redshift detected: %s' %np.max(zs))
        print('(max redshift sampled was %s)' %FLAGS.zmax)

        zbins=np.linspace(0, allzGen.max()*(1+0.1), 20)
        _ = plt.hist(zs, bins=zbins, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allzGen, bins=zbins, density=True, label='True', alpha=0.5)
        plt.xlabel('z')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'zs_dist.pdf') )
        plt.close()

        mbins=np.linspace(0, allm1Gen.max()*(1+0.1), 20)
        _ = plt.hist(m1s, bins=mbins, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allm1Gen, bins=mbins, density=True, label='True', alpha=0.5)
        plt.xlabel('m1')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'm1s_dist.pdf') )
        plt.close()
    
        _ = plt.hist(m2s, bins=mbins, density=True, label='Observed', alpha=0.5)
        _ = plt.hist(allm2Gen, bins=mbins, density=True, label='True', alpha=0.5)
        plt.xlabel('m2')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'm2s_dist.pdf') )
        plt.close()
    
        print('Getting samples from the likelihood for all observations....')
        myObs.get_likelihood_samples(Nsamples=FLAGS.Nsamples, out_path=out_path, seed=FLAGS.seed, samples_step=FLAGS.samples_step)

    else:
        raise ValueError()
    
    ##########################################################################
    
    print('\n\nDone. Total execution time: %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    myLog.close()
    
    
    
    
if __name__=='__main__':
    main()
