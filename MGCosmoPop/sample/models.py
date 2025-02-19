#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:01:46 2021

@author: Michi
"""
import os
import sys
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import Globals

import astropy.units as u

from population.astro.astroMassDistribution import AstroSmoothPowerLawMass, BrokenPowerLawMass, TruncPowerLawMass, PowerLawPlusPeakMass, MultiPeakMass, BNSGaussMass, BNSFlatMass, BrokenPowerLawMassBNS
from population.astro.astroSpinDistribution import DummySpinDist, GaussSpinDist, UniformSpinDistChiz, DefaultSpinModel, UniformOnSphereSpin
from population.astro.rateEvolution import PowerLawRateEvolution, AstroPhRateEvolution, RateEvolutionCOBA
from population.astro.astroPopulation import AstroPopulation
from cosmology.cosmo import Cosmo
from population.allPopulations import AllPopulations

from dataStructures.mockData import GWMockData, GWMockInjectionsData
from dataStructures.O3adata import O3aData, O3aInjectionsData
from dataStructures.O1O2data import O1O2Data, O1O2InjectionsData
from dataStructures.O3bdata import O3bData, O3bInjectionsData
from dataStructures.ABSdata import GWTC3InjectionsData

#import astropy.units as u

#from posteriors.prior import Prior
#from posteriors.likelihood import HyperLikelihood

#from posteriors.selectionBias import SelectionBiasInjections
#from posteriors.posterior import Posterior



mass_functions = {  'smooth_pow_law': AstroSmoothPowerLawMass,
                      'broken_pow_law': BrokenPowerLawMass,
                      'trunc_pow_law': TruncPowerLawMass,
                      'pow_law_peak': PowerLawPlusPeakMass,
                      'multi_peak': MultiPeakMass,
                      'flat_BNS':BNSFlatMass,
                      'gauss_BNS':BNSGaussMass,
                      'broken_BNS':BrokenPowerLawMassBNS
    
     }

spin_functions = {  'gauss': GaussSpinDist,
                      'skip': DummySpinDist,
                      'flat':UniformSpinDistChiz,
                  'default':DefaultSpinModel,
                  'uniform_on_sphere':UniformOnSphereSpin
     }


rate_functions = { # 'gauss': AstroSmoothPowerLawMass(),
                  
                      'simple_pow_law': PowerLawRateEvolution, 
                               'astro-ph'       : AstroPhRateEvolution, 
                               'COBA_BNS':RateEvolutionCOBA
     }



fnames_data  = { 'mock': 'observations.h5',
                'O3a': '',
                'O1O2': '',
                 'O3b': '',
    
    
    }


fnames_inj  = { 'mock': 'selected.h5',
                'O3a': 'o3a_bbhpop_inj_info.hdf',
                'O1O2':'injections_O1O2an_spin.h5',
                'O3b':''
    }


fnames_inj_3  = { 'mock': 'selected.h5',
                'O3a': 'endo3_bbhpop-LIGO-T2100113-v12-1238166018-15843600.hdf5',
                'O1O2':'injections_O1O2an_spin.h5',
                'O3b':'endo3_bbhpop-LIGO-T2100113-v12-1256655642-12905976.hdf5'
    }

fnames_inj_all  = { #'mock': 'selected.h5',
                'O3a':'o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5',
                'O1O2':'o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5',
                'O3b':'o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5',
    }


fnames_SNRs = { 'mock': 'optimal_snr.h5'
    
    }


def setup_chain(nwalkers, exp_values, priorNames, priorLimits, priorParams, params_inference, perc_variation_init=10, seed=1312):
    '''

    Returns initial position for the walkers
    -------
    pos : TYPE
        DESCRIPTION.

    '''
    
    ndim=len(params_inference)
    eps = [val if val!=0 else 1 for val in exp_values ]
    lowLims = [ max( [exp_values[i]-eps[i]*perc_variation_init/100, priorLimits[p][0] ] )  for i,p in enumerate(params_inference) ] 
    upLims = [ min( [ exp_values[i]+eps[i]*perc_variation_init/100, priorLimits[p][1] ]) for i,p in enumerate(params_inference) ] 
    for i in range(len(lowLims)):
            if lowLims[i]>upLims[i]:
                lowLim=upLims[i]
                upLim=lowLims[i]
                upLims[i]=upLim
                lowLims[i]=lowLim
            if priorNames[params_inference[i]]=='gauss':
                print('Re-adjusting intervals for %s to gaussian prior limits...' %params_inference[i])
                mu, sigma = priorParams[params_inference[i]]['mu'], priorParams[params_inference[i]]['sigma']
                lowLim = lowLims[i]
                upLim=upLims[i]
                lowLims[i] = max(lowLim, mu-5*sigma)
                upLims[i] = min(upLim, mu+5*sigma)
            
    print('lowLims: %s' %lowLims)
    print('upLims: %s' %upLims)
    Delta = [upLims[i]-lowLims[i] for i in range(ndim)]  
    print('Delta: %s' %Delta)
    print('Initial intervals for initialization of the walkers have an amplitude of +-%s percent around the expeced values of %s'%(perc_variation_init, str(exp_values)) )
    np.random.seed(seed)
    pos = Delta*np.random.rand(nwalkers,  ndim)+lowLims
    
    return pos



def build_model( populations, cosmo_args={}, mass_args={}, spin_args={}, rate_args={}, ):
    '''
    

    Parameters
    ----------
    populations : dict
        {  pop_name : {mass_function: ,  spin_distribution: , rate: }  }.

    Returns
    -------
    None.

    '''
    
    # Create cosmology
    myCosmo = Cosmo(**cosmo_args)
        
    try:
        n = rate_args['normalized']
    except:
        n = False
        
    # Collector of all populations
    allPops = AllPopulations(myCosmo, normalized=n)
    
    
    for population in populations.keys():
            
            print('Adding population %s' %population)
        
            # Create mass dist
            massFunction = mass_functions[populations[population]['mass_function']](**mass_args)
        
            # Create spin dist  
            spinDist = spin_functions[populations[population]['spin_distribution']](**spin_args)
        
            # Create rate
            rate = rate_functions[populations[population]['rate']](**rate_args)
        
            # Create population
            pop_ = AstroPopulation(rate, massFunction, spinDist)
            
            allPops.add_pop(pop_)
            
    return allPops
            




def load_data(dataset_name, injections_name=None, nObsUse=None, nSamplesUse=None, percSamplesUse=None, nInjUse=None, dist_unit=u.Gpc, data_args ={}, inj_args ={}, Tobs=None, data_path=None, inj_path=None):
        
        # for O2/O3, injections_name can be None in which case the LVC injections are used (they should be in the same folder as the data)
        # or specify the name of a folder containing a file 'selected.h5'
        
        ############################################################
        # DATA

        if data_path is None:
            data_path = Globals.dataPath
        if inj_path is None:
            inj_path = Globals.dataPath
        
        if "mock" in dataset_name:
            dataset_key='mock'
        else:
            dataset_key=dataset_name
            #if 'dLprior' in data_args.keys():
            #    _ = data_args.pop('dLprior')
            
            try:
                #injections_name=='GWTC-3-all'
                if inj_args['which_injections']=='GWTC-2':
                    fnameInj = os.path.join(inj_path, dataset_name, fnames_inj[dataset_key] )
                elif inj_args['which_injections']=='GWTC-3':
                    fnameInj = os.path.join(inj_path, dataset_name, fnames_inj_3[dataset_key] )
                elif inj_args['which_injections']=='GWTC-3-all':
                    fnameInj = os.path.join(inj_path, dataset_name, fnames_inj_all[dataset_key] )
                    _ = inj_args.pop('which_injections')
            except:
                pass
        
        if dataset_name not in ['O1O2', 'O3a', 'O3b']:
            fname = os.path.join(data_path, dataset_name, 'observations.h5')
        else:
            fname = os.path.join(data_path, dataset_name, fnames_data[dataset_key])



    
        if dataset_name=='O3a':
            
            Data = O3aData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            
            if injections_name is None:
                
                injData = O3aInjectionsData(fnameInj ,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
            
            elif injections_name=='GWTC-3-all':

                injData = None
            
            else:
                # the name of some folder was passed to injections_name
                
                # the folder must be under inj_path/O3a/
                
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5') #fnames_inj[dataset_key] )
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Data.Tobs, snr_th=snr_th )
                         
        elif dataset_name=='O1O2':
            
            Data = O1O2Data(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            
            if injections_name is None:
                injData = O1O2InjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
            
            elif injections_name=='GWTC-3-all':

                injData = GWTC3InjectionsData(fnameInj, nInjUse=nInjUse, dist_unit=dist_unit, **inj_args )
            
            else:
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5')
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Data.Tobs, snr_th=snr_th)
        
        
        elif dataset_name=='O3b':
            
            Data = O3bData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            
            if injections_name is None:
                injData = O3bInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
                #raise ValueError('LVC injections are not supported for O3b. Specify a name for the injections')

            elif injections_name=='GWTC-3-all':

                injData = None
                
            else:
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5')
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Data.Tobs, snr_th=snr_th )
        else:
            #raise ValueError('Dataset name not valid')
            print("Using simulated data")
            #if dataset_key=='mock':
            print(data_args)
            if 'dLprior' in data_args.keys():
                dLp = data_args['dLprior']
                print('dL prior is %s'%dLp)
            else:
                dLp=None
                print('dL prior is None')
            if dLp=='none':
                dLp=None
                print('dL prior is now None')
                
            
            Data = GWMockData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, Tobs=Tobs, dLprior=dLp)
            if 'SNR_th' in inj_args.keys():
                snr_th = inj_args['SNR_th']
            else: snr_th=None

            if dataset_key=='mock':
                fnameInj = os.path.join(inj_path, injections_name, fnames_inj[dataset_key] )
            else:
                 fnameInj = os.path.join(inj_path, injections_name, 'selected.h5' )
            
            injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Tobs, snr_th=snr_th)
        
        return Data, injData
    
    
    
    
def load_injections(dataset_name, injections_name=None,  nInjUse=None, dist_unit=u.Gpc,  inj_args ={}, Tobs=None, inj_path=None):
        
        # for O2/O3, injections_name can be None in which case the LVC injections are used (they should be in the same folder as the data)
        # or specify the name of a folder containing a file 'selected.h5'

        if inj_path is None:
            inj_path = Globals.dataPath
            
        ############################################################
        # DATA
        
        if "mock" in dataset_name:
            dataset_key='mock'
        else:
            dataset_key=dataset_name
        
        if inj_args['which_injections']=='GWTC-2':
            fnameInj = os.path.join(inj_path, dataset_name, fnames_inj[dataset_key] )
        elif inj_args['which_injections']=='GWTC-3':
            fnameInj = os.path.join(inj_path, dataset_name, fnames_inj_3[dataset_key] )
        #else:
        #    raise ValueError('which_injections you entered %s' %which_injections)
        
        
        if dataset_key=='mock':
            #Data = GWMockData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, Tobs=Tobs)
            if 'SNR_th' in inj_args.keys():
                snr_th = inj_args['SNR_th']
            else: snr_th=None
            fnameInj = os.path.join(inj_path, injections_name, fnames_inj[dataset_key] )
            injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Tobs, snr_th=snr_th)
        elif dataset_name=='O3a':
            #Data = O3aData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            if injections_name is None:
                injData = O3aInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
            else:
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5') #fnames_inj[dataset_key] )
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=183.375/365., snr_th=snr_th )
                         
        elif dataset_name=='O1O2':
            #Data = O1O2Data(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            if injections_name is None:
                injData = O1O2InjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
            else:
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5')
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=(129+267)/365., snr_th=snr_th)
        
        elif dataset_name=='O3b':
            #Data = O3bData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, percSamplesUse=percSamplesUse, dist_unit=dist_unit, **data_args)
            if injections_name is None:
                injData = O3bInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
                #raise ValueError('LVC injections are not supported for O3b. Specify a name for the injections')
            else:
                fnameInj = os.path.join(inj_path, dataset_name, injections_name, 'selected.h5')
                if 'SNR_th' in inj_args.keys():
                    snr_th = inj_args['SNR_th']
                else: snr_th=None
                injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=147.083/365. , snr_th=snr_th )
        else:
            raise ValueError('Dataset name not valid')
        
        return  injData

