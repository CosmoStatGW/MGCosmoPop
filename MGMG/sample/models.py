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

from population.astro.astroMassDistribution import AstroSmoothPowerLawMass, BrokenPowerLawMass, TruncPowerLawMass
from population.astro.astroSpinDistribution import DummySpinDist, GaussSpinDist
from population.astro.rateEvolution import PowerLawRateEvolution
from population.astro.astroPopulation import AstroPopulation
from cosmology.cosmo import Cosmo
from population.allPopulations import AllPopulations

from dataStructures.mockData import GWMockData, GWMockInjectionsData
from dataStructures.O3adata import O3aData, O3InjectionsData
from dataStructures.O1O2data import O1O2Data, O1O2InjectionsData
#import astropy.units as u

#from posteriors.prior import Prior
#from posteriors.likelihood import HyperLikelihood

#from posteriors.selectionBias import SelectionBiasInjections
#from posteriors.posterior import Posterior




mass_functions = {  'smooth_pow_law': AstroSmoothPowerLawMass,
                      'broken_pow_law': BrokenPowerLawMass,
                      'trunc_pow_law': TruncPowerLawMass,
    
     }

spin_functions = {  'gauss': GaussSpinDist,
                      'skip': DummySpinDist
     }


rate_functions = { # 'gauss': AstroSmoothPowerLawMass(),
                  
                      'simple_pow_law': PowerLawRateEvolution
     }



fnames_data  = { 'mock': 'observations.h5',
                'O3a': '',
                'O1O2': ''
    
    
    }


fnames_inj  = { 'mock': 'selected.h5',
                'O3a': 'o3a_bbhpop_inj_info.hdf',
                'O1O2':'injections_O1O2an_spin.h5'
    
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



def build_model( populations, cosmo_args={}, mass_args={}, spin_args={}, rate_args={} ):
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
        
    # Collector of all populations
    allPops = AllPopulations(myCosmo)
    
    
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
            




def load_data(dataset_name, nObsUse=None, nSamplesUse=None, nInjUse=None, dist_unit=u.Gpc, data_args ={}, inj_args ={}, Tobs=None):
        
        
        ############################################################
        # DATA
        
        if "mock" in dataset_name:
            dataset_key='mock'
        else:
            dataset_key=dataset_name
        
        fname = os.path.join(Globals.dataPath, dataset_name, fnames_data[dataset_key])
        fnameInj = os.path.join(Globals.dataPath, dataset_name, fnames_inj[dataset_key] )
        
        if dataset_key=='mock':
            Data = GWMockData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, Tobs=Tobs)
            injData = GWMockInjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, Tobs=Tobs)
        elif dataset_name=='O3a':
            Data = O3aData(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, **data_args)
            injData = O3InjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
        elif dataset_name=='O1O2':
            Data = O1O2Data(fname,  nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, **data_args)
            injData = O1O2InjectionsData(fnameInj,  nInjUse=nInjUse, dist_unit=dist_unit, **inj_args)
        else:
            raise ValueError('Dataset name not valid')
        
        return Data, injData