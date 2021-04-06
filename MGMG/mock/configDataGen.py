#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:42:29 2021

@author: Michi
"""


#fout='mockTruncated'




###############################################################################
# CONFIGURE THE POPULATIONS
###############################################################################

populations = { 'astro' : { 'mass_function': 'broken_pow_law' , #'smooth_pow_law',
                           'spin_distribution': 'skip',
                           'rate': 'simple_pow_law'
    
                            }
    
    }


# Any extra argument to the mass, spin distributions and rate evolution
# The units for the rate will be passed automatically. No need to put them here
mass_args={} #{'normalization': 'integral'} # pivot
spin_args={}
rate_args={}



lambdaBase = {'R0': 25. , 'lambdaRedshift': 2. , 'deltam': 5., 'b': 0.4, 'mh':90. } # with these choices, m_break = 39


nsamples = 100000
zmax=2.5
rho_th=8.

psd_name= 'aLIGODesignSensitivityP1200087'  # aLIGOEarlyHighSensitivityP1200087 #aLIGODesignSensitivityP1200087
psd_path=None#'/Users/Michi/Dropbox/Local/Physics_projects/MGMG/data/detectors/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt' 
#O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt_optimal_snr_IMRPhenomXAS.h5
# O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt

from_file=False

approximant='IMRPhenomXAS'

tot_time_yrs=5.
duty_cycle=1.

time_steps=[1/12., 6/12., 1., 2., 3., 4., ] # yrs

seed=151157


# This if generating data
Nsamples=4000


# This if generating injections
chunk_size = int(1e05)
N_desired = 1000000




