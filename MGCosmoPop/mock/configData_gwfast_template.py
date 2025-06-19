#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:25:11 2022

@author: Michi
"""



###############################################################################
# CONFIGURE THE POPULATIONS FOR MGCOSMOPOP
###############################################################################

populations = { 'astro' : { 'mass_function': 'pow_law_peak', #'trunc_pow_law' , #'smooth_pow_law', pow_law_peak
                           'spin_distribution': 'skip', #'flat', #'skip',
                           'rate': 'astro-ph' , #'simple_pow_law' #'astro-ph'
    
                            }
    
    }


# Any extra argument to the mass, spin distributions and rate evolution
# The units for the rate will be passed automatically. No need to put them here
mass_args={} #{'normalization': 'integral'} # pivot
spin_args={}
rate_args={}

lambdaBase = { 'alpha': 3.4,
                 'beta': 1.1,
                 'mh': 87,
                 'ml': 5.1,
                 'lambdaPeak': 0.039,
                 'muMass': 34,
                 'sigmaMass': 3.6,
                 'deltam': 4.8,

 
                  'gamma':2.7,    
                'kappa':3., 
                    'zp':2.,
                   'R0':17, 
    
                  
                    'H0': 67.7,
                     'Om': 0.31, 

                 #'muChi': 0.3, 'varChi':0.03,  'zeta': 0.76,  'sigmat':0.8,
             
             }



# 'R0', 'alphaRedshift', 'betaRedshift', 'zp'

zmax=20.



###############################################################################
# CONFIGURE THE DETECTOR NETWORK
###############################################################################

gwastpath = '/Users/Michi/Dropbox/Local/Physics_projects/gwfast_dev/'


wf_model = 'IMRPhenomHM'

#snr_th = 12

# snr threshold 
snr_th_dets = { 'L1': -1, 'net':8 }

fmin = 10
fmax = None

net = { 'L1': { 'lat': 90. ,
                     'long': 0. ,
                     'xax': 45.,
                     'shape':'L',
               
            } 
      }

psds = { 'L1': 'LVK_ObsScenario/AplusDesign.txt', }
netfile = None

duty_factor = 0.7

rot = 0

lalargs = None

seeds = [None, ] #[ 171155, ]

condition = 'and'



###############################################################################
# CONFIGURE THE OPTIONS FOR SAMPLING
###############################################################################



seed=151157
add_noise = True # leave it true


###############################################################################
# For datasets
###############################################################################

tot_time_yrs = 5
time_steps = [1,2,3,4,5]
Nsamples = 5000
samples_step = 500

eta_scatter=5e-03 
mc_scatter = 3e-02
theta_scatter = 5e-02

###############################################################################
# For injections
###############################################################################


chunk_size = int(2e04)
N_desired = 100000






