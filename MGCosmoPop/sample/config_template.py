#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:20:27 2021

@author: Michi
"""




# This is to receive notifications on the progress of the mcmc via telegram
# One has to set up a telegram bot as in https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e
# Then enter the id and bot token here
telegram_notifications = False
telegram_id=''
telegram_bot_token=''



###############################################################################
# CONFIGURE THE POPULATIONS
###############################################################################

# check models.py for names
populations = { 'astro' : { 'mass_function': 'pow_law_peak', #'broken_pow_law' , #'smooth_pow_law', # pow_law_peak
                           'spin_distribution': 'skip',
                           'rate': 'astro-ph' # astro-ph simple_pow_law'
    
                            }
    
    }


normalized=False

# Any extra argument to the mass, spin distributions and rate evolution
# The units for the rate will be passed automatically. No need to put them here
mass_args={} #{'normalization': 'integral'} # pivot
spin_args={}
rate_args={'normalized':normalized}


###############################################################################
# CONFIGURE THE DATA AND UNITS
###############################################################################

dataset_names = ['O1O2', 'O3a', 'O3b' ] 

injections_names = ['injections_O1O2_IMRPhenomPv2_30k_alpha-11_beta-075_lamb-4_snr12', 'injections_O3a_IMRPhenomPv2_100k_alpha-11_beta-075_lamb-4_snr12', 'injections_O3b_IMRPhenomXPHM_100k_alpha-11_beta-075_lamb-4_snr12']

dist_unit = 'Gpc'


# O3 events to include (or not)
O3_use = {'use': None, #['GW190424_180648', 'GW190910_112807', 'GW190828_065509'] , #['GW190521', 'GW190521_074359'],
          'not_use': ['GW190814', ]
#[ 'GW170817', 'GW190814', 'GW190425', 'GW190426', 'GW190719', 'GW190909', 'GW190426_152155', 'GW190719_215514', 'GW190909_114149']
          
          }


SNR_th = 12.
FAR_th = 1./4.

verbose_inj=False
verbose_lik=False

safety_factor=50



###############################################################################
# CONFIGURE THE INFERENCE
###############################################################################

### The parameters have to be in the same order as they are listed in 
### the population object in the code !!! 

params_inference = ["H0", "Om", 'R0',  'alphaRedshift', 'betaRedshift', 'zp', 'lambdaPeak', 'alpha', 'beta', 'deltam', 'ml', 'mh', 'muMass', 'sigmaMass']

#"alpha1", "alpha2", "beta", "deltam", "ml",  "mh", "b"]

# "H0" "Om" "Xi0" "n"
# "R0" "lambdaRedshift"
# "alpha" "beta" "ml" "sl" "mh" "sh" 
#  "alpha1" "alpha2" "beta" "deltam" "ml"  "mh" "b" 
# "muEff", "sigmaEff", "muP", "sigmaP" 
# 'R0', 'alphaRedshift', 'betaRedshift', 'zp'

#PL Peak
#'lambdaPeak', 'alpha', 'beta','deltam', 'ml', 'mh', 'muMass', 'sigmaMass'

# Specify parameters that are kept fixed and their values 
params_fixed = {        'w0': -1. , 
                        'Xi0': 1. , 
                        'n' : 0., 
                        #'H0': 67.66,
                        #'Om': 0.311,
    }



priorLimits = { 'H0': (10, 200),  
               'Xi0': (0.1, 10) ,
                'Om': (0.05, 1.),
                'w0': (-3., 0.),
               'n':(0.,10.),
               
               
                'R0': (0., 1e02), 
               'lambdaRedshift': (-15, 15),
               
                'alphaRedshift': (0., 12.),
               'betaRedshift': (0., 6.),
               'zp':(0., 4.),
               
               
               
               'alpha': (1.5, 12. ),
               'beta': (-4, 12 ), 
                'ml': (2, 50),
               'sl':( 0.01 , 1),
               'mh':( 50, 200),
               'sh':(0.01, 1 ),
               
               
               'alpha1': (1.5, 12),                          
               'alpha2': (1.5, 12), 
               'deltam':  (0, 10),
               #'ml': {}, 
               #'mh': {}, 
               'b':  (0, 1) ,
               
               'muEff':(-1, 1.),
                'sigmaEff':(0.01, 1.),
                'muP':(0.01, 1.),
                'sigmaP':(0.01, 1.),
                
                # PLpeak mass
                'lambdaPeak': (0., 1.), 'muMass':(20., 50.), 'sigmaMass':(0.4, 10.)
               
               }


priorNames = {'H0' : 'flat',
              'Xi0': 'flatLog',
              'Om': 'flat',
               'w0': 'flat',
              'n':'flat',
              
               'R0': 'flat',
              'lambdaRedshift': 'flat',
              
              'alphaRedshift': 'flat',
               'betaRedshift': 'flat',
               'zp':'flat',
              
              'alpha': 'flat',
               'beta': 'flat', 
               'ml': 'flat',
               'sl':'flat',
               'mh':'flat',
               'sh':'flat',
               
               'alpha1':'flat',
               'alpha2':'flat',
               'deltam':'flat',
               'b':'flat',
               
               'muEff':'flat',
               'sigmaEff':'flat',
                'muP':'flat',
                'sigmaP':'flat',
              'lambdaPeak':'flat',  'muMass':'flat', 'sigmaMass':'flat'
               
               }


priorParams = { 'H0' : {'mu': 67.74, 'sigma': 0.6774},
                'Om' : {'mu': 0.3075, 'sigma': 0.003075}}



# Duration in yrs of the observing run , needed only if using mock data. Set None otherwise
Tobs=None


include_sel_uncertainty = True

seed=1312
nwalkers = 30
perc_variation_init=50
max_steps=100000


convergence_ntaus = 100
convergence_percVariation = 0.01


# How to handle parallelization: if 'mpi', the script should be launched with mpiexec
# (suitable for clusters)
# If 'pool' , it uses python multipricessing module, and we can specify the number of pools

parallelization='mpi'  # pool

# only needed if parallelization='pool'
nPools = 3 



###############################################################################
# For testing
###############################################################################

nObsUse=None
nSamplesUse=3000
percSamplesUse=None
nInjUse=None
