#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:20:27 2021

@author: Michi
"""




# This is to receive notifications on the progress of the mcmc via telegram
# One has to set up a telegram bot as in https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e
# Then enter the id and bot token here
telegram_notifications = True
telegram_id='158370570'
telegram_bot_token='1580787370:AAGyCFRjxRTt4Dsg8S4XaKwNkBDjwoYuZIQ'



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


###############################################################################
# CONFIGURE THE DATA AND UNITS
###############################################################################

dataset_names = ['mock_BPL_5yr_aLIGOdesignSensitivity_MG', ] # 'O3a' # mock_BPL_5yr_aLIGOdesignSensitivity_MG  #mock_BPL_5yr_aLIGOdesignSensitivity #O1O2 # O3a
dist_unit = 'Gpc'


# O3 events to include (or not)
O3_use = {'use': None, #['GW190424_180648', 'GW190910_112807', 'GW190828_065509'] , #['GW190521', 'GW190521_074359'],
          'not_use': ['GW170817', 'GW190814', 'GW190425', 'GW190426', 'GW190719', 'GW190909', 'GW190426_152155', 'GW190719_215514', 'GW190909_114149']
          
          }


###############################################################################
# CONFIGURE THE INFERENCE
###############################################################################

### The parameters have to be in the same order as they are listed in 
### the population object in the code !!! 

params_inference = [ "H0", "Om", "Xi0", "n", "R0", "lambdaRedshift", "alpha1", "alpha2", "beta", "deltam", "ml",  "mh", "b"]

# "H0" "Om" "Xi0" "n"
# "R0" "lambdaRedshift"
# "alpha" "beta" "ml" "sl" "mh" "sh" 
#  "alpha1" "alpha2" "beta" "deltam" "ml"  "mh" "b" 
# "muEff", "sigmaEff", "muP", "sigmaP" 


# Specify parameters that are kept fixed and their values 
params_fixed = {   'w0': -1. , 
                        #'Xi0': 1. , 
                        #'n' : 0.,   
                        #'lambdaRedshift ':0. ,                                     
    }



priorLimits = { 'H0': (20, 140),  
               'Xi0': (0.1, 10) ,
               'Om': (0.05, 1),
               'w0': (-2, -0.5),
               'n':(0.,10.),
               'R0': (1e-01, 1e03), 
               'lambdaRedshift': (-10, 10),
               
               'alpha': (-5, 10 ),
               'beta': (-4, 12 ), 
               'ml': (2, 10),
               'sl':( 0.01 , 1),
               'mh':( 30, 100),
               'sh':(0.01, 1 ),
               
               
               'alpha1': (-4, 12),                          
               'alpha2': (-4, 12), 
               'deltam':  (0, 10),
               #'ml': {}, 
               #'mh': {}, 
               'b':  (0, 1) ,
               
               'muEff':(-1, 1.),
                'sigmaEff':(0.01, 1.),
                'muP':(0.01, 1.),
                'sigmaP':(0.01, 1.),
               
               }


priorNames = {'H0' : 'gauss',
              'Xi0': 'flat',
              'Om': 'gauss',
               'w0': 'flat',
              'n':'flat',
               'R0': 'flatLog',
              'lambdaRedshift': 'flat',
              
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
                       
               
               }


#priorParams = { 'H0' : {'mu': 67.74, 'sigma': 0.6774},
#                'Om' : {'mu': 0.3075, 'sigma': 0.003075}}

priorParams = { 'H0' : {'mu': 67.66, 'sigma': 0.42},
                'Om' : {'mu': 0.311, 'sigma': 0.056}}


# Duration in yrs of the observing run , needed only if using mock data
Tobs=5.


include_sel_uncertainty = True

seed=1312
nwalkers = 72
perc_variation_init=60
max_steps=300


convergence_ntaus = 100
convergence_percVariation = 0.01


# How to handle parallelization: if 'mpi', the script should be launched with mpiexec
# (suitable for clusters)
# If 'pool' , it uses python multipricessing module, and we can specify the number of pools

parallelization='pool'  # pool

# only needed if parallelization='pool'
nPools = 3 



###############################################################################
# For testing
###############################################################################

nObsUse=10
nSamplesUse=100
nInjUse=100