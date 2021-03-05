#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:20:27 2021

@author: Michi
"""



fout='runLCDM'


# This is to receive notifications on the progress of the mcmc via telegram
# One has to set up a telegram bot as in https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e
# Then enter the id and bot token here
telegram_notifications = True
telegram_id='158370570'
telegram_bot_token='1580787370:AAGyCFRjxRTt4Dsg8S4XaKwNkBDjwoYuZIQ'



###############################################################################
# CONFIGURE THE POPULATIONS
###############################################################################

populations = { 'astro' : { 'mass_function': 'smooth_pow_law',
                           'spin_distribution': 'skip',
                           'rate': 'simple_pow_law'
    
                            }
    
    }


# Any extra argument to the mass, spin distributions and rate evolution
# The units for the rate will be passed automatically. No need to put them here
mass_args={'normalization': 'integral'} # pivot
spin_args={}
rate_args={}


###############################################################################
# CONFIGURE THE DATA AND UNITS
###############################################################################

dataset_name = 'mock' # 'O3a'
dist_unit = 'Gpc'


###############################################################################
# CONFIGURE THE INFERENCE
###############################################################################

### The parameters have to be in the same order as they are listed in 
### the population object in the code !!! 

params_inference = ['H0', 'Om', 'R0', 'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh'] #'Xi0',

priorLimits = { 'H0': (20, 140),  
               'Xi0': (0.1, 10) ,
               'Om': (0.05, 1),
               'w0': (-2, -0.5),
               'R0': (1e-03, 1e03), 
               'lambdaRedshift': (-10, 10),
               'alpha': (-5, 10 ),
               'beta': (-5, 10 ), 
               'ml': (2, 20),
               'sl':( 0.01 , 1),
               'mh':( 20, 100),
               'sh':(0.01, 1 )
               }


priorNames = {'H0' : 'flat',
              'Xi0': 'flat',
              'Om': 'flat',
               'w0': 'flat',
              'R0': 'flatLog',
              'lambdaRedshift': 'flat',
              'alpha': 'flat',
               'beta': 'flat', 
               'ml': 'flat',
               'sl':'flat',
               'mh':'flat',
               'sh':'flat'}


priorParams = { 'Om' : {'mu': 0.301, 'sigma': 0.002} }

include_sel_uncertainty = True

seed=1312
nwalkers = 22
perc_variation_init=10
max_steps=10000

nPools = nwalkers 

convergence_ntaus = 100
convergence_percVariation = 0.01

###############################################################################
# For testing
###############################################################################

nObsUse=None
nSamplesUse=None
nInjUse=None