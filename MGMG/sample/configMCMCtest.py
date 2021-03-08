#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:20:27 2021

@author: Michi
"""



fout='runXi01perc_n05TEST'


# This is to receive notifications on the progress of the mcmc via telegram
# One has to set up a telegram bot as in https://medium.com/@ManHay_Hong/how-to-create-a-telegram-bot-and-send-messages-with-python-4cf314d9fa3e
# Then enter the id and bot token here
telegram_notifications = False
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

params_inference = ['H0', 'Om', 'Xi0', 'R0', 'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh'] #'Xi0',


# Specify parameters that are kept fixed and their values 
params_fixed = {   'w0': -1. , 
                        #'Xi0': 1. , 
                        'n' : 0.5,                                         
    }



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


priorNames = {'H0' : 'gauss',
              'Xi0': 'flat',
              'Om': 'gauss',
               'w0': 'flat',
              'R0': 'flatLog',
              'lambdaRedshift': 'flat',
              'alpha': 'flat',
               'beta': 'flat', 
               'ml': 'flat',
               'sl':'flat',
               'mh':'flat',
               'sh':'flat'}


priorParams = { 'H0' : {'mu': 67.74, 'sigma': 0.6774},
                'Om' : {'mu': 0.3075, 'sigma': 0.003075}}



include_sel_uncertainty = True

seed=1312
nwalkers = 50
perc_variation_init=10
max_steps=10000


convergence_ntaus = 50
convergence_percVariation = 0.01


# How to handle parallelization: if 'mpi', the script should be launched with mpiexec
# (suitable for clusters)
# If 'pool' , it uses python multipricessing module, and we can specify the number of pools

parallelization='mpi'  # pool

# only needed if parallelization='pool'
nPools = nwalkers 


###############################################################################
# For testing
###############################################################################

nObsUse=10
nSamplesUse=100
nInjUse=100