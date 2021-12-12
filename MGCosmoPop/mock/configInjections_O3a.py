#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:25:04 2021

@author: Michi
"""


base_psd_path = '/home/mancarel/MGMG/data/detectors/' #'/Users/Michi/Dropbox/Local/Physics_projects/MGCosmoPop/data/detectors/'
#'/home/mancarel/MGMG/data/detectors/'



###############################################################################
# CONFIGURE THE POPULATIONS
###############################################################################

populations = { 'astro' : { 'mass_function': 'trunc_pow_law' , #'smooth_pow_law',
                           'spin_distribution': 'skip',
                           'rate': 'simple_pow_law' #'astro-ph'

                            }
    
    }


# Any extra argument to the mass, spin distributions and rate evolution
# The units for the rate will be passed automatically. No need to put them here
mass_args={} #{'normalization': 'integral'} # pivot
spin_args={}
rate_args={}

lambdaBase = {'R0': 20. ,
              'lambdaRedshift':4.,                                                                                                                                         
              #'alphaRedshift': 0., 'betaRedshift':0.5, 'zp':2 ,
              'mh': 500., 'ml':2.,
              'alpha': 1.1 , 'beta': 0.75 }



# 'R0', 'alphaRedshift', 'betaRedshift', 'zp'                                                                                                                               


zmax=5.



###############################################################################
# CONFIGURE THE DETECTOR NETWORK
###############################################################################


# This configuration is for O3a

detectors = { 'L1': { 'lat':30.563,
                     'long':-90.774,
                     'xax':198.,
                     'shape':'L',
                     'duty_cycle': 0.758,
                     'detector_args': {'psd_path': base_psd_path+'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt' ,
                               'from_file':True,
                               'psd_name':None,
                               'approximant': 'IMRPhenomPv2' #'IMRPhenomXAS' #'IMRPhenomXAS'
                            },
                     'interpolator_args':{  }
    
                    },
             
             'H1': { 'lat':46.455,
                     'long':-119.408,
                     'xax':126.,
                     'shape':'L',
                     'duty_cycle': 0.712,
                     'detector_args': {'psd_path':  base_psd_path+'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt' ,
                               'from_file':True,
                               'psd_name':None,
                               'approximant': 'IMRPhenomPv2' #'IMRPhenomXAS' #'IMRPhenomXAS'
                            },
                     'interpolator_args':{  }
    
                    },
             
             'Virgo': { 'lat':43.631,
                     'long':10.504,
                     'xax':71.,
                     'shape':'L',
                     'duty_cycle': 0.763,
                     'detector_args': {'psd_path':  base_psd_path+'O3-V1_sensitivity_strain_asd.txt' ,
                               'from_file':True,
                               'psd_name':None,
                               'approximant': 'IMRPhenomPv2' #'IMRPhenomXAS' #'IMRPhenomXAS'
                            },
                     'interpolator_args':{ }
    
                    }

    
    
    
            }



###############################################################################
###############################################################################
# Values

# LIGO H: latLH=46.455 -- longLH=-119.408 -- xLH=126.
# LIGO L: latLL=30.563 -- longLL=-90.774 -- xLL=198.
# Virgo: latVi=43.631 -- longVi= 10.504 -- xVi=71.


# O3 H1 psd : O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt
# O3 L1 psd : O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt
# O3 V1 psd : O3-V1_sensitivity_strain_asd.txt

# O2 H1 psd : 2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt
# O2 L1 psd : 2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt
# O2 V1 psd : Hrec_hoft_V1O2Repro2A_16384Hz.txt

# O2 duty cycles: H1 0.617, L1 0.606 , V1  0.85
# O3 duty cycles: H1 0.712, L1 0.758 , V1  0.763



###############################################################################
###############################################################################
# Sources

# O3a reprentative strain : https://dcc.ligo.org/LIGO-P2000251/public
# O2 reprentative strain :   all https://dcc.ligo.org/P1800374/public/
#                            H1 https://dcc.ligo.org/LIGO-G1801950/public
#                            L1 https://dcc.ligo.org/LIGO-G1801952/public 
#                            V1 https://dcc.ligo.org/P1800374/public/

# O3a duty cycles: https://www.gw-openscience.org/detector_status/O3a/
# O2: https://www.gw-openscience.org/summary_pages/detector_status/O2/
# O2 Virgo : https://www.virgo-gw.eu/O2.html
    



rho_th=10.

###############################################################################
# SPECIFY NUMBER OF DESIRED INJECTIONS
###############################################################################



seed=26986

chunk_size = int(1e05)
N_desired = 100000




