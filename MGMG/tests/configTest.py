#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:27:44 2021

@author: Michi
"""

param='H0'
fout='testO3H01'

nObsUse=None

events_use = {'use': None, #['GW190521', 'GW190521_074359'],
          'not_use': ['GW170817', 'GW190814', 'GW190425', 'GW190426', 'GW190719', 'GW190909', 'GW190426_152155', 'GW190719_215514', 'GW190909_114149']
          
          }
nSamplesUse=100
nInjUse=100
npoints=5
data=['O3a','O1O2'] # O3a mock
massf= 'broken_pow_law' #'broken_pow_law' #'smooth_pow_law'
spindist='gauss' # 'gauss'
