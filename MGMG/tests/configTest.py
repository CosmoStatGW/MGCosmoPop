#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:27:44 2021

@author: Michi
"""

param='mh'
fout='testO3mh4'
nObsUse=None

O3_use = {'use': None, #['GW190521', 'GW190521_074359'],
          'not_use': ['GW190521']
          
          }
nSamplesUse=500
nInjUse=None
npoints=5
data='O3a' # O3a mock
massf= 'broken_pow_law' #'broken_pow_law' #'smooth_pow_law'