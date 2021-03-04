#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:15:52 2021

@author: Michi
"""

from ..ABSpopulation import RateEvolution
import numpy as np
import astropy.units as u


class PowerLawRateEvolution(RateEvolution):
    
    def __init__(self):
        RateEvolution.__init__(self)
        
        self.params = ['R0', 'lambdaRedshift']
        self.baseValues = {'R0': 60, #60 , # Gpc^-3 yr^-1
                           'lambdaRedshift':3.0,}
        self.n_params = len(self.params)
        
        self.names = { 'R0':r'$R_0$', 
                           'lambdaRedshift':r'$\lambda$',}
        

    
    def set_rate_units(self, unit):
        if unit==u.Mpc:
            self.baseValues['R0']*=1e-09
            print('New fiducial value for R0 using yr^-1 Mpc^-3: %s' %self.baseValues['R0'])
        
    
    def log_dNdVdt(self, theta_rate, lambdaBBHrate):
        '''
        Evolution of total rate with redshift: R0*(1+z)**lambda
        '''
        R0, lambdaRedshift = lambdaBBHrate
        z=theta_rate
        return np.log(R0)+lambdaRedshift*np.log1p(z)
    
    


class DummyRateEvolution(RateEvolution):
    
    def __init__(self):
        RateEvolution.__init__(self)
        
        self.params = ['R1', 'gamma']
        self.baseValues = {'R1': 1, #60 , # Gpc^-3 yr^-1
                           'gamma': -1 ,}
        self.n_params = len(self.params)
        
        self.names = {
                           'R1':r'$R_1$', 
                           'gamma':r'$\gamma$',}
        

    def log_dNdVdt(self, theta_rate, lambdaBBHrate):
        '''
        Evolution of total rate with redshift: R0*(1+z)**lambda
        '''
        R1, gamma = lambdaBBHrate
        z=theta_rate
        return np.log(R1)+gamma*np.log1p(z)
    
    