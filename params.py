#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:51:38 2021

@author: Michi
"""

import Globals
import numpy as np
import Globals
import astropy.units as u

class Params(object):

    def __init__(self, dataset_name ):
        
        self.dataset_name=dataset_name
        
        if dataset_name=='mock':
            
            self.allParams = [ 'H0', 'Om0', 'w0', 'Xi0', 'n', 'R0', 'lambdaRedshift', 'alpha', 'beta', 'ml', 'sl', 'mh', 'sh' ]
        
            self.trueValues = {'H0':Globals.H0GLOB,
                               'Om0':Globals.Om0GLOB,
                               'w0':-1.,
                           'Xi0':1.0, 
                           'n':1.91, 
                           'R0': 64.4, #60 , # Gpc^-3 yr^-1
                           'lambdaRedshift':3.0,
                           'alpha':0.75,
                           'beta':0.0, 
                           'ml':5.0, 
                           'sl':0.1, 
                           'mh':45.0,
                           'sh':0.1}
        
            self.names = {'H0':r'$H_0$', 
                          'Om0':r'$\Omega_{\rm {m,}0 }$',
                          'w0':r'$w_{0}$',
                           'Xi0':r'$\Xi_0$', 
                           'n':r'$n$', 
                           'R0':r'$R_0$', 
                           'lambdaRedshift':r'$\lambda$',
                           'alpha':r'$\alpha$',
                           'beta':r'$\beta$', 
                           'ml':r'$M_l$', 
                           'sl':r'$\sigma_l$', 
                           'mh':r'$M_h$',
                           'sh':r'$\sigma_h$'}
             
            if Globals.which_unit==u.Mpc:
                #self.trueValues['logR0']-=9*np.log(10)
                #print('New fiducial value for logR0 using yr^-1 Mpc^-3: %s' %self.trueValues['logR0'])
                self.trueValues['R0']*=1e-09
                print('New fiducial value for R0 using yr^-1 Mpc^-3: %s' %self.trueValues['R0'])
                 
    
        
        else:
            raise NotImplementedError('only mock dataset available.')
            
            
    def get_expected_values(self, params):
        if self.dataset_name=='mock':
            return  [self.trueValues[param] for param in params]
        
    def get_true_values(self, params):
        if self.dataset_name=='mock':
            return  [self.trueValues[param] for param in params]
        else:
            return None
        
    def get_labels(self, params):
        
        return  [self.names[param] for param in params]
        



class PriorLimits(object):
    
    def __init__(self,):
        
        self.limInf = {'H0': 20, 
                           'Xi0':0.1, 
                           'Om0':0.05,
                           'w0':-2,
                           'n':0, 
                           'R0': 0.1, # Gpc^-3 yr^-1
                           'lambdaRedshift':-15,
                           'alpha':-5,
                           'beta':-5, 
                           'ml':2, 
                           'sl':0.01, 
                           'mh':20,
                           'sh':0.01}
        self.limSup= {'H0':140, 
                      'Om0':1.,
                           'w0':-0.1,
                           'Xi0':10, 
                           'n':10, 
                           'R0': 1000,
                           'lambdaRedshift':10,
                           'alpha':10,
                           'beta':10, 
                           'ml':20, 
                           'sl':1, 
                           'mh':200,
                           'sh':1}
          
        
        self.logVals = {'H0': lambda x: 0., 
                           'Xi0': lambda x: 0. , 
                           'Om0': lambda x: 0.,
                           'w0': lambda x: 0.,
                           'n': lambda x: 0., 
                           'lambdaRedshift': lambda x: 0.,
                           'alpha': lambda x: 0.,
                           'beta': lambda x: 0., 
                           'ml':lambda x: 0., 
                           'sl':lambda x: 0., 
                           'mh':lambda x: 0.,
                           'sh':lambda x: 0.,
                           'R0': lambda x: 0,      
        }
        if Globals.which_unit==u.Mpc:
            #self.limInf['logR0']-=9*np.log(10)
            #self.limSup['logR0']-=9*np.log(10)
            self.limInf['R0']*=1e-09
            self.limSup['R0']*=1e-09
            #print('New lower limit for logR0 using yr^-1 Mpc^-3: %s' %self.limInf['logR0'])
            #print('New upper limit for logR0 using yr^-1 Mpc^-3: %s' %self.limSup['logR0'])
            print('New lower limit for R0 using yr^-1 Mpc^-3: %s' %self.limInf['R0'])
            print('New upper limit for R0 using yr^-1 Mpc^-3: %s' %self.limSup['R0'])
        
        
    def _set_prior(self, param, priorType='flat', mu=None, sigma=None):
            
            if priorType=='flat':
                self.logVals[param] = lambda x: 0.
            elif priorType=='flatLog':
                print("Setting flat-in-log prior on %s " %(param))
                self.logVals[param] = lambda x: -np.log(x)
            elif priorType=='gauss':
                print("Setting gaussian prior on %s with mu=%s, sigma=%s" %(param, mu, sigma))
                self.logVals[param] = lambda x: -np.log(sigma)-(x-mu)**2/(2*sigma**2)
            
          
    def set_priors(self, priors_types=None, priors_params=None):
            
            if priors_types is None:
                print('All priors flat.')
            else:
                for key in priors_types.keys():
                        
                    ptype=priors_types[key]
                    if ptype=='gauss':
                        self._set_prior(key, priorType=ptype,mu=priors_params[key]['mu'], sigma=priors_params[key]['sigma'] )
                    elif ptype=='flat' or ptype=='flatLog':
                        self._set_prior(key, priorType=ptype)
                    else:
                        raise ValueError('Supported priors are : flat, flatLog, gauss')
            
        