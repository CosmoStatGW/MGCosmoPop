#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:31:43 2021

@author: Michi
"""
import numpy as np



class Prior(object):
    '''
    Class implementing the prior. At the moment it only supports disjoint priors.
    contains a method logPrior that returns the sum of log priors for each variable
    in the inference
    
    '''
    
    def __init__(self, priorLimits, params_inference, priorNames, priorParams):
        '''
        

        Parameters
        ----------
        priorLimits : list
            list of max and min values for the prior range used for every parameter
            of the inference, in the correcto oder. 
            Example: for inference on H0, lambda:
                [ (20, 140) , (-10, 10)  ]
        params_inference : list
            list of  names of parameters used in the inference .
            Example: ['H0', 'lambdaRedshift']
        priorNames : dict
            Ditrionary specifying the type of prior used for each parameter.
            Supported so far are 'flat', 'flatLog', 'gauss'
            Example: gaussian prior on H0, flat on lambda:
                {'H0': gauss, 'lambdaRedshift':flat}
        priorParams : dict
            If any of the prior types requires parameters (e.g. mu and sigma for the gaussian)
            they are passed though this argument.
            Example: mu and sigma for gauss prior on H0
                {'mu': 67.9, 'sigma': 0.1 }

        

        '''
        self.priorLimits = priorLimits
        self.params_inference = params_inference
        self.priorNames = priorNames
        self.priorParams = priorParams
        
    
    
    def _logGauss(self, x, mu, sigma):
        '''
        gaussian prior
        '''
        if np.abs(x-mu)>7*sigma:
            return np.NINF
        return (-np.log(sigma)-(x-mu)**2/(2*sigma**2))
    
    def _flatLog(self, x):
        '''
        1/x prior
        '''
        return -np.log(x)
    
    
    def logPrior(self, Lambda_test):
        
        if np.isscalar(Lambda_test):
            limInf, limSup =  self.priorLimits[self.params_inference[0]]
            condition = limInf < Lambda_test < limSup
        else:
            condition = True
            for i,param in enumerate(self.params_inference):
                limInf, limSup = self.priorLimits[param]
                condition &= limInf < Lambda_test[i] < limSup
    
        if not condition:
            return np.NINF
        
        lp = 0
        for i,param in enumerate(self.params_inference):
            pname= self.priorNames[param]
            if np.isscalar(Lambda_test):
                    x = Lambda_test
            else:
                    x=Lambda_test[i]
            if pname=='flatLog':
                lp+=self._flatLog(x)
            elif pname=='gauss':
                mu, sigma = self.priorParams[param]['mu'], self.priorParams[param]['sigma']
                lp +=  self._logGauss( x, mu, sigma)
        return lp
        
        