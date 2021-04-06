#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:55:12 2021

@author: Michi
"""

from abc import ABC, abstractmethod
import numpy as np
#import scipy.stats as ss
#from scipy.integrate import cumtrapz
#from scipy.interpolate import interp1d


########################################################################
########################################################################

# Abstract logic for a single population . Should implement the differential merger rate dR/dm1dm2

class Population(ABC):
    
    def __init__(self, ):
        self.params = []
        self.baseValues = {}
        self.n_params = 0
        self.names={}


    @abstractmethod
    def log_dR_dm1dm2(self, theta, LambdaPop):
        '''
        Every population should return the differential rate
        '''
        pass
    
    @abstractmethod
    def _set_values(self, params_values):
        # How to set the values of the population parameters
        # Must change them also in other objects that enter the population !
        pass



class RateEvolution(ABC):
    
    def __init__(self, ):
        self.params = []
        self.baseValues = {}
        self.n_params = 0
        self.names={}


    @abstractmethod
    def log_dNdVdt(self, theta, LambdaPop):
        pass
    
    
    def _set_values(self, values_dict):
    # update value also in this object
            #print('rate basevalues: %s' %str(self.baseValues))
            for key, value in values_dict.items():
                if key in self.baseValues:
                    self.baseValues[key] = value
                    print('Setting value of %s to %s in %s' %(key, value, self.__class__.__name__))

    




class BBHDistFunction(ABC):
    
    ''''
    Abstract base class for mass and spin distributions
    '''
    
    def __init__(self):
        self.params = []
        self.baseValues = {}
        self.n_params = 0
        self.names={}
    
    #@abstractmethod
    #def _get_normalization(lambdaBBHmass):
    #    '''Normalization of p(m1, m2 | Lambda )'''
    #    pass
    
    
    @abstractmethod
    def logpdf(theta, lambdaBBHmass):
        '''p(m1, m2 | Lambda ), or p(chi1, chi2 | Lambda ) normalized to one'''
        pass
    
    
    def _sample_pdf(self, nSamples, pdf, lower, upper):
        res = 100000
        eps=1e-02
        x = np.linspace(lower+eps, upper-eps, res)
        cdf = np.cumsum(pdf(x))
        cdf /= cdf[-1]
        return np.interp(np.random.uniform(size=nSamples), cdf, x)
        #ps = pdf(x)
        #cms = cumtrapz(ps, x, initial=0)
        #icdfm1 = interp1d(cms, x)
        #def draw_m1():
        #    return icdfm1(cms[-1]*np.random.rand())
        #return np.array([draw_m1() for i in range(nSamples)])
        
    
    def _sample_vector_upper(self, pdf, lower, upper):
        nSamples = len(upper)
        res = 100000
        eps=1e-02
        x = np.linspace(lower+eps, upper.max()-eps, res)
        cdf = np.cumsum(pdf(x))
        cdf = cdf / cdf[-1]
        probTilUpper = np.interp(upper, x, cdf)
        return np.interp(probTilUpper*np.random.uniform(size=nSamples), cdf, x)
    

    def _set_values(self, values_dict):
            #print('BBhpop basevalues: %s' %str(self.baseValues))
            for key, value in values_dict.items():
                if key in self.baseValues:
                    self.baseValues[key] = value
                    print('Setting value of %s to %s in %s' %(key, value, self.__class__.__name__))
                


