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
    
     
class RateEvolution(ABC):
    
    def __init__(self, ):
        self.params = []
        self.baseValues = {}
        self.n_params = 0
        self.names={}


    @abstractmethod
    def log_dNdVdt(self, theta, LambdaPop):
        pass



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
        x = np.linspace(lower, upper, res)
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
        x = np.linspace(lower, upper.max(), res)
        cdf = np.cumsum(pdf(x))
        cdf = cdf / cdf[-1]
        probTilUpper = np.interp(upper, x, cdf)
        return np.interp(probTilUpper*np.random.uniform(size=nSamples), cdf, x)
    

    


