#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:58:40 2021

@author: Michi
"""

from ..ABSpopulation import BBHDistFunction
import numpy as np
#from numpy.linalg import inv, det
from scipy.stats import truncnorm #, multivariate_normal

########################################################################
# SPIN DISTRIBUTION
########################################################################

class DummySpinDist(BBHDistFunction):
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
    
    def logpdf(theta, lambdaBBHmass):
        #chi1, chi2 =theta
        return np.zeros(theta[0].shape[0])
    

def get_truncnorm(a, b, mu, sigma):
    
    aa, bb = (a - mu) / sigma, (b - mu) / sigma
    
    return truncnorm( aa, bb, loc=mu, scale=sigma)


    
    
class GaussSpinDist(BBHDistFunction):
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
        self.params = ['muEff', 'sigmaEff', 'muP', 'sigmaP', ] #'rho' ] # For the moment we ignore correlation
        
        self.baseValues = {
                           
                           'muEff':0.06,
                           'sigmaEff':0.12, 
                           'muP':0.21, 
                           'sigmaP':0.09, 
                           'rho':0.,
                           }
        
        self.names = {
                           'muEff':r'$\mu_{eff}$',
                           'sigmaEff':r'$\sigma_{eff}$', 
                           'muP':r'$\mu_{p}$', 
                           'sigmaP':r'$\sigma_{p}$',
                           'rho':r'$\rho$'}
         
        self.n_params = len(self.params)
    
        self.maxChiEff = 1
        self.minChiEff = -1
        
        self.maxChiP = 1
        self.minChiP = 0
        
        print('Gaussian spin distribution base values: %s' %self.baseValues)
    
    
    def logpdf(self, theta, lambdaBBHspin):
        
        chiEff, chiP = theta
        muEff, sigmaEff, muP, sigmaP, = lambdaBBHspin
        
        #mean = np.array([muEff, muP])
        #C = np.array( [[sigmaEff**2, rho*sigmaEff*sigmaP ], [rho*sigmaEff*sigmaP , sigmaP**2]]  )
        
        #logpdf = -np.log(2*np.pi)-0.5*np.log(det(C))-0.5*(theta-mean).dot(inv(C)).dot(theta-mean)
        #logpdf = multivariate_normal.logpdf(theta, mean=mean, cov=C )
        
        pdf1 = get_truncnorm(self.minChiEff, self.maxChiEff, muEff, sigmaEff ).logpdf(chiEff)
        
        # Put zero (i.e. ignore the effect) when chi_p is not available - this is used when computing selection effects, for which we don't have chi_p
        pdf2 = np.where( np.isnan(chiP), 0. , get_truncnorm(self.minChiP, self.maxChiP, muP, sigmaP ).logpdf(chiP) )
        
        
        return pdf1+pdf2#.logpdf(chiP)
    
    
    
    
    