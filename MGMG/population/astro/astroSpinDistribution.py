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
from scipy.special import erfc
########################################################################
# SPIN DISTRIBUTION
########################################################################



def trunc_gaussian_logpdf(x, mu = 1, sigma = 1, lower = 0, upper=100):

    where_compute= (x>lower) & (x<upper)
    
    pdf=np.empty_like(x)
    pdf[~where_compute]=np.NINF
    x=x[where_compute]
    
    Phialpha = 0.5*erfc(-(lower-mu)/(np.sqrt(2)*sigma))
    Phibeta = 0.5*erfc(-(upper-mu)/(np.sqrt(2)*sigma))
    
    pdf[where_compute] = -np.log(2*np.pi)/2-np.log(sigma)-np.log(Phibeta-Phialpha) -(x-mu)**2/(2*sigma**2)
    
    return pdf



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
        
        
        where_compute=~np.isnan(chiP)
        
        #pdftot=np.empty_like(chiEff)
        pdf2=np.empty_like(chiEff)
        
        pdf2[~where_compute]=0.
        
        chiP, sigmaP = chiP[where_compute], sigmaP[where_compute]
        
        pdf1 = trunc_gaussian_logpdf(chiEff, lower=self.minChiEff, upper=self.maxChiEff, mu=muEff, sigma=sigmaEff ) #get_truncnorm(self.minChiEff, self.maxChiEff, muEff, sigmaEff ).logpdf(chiEff)
        
        # Put zero (i.e. ignore the effect) when chi_p is not available - this is used when computing selection effects, for which we don't have chi_p
        pdf2[where_compute] =  trunc_gaussian_logpdf(chiP, lower=self.minChiP, upper=self.maxChiP, mu=muP, sigma=sigmaP )#get_truncnorm(self.minChiP, self.maxChiP, muP, sigmaP ).logpdf(chiP)
        
        pdftot = pdf1+pdf2
        
        return pdftot
    
    
    
    
    