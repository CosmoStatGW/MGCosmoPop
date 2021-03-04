#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:54:39 2021

@author: Michi
"""

from ..ABSpopulation import BBHDistFunction
import numpy as np
import scipy.stats as ss


########################################################################
# MASS DISTRIBUTION
########################################################################
class AstroSmoothPowerLawMass(BBHDistFunction):
    
    '''
    Mass distribution  - Power law with smooth transition to zero at the edges; from 1908.09084
    '''
    
    
    def __init__(self, normalization='integral' ):
        
        BBHDistFunction.__init__(self)
        
        self.normalization=normalization
        self.params = ['alpha', 'beta', 'ml', 'sl', 'mh', 'sh' ]
        
        self.baseValues = {
                           
                           'alpha':0.75,
                           'beta':0.0, 
                           'ml':5.0, 
                           'sl':0.1, 
                           'mh':45.0,
                           'sh':0.1}
        
        self.names = {
                           'alpha':r'$\alpha$',
                           'beta':r'$\beta$', 
                           'ml':r'$M_l$', 
                           'sl':r'$\sigma_l$', 
                           'mh':r'$M_h$',
                           'sh':r'$\sigma_h$'}
         
        self.n_params = len(self.params)
    
    
    def _logpdfm1(self, m1, alpha, ml, sl, mh, sh ):
        
        logp = np.log(m1)*(-alpha)+self._logf_smooth(m1, ml=ml, sl=sl, mh=mh, sh=sh)
        
        return logp
    
    def _logpdfm2(self, m2, beta, ml, sl, mh, sh ):
        
        logp = np.log(m2)*(beta)+self._logf_smooth(m2, ml=ml, sl=sl, mh=mh, sh=sh)
        
        return logp
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        
        logpdfMass = self._logpdfm1(m1,alpha, ml, sl, mh, sh ) + self._logpdfm2(m2, beta, ml, sl, mh, sh )
        
        if self.normalization=='integral':
            logNorm = -np.log(self._get_normalization(lambdaBBHmass))
        elif self.normalization=='pivot':
            logNorm = alpha*np.log(30)-beta*np.log(30)-2*self._logf_smooth(30, ml=ml, sl=sl, mh=mh, sh=sh)-2*np.log(30)
        
        return logpdfMass+logNorm
        
       
    def _logf_smooth(self, m, ml=5, sl=0.1, mh=45, sh=0.1):
        return np.log(ss.norm().cdf((np.log(m)-np.log(ml))/sl))+np.log((1-ss.norm().cdf((np.log(m)-np.log(mh))/sh)))
       
    
    def _get_normalization(self, lambdaBBHmass):
        '''Normalization of p(m1, m2 | Lambda )'''
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        
        if (beta!=-1) & (alpha!=1) : 
            return (( mh**(-alpha+beta+2)- ml**(-alpha+beta+2) )/(-alpha+beta+2) - ml**(beta+1) *(mh**(-alpha+1)-ml**(-alpha+1))/(-alpha+1) )/(beta+1)
        else:
            raise ValueError 
    
    
    def sample(self, nSamples, lambdaBBHmass):
        alpha, beta, ml, sl, mh, sh = lambdaBBHmass
        mMin = 3
        mMax = 100
        
        pm1 = lambda x: np.exp(self._logpdfm1(x, alpha, ml, sl, mh, sh ))
        pm2 = lambda x: np.exp(self._logpdfm2(x, beta, ml, sl, mh, sh ))
        
        m1 = self._sample_pdf(nSamples, pm1, mMin, mMax)
        #m2 = self._sample_pdf(nSamples, pm2, mMin, mMax)
        m2 = self._sample_vector_upper(pm2, mMin, m1)
        
            
        return m1, m2
    
    

class AstroGaussMass(BBHDistFunction):
    
    '''
    Mass distribution  - Gaussian in the first mass flat in the second
    '''
    
    def __init__(self, ):
        
        BBHDistFunction.__init__(self)
        
        self.params = ['meanMass', 'sigmaMass']
        
        self.baseValues = {
                           
                           'meanMass':20,
                           'sigmaMass':10, 
                           }
        
        self.names = {
                           'meanMass':r'$\mu_{\rm mass}$',
                           'sigmaMass':r'$\sigma_{\rm mass}$', 
                           }
         
        self.n_params = len(self.params)
    
        
    
    
    def logpdf(self, theta, lambdaBBHmass):
        
        '''p(m1, m2 | Lambda ), normalized to one'''
        
        m1, m2 = theta
        meanMass, sigmaMass = lambdaBBHmass
        
        condition = (m2<m1) & (m2>5)
        if not condition:
            return np.NINF
    
        logpdfMass = (-np.log(sigmaMass)-(m1-meanMass)**2/(2*sigmaMass**2)) 
        
        return logpdfMass
    
    