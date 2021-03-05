#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:06:04 2021

@author: Michi
"""
 
from abc import ABC, abstractmethod
import numpy as np
#from .. import utils
import scipy.stats as ss




def logdiffexp(x, y):
    '''
    computes log( e^x - e^y)
    '''
    return x + np.log1p(-np.exp(y-x))


class SelectionBias(ABC):
    
    def __init__(self, population, injData, params_inference ):
        
        self.injData=injData
        self.population=population
        self.params_inference = params_inference 


    @abstractmethod
    def logNdet(Lambda, **kwargs):
        pass



    
class SelectionBiasInjections(SelectionBias):
    
    '''
    Logic for computing the selection effects
    '''
    
    def __init__(self, population, injData, params_inference, get_uncertainty=True ):
        ''' 

        Parameters
        ----------
        population: obj of type AllPopulations 
            
    
        injData: Data object . Contains the injections data. Shoulf have attributes:
                log_weights_sel' : [array of log_p_draw]
                 'logN_gen': number of injections 
                 'condition': if any, some condition to filter injections
           

        '''
        
        
        self.get_uncertainty=get_uncertainty
        SelectionBias.__init__(self, population, injData, params_inference)
    
    
    def _get_mass_redshift(self, Lambda):
        
        LambdaCosmo, LambdaAllPop = self.population._split_params(Lambda)
        H0, Om0, w0,  Xi0, n = self.population.cosmo._get_values(LambdaCosmo, ['H0', 'Om', 'w0','Xi0', 'n'])
        
        z = self.population.cosmo.z_from_dLGW_fast(self.injData.dL, H0, Om0, w0, Xi0, n)
        m1 = self.injData.m1z / (1 + z)    
        m2 = self.injData.m2z / (1 + z)
        
        return m1, m2, z
    
    
    def _getSpins(self, ):
        return self.injData.chiEff
    
    def _getTobs(self):
        return self.injData.Tobs
    
    
    def logNdet(self, Lambda_test, verbose=False, Nobs = None):
        
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        
        m1, m2, z = self._get_mass_redshift(Lambda)
        chiEff = self._getSpins()
        Tobs = self._getTobs()
    
        
        logdN =  np.where( self.injData.condition, self.population.log_dN_dm1zdm2zddL(m1, m2, z, chiEff, Tobs, Lambda),  np.NINF) 
        logdN -= self.injData.log_weights_sel
        
        logMu = np.logaddexp.reduce(logdN) - self.injData.logN_gen
        
        if np.isnan(logMu):
            raise ValueError('NaN value for logMu. Values of Lambda: %s' %( str(Lambda) ) )
        
        mu = np.exp(logMu)
        if not self.get_uncertainty:
            return mu, np.NaN
        
        logs2 = ( np.logaddexp.reduce(2*logdN) -2*self.injData.logN_gen)#.astype('float128')
        logSigmaSq = logdiffexp( logs2, 2.0*logMu - self.injData.logN_gen )
        
        muSq = np.exp(2*logMu)
        SigmaSq = np.exp(logSigmaSq)
        
        if Nobs is not None and verbose:
            Neff = muSq/SigmaSq #np.exp( 2.0*logMu - logSigmaSq)
            if Neff < 4 * Nobs:
                print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Values of Lambda: %s' %str(Lambda))
        
        Sigma = np.sqrt(SigmaSq)
        
        ## Effects of uncertainty on selection effect and/or marginalisation over total rate
        ## Adapted from 1904.10879
        error = SigmaSq/2-ss.norm(loc=mu, scale=Sigma ).logsf(0)+ss.norm(loc=mu-SigmaSq, scale=Sigma).logsf(0)
        
        return mu, error
    
    