#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:54:06 2021

@author: Michi
"""
import numpy as np


class HyperLikelihood(object):
    
    '''
    Implements the logic for computing the hyper-likelihood, i.e. the likelihood for hyper-parameters 
    marginalised over the GW parameters
    
    '''
    def __init__(self, population, data, params_inference ):
        '''
        

        Parameters
        ----------
        population : TYPE object of type AllPopulations

        data : TYPE object Data

        '''
        self.population=population
        self.data = data
        self.params_inference=params_inference
    
    
    def _get_mass_redshift(self, Lambda):
        
        LambdaCosmo, LambdaAllPop = self.population._split_params(Lambda)
        H0, Om0, w0,  Xi0, n = self.population.cosmo._get_values(LambdaCosmo, ['H0', 'Om', 'w0','Xi0', 'n'])
        
        z = self.population.cosmo.z_from_dLGW_fast(self.data.dL, H0, Om0, w0, Xi0, n)
        m1 = self.data.m1z / (1 + z)    
        m2 = self.data.m2z / (1 + z)
        
        return m1, m2, z
    
    def _getSpins(self, ):
        return self.data.chiEff
    
    def _getTobs(self):
        return self.data.Tobs
     
    def logLik(self, Lambda_test, ):
        """
        Returns log likelihood for all data
        """
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        m1, m2, z = self._get_mass_redshift(Lambda)
        chiEff = self._getSpins()
        Tobs = self._getTobs()
        logLik_ = self.population.log_dN_dm1zdm2zddL(m1, m2, z, chiEff, Tobs, Lambda) #m1, m2, z, chiEff, Tobs, Lambda
        logLik_ -= self.data.logOrMassPrior()
        logLik_ -= self.data.logOrDistPrior()

        allLogLiks = np.logaddexp.reduce(logLik_, axis=-1)-self.data.logNsamples # mean over posterior samples ~ marginalise over GW parameters for every observation
    
        ll = allLogLiks.sum() # add log likelihoods for all observations
   
        if np.isnan(ll):
            raise ValueError('NaN value for logLik. Values of Lambda: %s' %(str(Lambda) ) )

        return ll
    
 