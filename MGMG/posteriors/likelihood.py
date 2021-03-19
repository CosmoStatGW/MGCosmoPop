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

        data : TYPE list of objects Data

        '''
        self.population=population
        self.data = data # list of data objects
        self.params_inference=params_inference
    
    
    def _get_mass_redshift(self, Lambda, data):
        
        LambdaCosmo, LambdaAllPop = self.population._split_params(Lambda)
        H0, Om0, w0,  Xi0, n = self.population.cosmo._get_values(LambdaCosmo, ['H0', 'Om', 'w0','Xi0', 'n'])
        
        z = self.population.cosmo.z_from_dLGW_fast(data.dL, H0, Om0, w0, Xi0, n)
        m1 = data.m1z / (1 + z)    
        m2 = data.m2z / (1 + z)
        
        return m1, m2, z
    
    def _getSpins(self,data ):
        return data.spins
    
    def _getTobs(self, data):
        return data.Tobs
     
    
    def _logLik(self, Lambda_test, data):
        """
        Returns log likelihood for each dataset
        """
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        m1, m2, z = self._get_mass_redshift(Lambda, data)
        spins = self._getSpins(data)
        Tobs = self._getTobs(data)
        
        # If different events have different number of samples, 
        # This is taken into account by filling the likelihood with -infty
        # where the array of samples has been filled with nan
        
        logLik_ = np.where( ~np.isnan(m1), self.population.log_dN_dm1zdm2zddL(m1, m2, z, spins, Tobs, Lambda), np.NINF) #m1, m2, z, spins, Tobs, Lambda
        
        # Remove original prior from posterior samples to get the likelihood
        
        logLik_ -= data.logOrMassPrior()
        logLik_ -= data.logOrDistPrior()
        
        
        # mean over posterior samples ~ marginalise over GW parameters for every observation
        allLogLiks = np.logaddexp.reduce(logLik_, axis=-1)-data.logNsamples 
        
        # add log likelihoods for all observations
        ll = allLogLiks.sum() 
   
        if np.isnan(ll):
            raise ValueError('NaN value for logLik. Values of Lambda: %s' %(str(Lambda) ) )

        return ll
    
    
    def logLik(self, Lambda_test, ):
        
        allL = []
        for data_ in self.data:
            allL.append(self._logLik( Lambda_test, data_))
        return  allL  
        
            
            