#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:54:06 2021

@author: Michi
"""
import numpy as np

def logdiffexp(x, y):
    '''                                                                                                                                                                      
    computes log( e^x - e^y)                                                                                                                                                 
    '''
    return x + np.log1p(-np.exp(y-x))



class HyperLikelihood(object):
    
    '''
    Implements the logic for computing the hyper-likelihood, i.e. the likelihood for hyper-parameters 
    marginalised over the GW parameters
    
    '''
    def __init__(self, population, data, params_inference, safety_factor=100, verbose=False, normalized=False):
        '''
        

        Parameters
        ----------
        population : TYPE object of type AllPopulations

        data : TYPE list of objects Data

        '''
        self.population=population
        self.data = data # list of data objects
        self.params_inference=params_inference
        self.safety_factor = safety_factor
        self.verbose=verbose
        self.normalized=normalized
        if normalized :
            print('This model will marginalize analytically over the overall normalization with a flat-in-log prior!')

    
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
     
    
    def _logLik(self, Lambda_test, data,):
        """
        Returns log likelihood for each dataset
        """
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        m1, m2, z = self._get_mass_redshift(Lambda, data)
        spins = self._getSpins(data)
        
        if not self.normalized :
            Tobs = self._getTobs(data)
        else:
            Tobs=1.
        
        # If different events have different number of samples, 
        # This is taken into account by filling the likelihood with -infty
        # where the array of samples has been filled with nan
        logLik_=np.zeros(shape=m1.shape)
        #print(logLik_.shape)
        where_compute=~np.isnan(m1)
        logLik_[~where_compute]=np.NINF
        
        #logLik_ = np.where( ~np.isnan(m1), self.population.log_dN_dm1zdm2zddL(m1, m2, z, spins, Tobs, Lambda), np.NINF) #m1, m2, z, spins, Tobs, Lambda
        spins=[s[where_compute] for s in spins]
        logLik_[where_compute] = self.population.log_dN_dm1zdm2zddL( m1[where_compute], m2[where_compute], z[where_compute], spins, Lambda, Tobs, dL=data.dL[where_compute])
        #print(logLik_.shape)
        
        if self.normalized :
            logLik_[where_compute] -= np.log(self.population.Nperyear_expected(Lambda, zmax=20, verbose=False))
        
        #if not self.normalized :
        #    logLik_[where_compute] += np.log(Tobs)
        
        # Remove original prior from posterior samples to get the likelihood        
        logLik_ -= data.logOrMassPrior()
        logLik_ -= data.logOrDistPrior()
        
        assert (np.log(where_compute.sum(axis=-1))==data.logNsamples).all()
        # mean over posterior samples ~ marginalise over GW parameters for every observation
        allLogLiks = np.logaddexp.reduce(logLik_, axis=-1)-data.logNsamples 
        # Now allLogLiks has shape=n. of observations
        
        #print(allLogLiks)
        #print(data.bin_weights)
        #print(data.bin_weights)
        
        # Weight in case data compression was applied
        allLogLiks*=data.bin_weights
        
        # Check number of effective samples
        logs2 = ( np.logaddexp.reduce(2*logLik_, axis=-1) -2*data.logNsamples)
        logSigmaSq = logdiffexp( logs2, 2.0*allLogLiks - data.logNsamples)
        Neff = np.exp( 2.0*allLogLiks - logSigmaSq)
        if np.any(Neff<self.safety_factor):
            if self.verbose:
                print('Not enough samples to safely evaluate the likelihood. Neff: %s at position(s) %s for safety factor: %s. Rejecting sample. Values of Lambda: %s' %(str(Neff[Neff<self.safety_factor]), str(np.argwhere(Neff<self.safety_factor).T),self.safety_factor,str(Lambda)))
            return np.NINF

        else:
            # add log likelihoods for all observations
            ll = allLogLiks.sum()
            #if self.normalized :
            #    ll -= np.log(self.population.Nperyear_expected(Lambda, zmax=20, verbose=False))*data.Nobs
            
            if np.isnan(ll):
                raise ValueError('NaN value for logLik. Values of Lambda: %s' %(str(Lambda) ) )
            return ll
    
    
    def logLik(self, Lambda_test, **kwargs):
        
        allL = []
        for data_ in self.data:
            allL.append(self._logLik( Lambda_test, data_, **kwargs))
        return  allL  
        
            
            
