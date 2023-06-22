#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:54:06 2021

@author: Michi
"""
import numpy as np
import numpy.ma as ma

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
    def __init__(self, population, data, params_inference, safety_factor=10, verbose=False, normalized=False, zmax=20):
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
        self.zmax = zmax
        print("zmax in lik is %s"%self.zmax)
        if normalized :
            print('This model will marginalize analytically over the overall normalization with a flat-in-log prior!')

    
    def _get_mass_redshift(self, Lambda, data):
        #print("\n_get_mass_redshift call")
        LambdaCosmo, LambdaAllPop = self.population._split_params(Lambda)
        H0, Om0, w0,  Xi0, n = self.population.cosmo._get_values(LambdaCosmo, ['H0', 'Om', 'w0','Xi0', 'n'])
        #print(H0, Om0, w0,  Xi0, n)
        #print(data.dL)
        z = self.population.cosmo.z_from_dLGW_fast(data.dL, H0, Om0, w0, Xi0, n)
        #print(z)
        m1 = data.m1z / (1 + z)    
        m2 = data.m2z / (1 + z)
        #print("\n\n")
        return m1, m2, z
    
    def _getSpins(self,data ):
        return data.spins
    
    def _getTobs(self, data):
        return data.Tobs
     
    
    def _logLik(self, Lambda_test, data, marginalize=True, return_all=False):
        """
        Returns log likelihood for each dataset
        """
        
        if not return_all and not marginalize:
            if data.dL.shape[1]>1:
                raise ValueError('It does not make sense to add likelihoods before marginalizing over single event parameters, unless you have a single sample for theta for each event. Here dl shape is %s' %str(data.dL.shape))
        
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        m1, m2, z = self._get_mass_redshift(Lambda, data)
        #print(Lambda)
        #print('m1 shape: %s' %str(m1.shape))
        #print(m1)
        spins = self._getSpins(data)
        
        if not self.normalized :
            Tobs = self._getTobs(data)
        else:
            Tobs=1.
        
        # If different events have different number of samples, 
        # This is taken into account by filling the likelihood with -infty
        # where the array of samples has been filled with nan
        logLik_=np.zeros(shape=m1.shape)
        #print('logLik_ init shape: %s' %str(logLik_.shape))
        where_compute=~np.isnan(m1)
        #print('where_compute shape: %s' %str(where_compute.shape))
        # Apply masks
        #logLik_ = ma.array(logLik_, mask=where_compute)
        #logLik_ = logLik_[~logLik_.mask]
        #logLik_ = np.where(where_compute, logLik_ , np.NINF)
        #m1 = np.where(where_compute, m1 , np.NINF)
        #logLik_[~where_compute]=np.NINF
        
        #print('where_compute init shape: %s' %str(where_compute.shape))
        #print(where_compute)
        
        #logLik_ = np.where( ~np.isnan(m1), self.population.log_dN_dm1zdm2zddL(m1, m2, z, spins, Tobs, Lambda), np.NINF) #m1, m2, z, spins, Tobs, Lambda
        #spins=[ np.where(where_compute, s , np.NINF) for s in spins]
        #print('m1 input:')
        #print(m1)
        logLik_ = np.where(where_compute,  self.population.log_dN_dm1zdm2zddL( m1, m2, z, spins, Lambda, Tobs, dL=data.dL), np.NINF)
        #print('logLik_  shape: %s' %str(logLik_.shape))
        
        
        #if not self.normalized :
        #    logLik_[where_compute] += np.log(Tobs)
        
        # Remove original prior from posterior samples to get the likelihood        
        logLik_ -= data.logOrMassPrior()
        logLik_ -=  data.logOrDistPrior()
        
        #print('logLik_  shape after mask and dNdz: %s' %str(ll.shape))
        
        #print(np.log(where_compute.sum(axis=-1)).shape)
        #print(data.logNsamples.shape)
        #print(np.log(where_compute.sum(axis=-1))==data.logNsamples)
        assert (np.log(where_compute.sum(axis=-1))==data.logNsamples).all()
        #logLik_ = np.where(where_compute, ll, np.NINF)
        #print('logLik_ where_compute shape: %s' %str(logLik_.shape))
        if marginalize:
            # mean over posterior samples ~ marginalise over GW parameters for every observation
            allLogLiks = np.logaddexp.reduce(logLik_, axis=-1)-data.logNsamples 
            # Now allLogLiks has shape=n. of observations
            
            # Weight in case data compression was applied
            allLogLiks*=data.bin_weights
            
            # Check number of effective samples for computing MC integral when marginalizing
            logs2 = ( np.logaddexp.reduce(2*logLik_, axis=-1) -2*data.logNsamples)
            logSigmaSq = logdiffexp( logs2, 2.0*allLogLiks - data.logNsamples)
            Neff = np.exp( 2.0*allLogLiks - logSigmaSq)
        else:
            allLogLiks = logLik_
            Neff = np.full( allLogLiks.shape, np.inf)
    
        
        #print('allLogLiks  shape after marginalization of logLik_ (i.e. sum over samples for each event): %s' %str(allLogLiks.shape))
        
        
        
        #print('allLogLiks  shape after weights: %s' %str(allLogLiks.shape))
        #print('allLogLiks after weights: %s' %allLogLiks)
        
        if np.any(Neff<self.safety_factor):
            if self.verbose:
                print('Not enough samples to safely evaluate the likelihood. Neff: %s at position(s) %s for safety factor: %s. Rejecting sample. Values of Lambda: %s' %(str(Neff[Neff<self.safety_factor]), str(np.argwhere(Neff<self.safety_factor).T),self.safety_factor,str(Lambda)))
            if not return_all:
                return np.NINF
            else:
                return np.where(Neff<self.safety_factor, np.NINF, allLogLiks )

        else:
            if not return_all:
                # add log likelihoods for all observations
                ll = allLogLiks.sum(axis=0)
                #print('ll shape after summing over observations: %s' %str(ll.shape))
            else:
                ll=allLogLiks
            #if self.normalized :
            #    ll -= np.log(self.population.Nperyear_expected(Lambda, zmax=20, verbose=False))*data.Nobs
            
            #if self.normalized :
            #    ll -= data.Nobs*np.log(self.population.Nperyear_expected(Lambda, zmax=self.zmax, verbose=False))
            
            if np.any(np.isnan(ll)):
                raise ValueError('NaN value for logLik. Values of Lambda: %s' %(str(Lambda) ) )
            return ll
    
    
    def logLik(self, Lambda_test, **kwargs):
        
        allL = []
        for data_ in self.data:
            allL.append(self._logLik( Lambda_test, data_, **kwargs))
        return  allL  
        
            
            
