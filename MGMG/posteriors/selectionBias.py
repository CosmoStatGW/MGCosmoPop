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


def logdiffexpvec(xs, ys):
    '''
    computes log( sum_i(e^x_i) - sum_i(e^y_i) )
    '''
    xs=np.array(xs)
    ys=np.array(ys)
    return xs[0] + np.log1p(  np.sum(np.exp(xs[1:]-xs[0]))  - np.sum(np.exp(ys-xs[0])) )





class SelectionBias(ABC):
    
    def __init__(self, population, injData, params_inference ):
        
        self.injData=injData
        self.population=population
        self.params_inference = params_inference 


    @abstractmethod
    def Ndet(Lambda, **kwargs):
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
    
    
    def _get_mass_redshift(self, Lambda, injData):
        
        LambdaCosmo, LambdaAllPop = self.population._split_params(Lambda)
        H0, Om0, w0,  Xi0, n = self.population.cosmo._get_values(LambdaCosmo, ['H0', 'Om', 'w0','Xi0', 'n'])
        
        z = self.population.cosmo.z_from_dLGW_fast(injData.dL, H0, Om0, w0, Xi0, n)
        m1 = injData.m1z / (1 + z)    
        m2 = injData.m2z / (1 + z)
        
        return m1, m2, z
    
    
    def _getSpins(self, injData):
        return injData.spins
    
    def _getTobs(self, injData):
        return injData.Tobs
    
    
    def _Ndet(self, Lambda_test, injData, verbose=False, Nobs = None):
        
        Lambda = self.population.get_Lambda(Lambda_test, self.params_inference )
        
        m1, m2, z = self._get_mass_redshift(Lambda, injData)
        spins = self._getSpins(injData)
        Tobs = self._getTobs(injData)
        
        #logdN=np.empty_like(m1)
        #logdN[~injData.condition]=np.NINF
    
        m1, m2, z, spins = m1[injData.condition], m2[injData.condition], z[injData.condition], [s[injData.condition] for s in spins]
        
        
        #logdN =  np.where( injData.condition, self.population.log_dN_dm1zdm2zddL(m1, m2, z, spins, Tobs, Lambda),  np.NINF) 
        #logdN -= injData.log_weights_sel
        logdN=self.population.log_dN_dm1zdm2zddL(m1, m2, z, spins, Tobs, Lambda)-injData.log_weights_sel[injData.condition]
        
        
        logMu = np.logaddexp.reduce(logdN) - injData.logN_gen
        
        if np.isnan(logMu):
            raise ValueError('NaN value for logMu. Values of Lambda: %s' %( str(Lambda) ) )
        
        mu = np.exp(logMu)#.astype('float128')
        
        if not self.get_uncertainty:
            return mu, 0
        
        logs2 = ( np.logaddexp.reduce(2*logdN) -2*injData.logN_gen)#.astype('float128')
        logSigmaSq = logdiffexp( logs2, 2.0*logMu - injData.logN_gen )
        
        muSq = np.exp(2*logMu)
        SigmaSq = np.exp(logSigmaSq)#.astype('float128')
        
        if Nobs is not None and verbose:
            #muSq = np.exp(2*logMu)
            #SigmaSq = np.exp(logSigmaSq)
            Neff = muSq/SigmaSq #np.exp( 2.0*logMu - logSigmaSq)
            if Neff < 4 * Nobs:
                print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Values of Lambda: %s' %str(Lambda))
        
        #mu = np.exp(logMu.astype('float128'))
        Sigma = np.sqrt(SigmaSq)
        
        ## Effects of uncertainty on selection effect and/or marginalisation over total rate
        ## Adapted from 1904.10879
        
        num = ss.norm(loc=mu-SigmaSq, scale=Sigma).logsf(0)
        den = ss.norm(loc=mu, scale=Sigma ).logsf(0)
        error = SigmaSq/2-den+num
        
        #logError = logSigmaSq-np.log(2) +np.log(num-den)
        
        return mu, error
    
    
    def Ndet(self, Lambda_test, verbose=False, allNobs = None):
        
        mus=[]
        errs=[]
        for i, injData_ in enumerate(self.injData):
            if allNobs is None:
                Nobs=None
            else: Nobs=allNobs[i]
            mu_, err_ = self._Ndet(Lambda_test, injData_, verbose=verbose, Nobs = Nobs)
            mus.append(mu_)
            errs.append(err_)
        return mus, errs
            
        
        
        