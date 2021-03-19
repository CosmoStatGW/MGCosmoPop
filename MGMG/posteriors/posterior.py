#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:18:13 2021

@author: Michi
"""
import numpy as np



def logdiffexp(x, y):
    '''
    computes log( e^x - e^y)
    '''
    return x + np.log1p(-np.exp(y-x))




class Posterior(object):
    
    def __init__(self, hyperLikelihood, prior, selectionBias):
        
        self.hyperLikelihood = hyperLikelihood
        self.prior = prior
        self.selectionBias = selectionBias
        #self.params_inference = params_inference
        
        
    def logPosterior(self, Lambda_test, return_all=False, **kwargs):
        
        # Compute prior
        lp = self.prior.logPrior(Lambda_test)
        if not np.isfinite(lp):
            return -np.inf
        
        # Get all parameters in case we are fixing some of them
        # Lambda = self.hyperLikelihood.population.get_Lambda(Lambda_test, self.prior.params_inference )
        
        # Compute likelihood
        ll = self.hyperLikelihood.logLik(Lambda_test)
        
        #logll = np.log(ll)
        
        # Compute selection bias
        # Includes uncertainty on MC estimation of the selection effects if required. err is =zero if we required to ignore it.
        logMu, logErr = self.selectionBias.logNdet(Lambda_test, **kwargs)
        
        logNdet = logdiffexp(logMu, logErr )
        
        
        logPost = ll-np.exp(logNdet.astype('float128')) 
        #logPost -= mu
        
        # Add uncertainty on MC estimation of the selection effects. err is =zero if we required to ignore it.
        #logPost += err
        
        # Add prior
        logPost+=lp
        
        #if err!=np.NAN:   
        #    logPost += err
        if not return_all:
            return logPost
        else:
            return logPost, lp, ll, np.exp( logMu.astype('float128')), np.exp(logErr.astype('float128'))
        
        
        