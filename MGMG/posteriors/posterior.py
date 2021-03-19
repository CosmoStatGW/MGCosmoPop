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
        lls = self.hyperLikelihood.logLik(Lambda_test)
        
        #logll = np.log(ll)
        
        # Compute selection bias
        # Includes uncertainty on MC estimation of the selection effects if required. err is =zero if we required to ignore it.
        mus, errs = self.selectionBias.Ndet(Lambda_test, **kwargs)
        
        #logNdet = logdiffexp(logMu, logErr )
        logPosts = np.zeros(len(lls))
        for i in range(len(lls)):
            logPosts[i] = lls[i]-mus[i] #-np.exp(logNdet.astype('float128')) 
            # Add uncertainty on MC estimation of the selection effects. err is =zero if we required to ignore it.
            logPosts[i] += errs[i]
        
        # sum log likelihood of different datasets
        logPost = logPosts.sum()
        
        
        # Add prior
        logPost += lp
        
        #if err!=np.NAN:   
        #    logPost += err
        if not return_all:
            return logPost
        else:
            return logPost, lp, lls, mus, errs #np.exp( logMu.astype('float128')), np.exp(logErr.astype('float128'))
        
        
        