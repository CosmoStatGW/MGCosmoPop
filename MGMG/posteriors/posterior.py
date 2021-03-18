#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:18:13 2021

@author: Michi
"""
import numpy as np


class Posterior(object):
    
    def __init__(self, hyperLikelihood, prior, selectionBias):
        
        self.hyperLikelihood = hyperLikelihood
        self.prior = prior
        self.selectionBias = selectionBias
        #self.params_inference = params_inference
        
        
    def logPosterior(self, Lambda_test, return_all=False):
        
        # Compute prior
        lp = self.prior.logPrior(Lambda_test)
        if not np.isfinite(lp):
            return -np.inf
        
        # Get all parameters in case we are fixing some of them
        # Lambda = self.hyperLikelihood.population.get_Lambda(Lambda_test, self.prior.params_inference )
        
        # Compute likelihood
        ll = self.hyperLikelihood.logLik(Lambda_test)
        logPost= ll+lp
        
        # Compute selection bias
        mu, err = self.selectionBias.Ndet(Lambda_test)
        logPost -= mu
        
        if err!=np.NAN:
            # Add uncertainty on MC estimation of the selection effects
            logPost += err
        if not return_all:
            return logPost
        else:
            return logPost, lp, ll, mu, err
        
        
        