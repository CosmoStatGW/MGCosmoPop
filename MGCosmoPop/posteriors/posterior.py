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
    
    def __init__(self, hyperLikelihood, prior, selectionBias, verbose=False, 
                 bias_safety_factor=10., 
                 normalized=False):
        
        self.hyperLikelihood = hyperLikelihood
        self.prior = prior
        self.selectionBias = selectionBias
        self.verbose=verbose
        self.bias_safety_factor=bias_safety_factor
        self.normalized=normalized
        if normalized :
            print('This model will marginalize analytically over the overall normalization with a flat-in-log prior!')
        #self.params_inference = params_inference

        
    def logPosterior(self, Lambda_test, return_all=False,):
        
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
        if self.selectionBias is not None:
            mus, errs, Neffs = self.selectionBias.Ndet(Lambda_test, )
        else:
            # Test case: we see the full population.
            if not self.normalized :
                Lambda = self.hyperLikelihood.population.get_Lambda(Lambda_test, self.hyperLikelihood.params_inference )
                mus = [ self.hyperLikelihood.population.Nperyear_expected(Lambda)*self.hyperLikelihood._getTobs(self.hyperLikelihood.data[i]) for i in range(len(lls))]
                errs = [0 for _ in range(len(lls))]
            else:
                mus = np.ones(len(lls))
                errs = np.zeros(len(lls))
        
        #logNdet = logdiffexp(logMu, logErr )
        logPosts = np.zeros(len(lls))
        for i in range(len(lls)):
            if Neffs[i] < self.bias_safety_factor * self.hyperLikelihood.data[i].Nobs:
                if self.verbose:
                    print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Nobs = %s, Neff = %s, Values of Lambda: %s' %(self.hyperLikelihood.data[i].Nobs, Neffs[i], str(Lambda_test)))
                # reject the sample
                logPosts[i] = -np.inf
            else:
                
                if not self.normalized:
                    logPosts[i] = lls[i]-mus[i] #-np.exp(logNdet.astype('float128')) 
                    # Add uncertainty on MC estimation of the selection effects. err is =zero if we required to ignore it.
                    logPosts[i] += errs[i]
                else:
                    logPosts[i] = lls[i]-self.hyperLikelihood.data[i].Nobs*np.log(mus[i])
                    err = (3*self.hyperLikelihood.data[i].Nobs+(self.hyperLikelihood.data[i].Nobs)**2 )/(2*Neffs[i])
                    logPosts[i] += err
        
        # sum log likelihood of different datasets
        logPost = logPosts.sum()
        
        #if not self.normalized :
        #    Tobs = np.array([self.hyperLikelihood._getTobs(self.hyperLikelihood.data[i]) for i in range(len(lls)) ])#.sum()
        #    Nobs = np.array([self.hyperLikelihood.data[i].Nobs for i in range(len(lls)) ])#.sum()
            # Add observation time
        #    logPost += (np.log(Tobs)*Nobs).sum()
        
        
        # Add prior
        logPost += lp
        
        
        
        if not return_all:
            return logPost
        else:
            return logPost, lp, lls, mus, errs #np.exp( logMu.astype('float128')), np.exp(logErr.astype('float128'))
        
        
        
