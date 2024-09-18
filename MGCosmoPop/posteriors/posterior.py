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
        print('Bias sefety factor is %s'%self.bias_safety_factor)

    
        
    def logPosterior(self, Lambda_test, return_all=False,):
        
        # Compute prior
        lp = self.prior.logPrior(Lambda_test)
        if (not np.isfinite(lp)) and (not return_all):
            return -np.inf
        
        # Get all parameters in case we are fixing some of them
        # Lambda = self.hyperLikelihood.population.get_Lambda(Lambda_test, self.prior.params_inference )
        
        # Compute likelihood
        lls = self.hyperLikelihood.logLik(Lambda_test)

        llsum = np.asarray(lls).sum()
        #print("lls : %s"%str(llsum))
        if (not np.isfinite(llsum)) and (not return_all):
            return -np.inf
        
        
        
        # Compute selection bias
        # Includes uncertainty on MC estimation of the selection effects if required. err is =zero if we required to ignore it.
        if self.selectionBias is not None:
            mus, errs, Neffs = self.selectionBias.Ndet(Lambda_test, )
            #print("Len mus is %s"%str(len(mus)))
            if len(mus)==1:
                # Using one injection set
                self.single_injection_set = True
            else:
                self.single_injection_set = False
            #print("Single injection set is %s"%self.single_injection_set)
            
        else:
            # Test case: we see the full population.
            if not self.normalized :
                Lambda = self.hyperLikelihood.population.get_Lambda(Lambda_test, self.hyperLikelihood.params_inference )
                mus = [ self.hyperLikelihood.population.Nperyear_expected(Lambda)*self.hyperLikelihood._getTobs(self.hyperLikelihood.data[i]) for i in range(len(lls))]
                errs = [0 for _ in range(len(lls))]
            else:
                mus = np.ones(len(lls))
                errs = np.zeros(len(lls))
        
        # put all together and get posterior
        logPosts = np.zeros(len(lls))
        for i in range(len(lls)):
            #print("like for dataset %s"%i)
            #print("Single injection set is %s"%self.single_injection_set)
            if not self.single_injection_set:
                #print("More than 1 inj set")
                reject = Neffs[i] < self.bias_safety_factor * self.hyperLikelihood.data[i].Nobs
            else:
                if i==0:
                    reject = Neffs[i] < self.bias_safety_factor * self.hyperLikelihood.data[i].Nobs
                else:
                    reject=False
                    
            if reject:
                if self.verbose:
                    print('NEED MORE SAMPLES FOR SELECTION EFFECTS! Nobs = %s, Neff = %s, Values of Lambda: %s' %(self.hyperLikelihood.data[i].Nobs, Neffs[i], str(Lambda_test)))
                    print( "Safety factor is %s"%self.bias_safety_factor )
                #print( "Nobs is %s"%self.hyperLikelihood.data[i].Nobs )
                # reject the sample
                logPosts[i] = -np.inf
            else:
                if ( not self.single_injection_set or i==0):
                    if not self.normalized:
                        logPosts[i] = lls[i]-mus[i] 
                        # Add uncertainty on MC estimation of the selection effects. err is =zero if we required to ignore it.
                        logPosts[i] += errs[i]
                    else:
                        if self.single_injection_set:
                            N = 0
                            for j in range(len(lls)):
                                N+=self.hyperLikelihood.data[j].Nobs
                            #print("In sel effect, using N=%s"%N)
                        else:
                            N = self.hyperLikelihood.data[i].Nobs
                        
                        logPosts[i] = lls[i]-N*np.log(mus[i])
                        if self.selectionBias.get_uncertainty:
                            err = (3*N+(N)**2 )/(2*Neffs[i])
                        else:
                            err = 0.
                        logPosts[i] += err
                else:
                    pass
                    
        
        # sum log likelihood of different datasets
        logPost = logPosts.sum()
        
        
        #print("mus : %s"%str(mus))
        mus = np.asarray(mus).sum()
        #print("mus sum : %s"%str(mus))
        #if mus==0:
        #    print(Lambda_test)
        errs= np.asarray(errs).sum()
        #print("errs : %s"%str(errs))
        
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
            return logPost, lp, llsum, mus, errs #np.exp( logMu.astype('float128')), np.exp(logErr.astype('float128'))
        
        
        
