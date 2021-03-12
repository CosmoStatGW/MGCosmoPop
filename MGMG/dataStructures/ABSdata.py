#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:27:42 2021

@author: Michi
"""
from abc import ABC, abstractmethod
from scipy.stats import ks_2samp
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


class Data(ABC):
    
    def __init__(self, ):
        pass
        
    @abstractmethod
    def _load_data(self):
        pass
    
    @abstractmethod
    def get_theta(self):
        pass
    
    
    def downsample(self, posterior, nSamples, pth=0.05, verbose=True):
        
        if nSamples is None:
            return posterior
        if verbose:
            print('Downsampling posterior to %s samples...' %nSamples)
        
        posterior = np.array(posterior)
        #nparams=posterior.shape[0]
        nOrSamples=posterior.shape[1]
        if verbose:
            print('Number of original samples: %s '%nOrSamples)
        
        if len(posterior) == 1:
            n, bins = np.histogram(posterior, bins=50)
            n = np.array([0] + [i for i in n])
            cdf = cumtrapz(n, bins, initial=0)
            cdf /= cdf[-1]
            icdf = interp1d(cdf, bins)
            samples = icdf(np.random.rand(nSamples))
        else:
            #posterior = np.array([i for i in arr])
            keep_idxs = np.random.choice(nOrSamples, nSamples, replace=False)
            samples = [i[keep_idxs] for i in posterior]

        
        #step = arr.shape[0]//nMax
        #shortarr = arr[::step][:nMax]
        
        #keep = np.random.choice(len(arr), nMax, replace=False)
        #samples = arr[keep] #[i[keep_idxs] for i in posterior]

        
        #ksres = ks_2samp(samples, arr)[1]
        #if ksres<pth:
        #    raise ValueError('P-value of KS test between origina samples and full samples is <0.05. Use a larger number of samples')
        
        return samples