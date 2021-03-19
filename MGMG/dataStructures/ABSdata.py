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
import astropy.units as u
import os


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
    
    
    
    
class LVCData(Data):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='chiEff' ):
        
        Data.__init__(self)
        self.events_use=events_use
        self.which_spins=which_spins
        
        if events_use['use'] is not None:
            nObsUse=len(events_use['use'])
        
        self.dist_unit = dist_unit
        self.events = self._get_events(fname, events_use)
        
        self.m1z, self.m2z, self.dL, self.spins, self.Nsamples = self._load_data(fname, nObsUse, nSamplesUse, which_spins=which_spins)  
        self.Nobs=self.m1z.shape[0]
        #print('We have %s observations' %self.Nobs)
        print('Number of samples used: %s' %self.Nsamples )
        
        self.logNsamples = np.log(self.Nsamples)
        #assert (self.m1z >= 0).all()
        #assert (self.m2z >= 0).all()
        #assert (self.dL >= 0).all()
        #assert(self.m2z<=self.m1z).all()
        
        # The first observing run (O1) ran from September 12th, 2015 to January 19th, 2016 --> 129 days
        # The second observing run (O2) ran from November 30th, 2016 to August 25th, 2017 --> 267 days
        self._set_Tobs() #Tobs=  (129+267)/365. # yrs
        
        print('Obs time (yrs): %s' %self.Tobs )
        
        self.Nobs=self.m1z.shape[0]
    
    @abstractmethod
    def _name_conditions(self, f ):
        pass
    
    @abstractmethod
    def _set_Tobs(self):
        pass
    
    @abstractmethod
    def _get_not_BBHs(self):
        pass

    @abstractmethod
    def _load_data_event(self, **kwargs):
        pass
    
    
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL , self.spins ] )  
    
    @abstractmethod
    def _get_name_from_fname(self, fname):
        pass
    
    
    def _get_events(self, fname, events_use, ):
        
        #dirlist = [ item for item in os.listdir(fname) if os.path.isdir(os.path.join(fname, item)) ]        
        #print(fname)
        #print(os.listdir(fname))
        #print(os.path.join(fname, '*.h5' ))
        allFiles = [f for f in os.listdir(fname) if f.endswith(self.post_file_extension)] #glob.glob(os.path.join(fname, '*.h5' ))
        #print(allFiles)
        #print([len(f.split('.')[0].split('_')) for f in allFiles])
        #print([f.split('_')[0][:2] for f in allFiles])
        elist = [self._get_name_from_fname(f) for f in allFiles if self._name_conditions(f) ]
        
        
        list_BBH = [x for x in elist if x not in self._get_not_BBHs() ]
        print('In the O3a data we have the following BBH events, total %s (excluding %s):' %(len(list_BBH) ,str(self._get_not_BBHs())) )
        print(list_BBH)
        if events_use['use'] is not None and events_use['not_use'] is not None:
            raise ValueError('You passed options to both use and not_use. Please only provide the list of events that you want to use, or the list of events that you want to exclude. ')
        elif events_use['use'] is not None:
            # Use only events given in use
            print('Using only BBH events : ')
            print(events_use['use'])
            list_BBH_final = [x for x in list_BBH if x in events_use['use']]
        elif events_use['not_use'] is not None:
            print('Excluding BBH events : ')
            print(events_use['not_use'])
            list_BBH_final = [x for x in list_BBH if x not in events_use['not_use']]
        else:
            print('Using all BBH events')
            list_BBH_final=list_BBH
        return list_BBH_final
 
    
    def _load_data(self, fname, nObsUse, nSamplesUse, which_spins='chiEff'):
        print('Loading data...')
    
        
        #events = self._get_events_names(fname)
        if nObsUse is None:
            nObsUse=len(self.events)
            
        
        #print('We have the following events: %s' %str(events))
        m1s, m2s, dLs, spins = [], [], [], []
        allNsamples=[]
        for event in self.events[:nObsUse]:
                print('Reading data from %s' %event)
            #with h5py.File(fname, 'r') as phi:
                m1z_, m2z_, dL_, spins_  = self._load_data_event(fname, event, nSamplesUse, which_spins=which_spins)
                print('Number of samples in LVC data: %s' %m1z_.shape[0])
                m1s.append(m1z_)
                m2s.append(m2z_)
                dLs.append(dL_)
                spins.append(spins_)
                assert len(m1z_)==len(m2z_)
                assert len(m2z_)==len(dL_)
                if which_spins!="skip":
                    assert len(spins_)==2
                    assert len(spins_[0])==len(dL_)
                else:  assert spins_==[]
                
                nSamples = len(m1z_)
                
                allNsamples.append(nSamples)
            #print('ciao')
        print('We have %s events.'%len(allNsamples))
        max_nsamples = max(allNsamples) 
        
        fin_shape=(nObsUse, max_nsamples)
        
        m1det_samples= np.full(fin_shape, np.NaN)  #np.zeros((len(self.events),max_nsamples))
        m2det_samples=np.full(fin_shape, np.NaN)
        dl_samples= np.full(fin_shape, np.NaN)
        if which_spins!="skip":
            spins_samples= [np.full(fin_shape, np.NaN), np.full(fin_shape, np.NaN) ]
        else: spins_samples=[]
        
        for i in range(nObsUse):
            
            m1det_samples[i, :allNsamples[i]] = m1s[i]
            m2det_samples[i, :allNsamples[i]] = m2s[i]
            dl_samples[i, :allNsamples[i]] = dLs[i]
            if which_spins!="skip":
                spins_samples[0][i, :allNsamples[i]] = spins[i][0]
                spins_samples[1][i, :allNsamples[i]] = spins[i][1]
        
        if self.dist_unit==u.Gpc:
            print('Using distances in Gpc')   
            dl_samples*=1e-03
        
        return m1det_samples, m2det_samples, dl_samples, spins_samples, allNsamples
    
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        # dl^2 prior on dL
        return np.where( ~np.isnan(self.dL), 2*np.log(self.dL), 0)
    
  
  

  
