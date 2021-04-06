#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:41:18 2021

@author: Michi
"""
 
from .ABSdata import Data

import numpy as np
import astropy.units as u
import h5py
#import os
from astropy.cosmology import Planck15, z_at_value

        
class GWMockData(Data):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, Tobs=2.5 ):
        
        self.dist_unit = dist_unit
        self.m1z, self.m2z, self.dL, self.Nsamples = self._load_data(fname, nObsUse, nSamplesUse, )  
        self.Nobs=self.m1z.shape[0]
        print('We have %s observations' %self.Nobs)
        print('Number of samples: %s' %self.Nsamples )
        
        self.logNsamples = np.log(self.Nsamples)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<=self.m1z).all()
        
        self.Tobs=Tobs
        #self.chiEff = np.zeros(self.m1z.shape)
        self.spins = []
        print('Obs time: %s' %self.Tobs )
        
        self.Nobs=self.m1z.shape[0]
        
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    
    def _load_data(self, fname, nObsUse, nSamplesUse,):
        print('Loading data...')
        if nObsUse is None:
            nObsUse=-1
        with h5py.File(fname, 'r') as phi: #observations.h5 has to be in the same folder as this code
               
                m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :]# m1
                m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :] # m2
                dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :]
                #print(dl_samples.shape)
        
        if self.dist_unit==u.Mpc:
            print('Using distances in Mpc')
            dl_samples*=1e03
        #theta =   np.array([m1det_samples, m2det_samples, dl_samples])
        
        if nSamplesUse is None:
            nSamplesUse=dl_samples.shape[1]
        m1z=np.empty((m1det_samples.shape[0],nSamplesUse) )
        m2z=np.empty((m1det_samples.shape[0],nSamplesUse))
        dL=np.empty((m1det_samples.shape[0],nSamplesUse))
        #print(m1z.shape)
        for i in range(m1det_samples.shape[0]):
            vb= (i==0)
            m1z[i], m2z[i], dL[i] = self.downsample([m1det_samples[i], m2det_samples[i], dl_samples[i]], nSamplesUse, verbose=vb)
            
        #m1det_samples, m2det_samples, dl_samples = self.downsample([m1det_samples, m2det_samples, dl_samples], nSamplesUse)
        return m1z, m2z, dL, np.count_nonzero(m1z, axis=-1)
      
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        return np.zeros(self.dL.shape)
    



class GWMockInjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None,  dist_unit=u.Gpc, Tobs=2.5 ):
        
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.weights_sel, self.N_gen = self._load_data(fname, nInjUse )        
        self.logN_gen = np.log(self.N_gen)
        self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<=self.m1z).all()
        self.condition=True
        
        self.Tobs=Tobs
        self.spins = []# np.zeros(self.m1z.shape)
        print('Obs time: %s' %self.Tobs )
        
        
        
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    def _load_data(self, fname, nInjUse,):
        print('Loading injections...')
        with h5py.File(fname, 'r') as f:
        
            if nInjUse is not None:
                m1_sel = np.array(f['m1det'])[:nInjUse]
                m2_sel = np.array(f['m2det'])[:nInjUse]
                dl_sel = np.array(f['dl'])[:nInjUse]
                weights_sel = np.array(f['wt'])[:nInjUse]
            else:
                m1_sel = np.array(f['m1det'])
                m2_sel = np.array(f['m2det'])
                dl_sel = np.array(f['dl'])
                weights_sel = np.array(f['wt'])
        
            N_gen = f.attrs['N_gen']
        if self.dist_unit==u.Mpc:
            dl_sel*=1e03
            
        #self.max_z = np.max(z)
        self.max_z=z_at_value(Planck15.luminosity_distance, dl_sel.max()*self.dist_unit)
        
        # Drop points in the unlikely case of m1==m2, to avoid crashes
        keep = m1_sel!=m2_sel
        throw = ~keep
        print('Dropping %s points with exactly equal masses' %str(throw.sum()) )
        
        
        print('Max redshift of injections: %s' %self.max_z)
        print('Number of total injections: %s' %N_gen)
        print('Number of detected injections: %s' %weights_sel[keep].shape[0])
        return m1_sel[keep], m2_sel[keep], dl_sel[keep], weights_sel[keep] , N_gen
      
    
    def originalMassPrior(self):
        return np.ones(self.m1z.shape)

    def originalDistPrior(self):
        return np.ones(self.dL.shape)    
    
