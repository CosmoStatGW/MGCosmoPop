#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.
 
from .ABSdata import Data

import numpy as np
import astropy.units as u
import h5py
#import os
from astropy.cosmology import Planck15, z_at_value

        
class GWMockData(Data):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, percSamplesUse=None, dist_unit=u.Gpc, Tobs=2.5 ):
        
        self.dist_unit = dist_unit
        self.m1z, self.m2z, self.dL, self.snr, self.Nsamples, self.bin_weights = self._load_data(fname, nObsUse, ) #nSamplesUse, )  
        self.Nobs=self.m1z.shape[0]
        self.logNsamples = np.log(self.Nsamples)

        if nSamplesUse is not None or percSamplesUse is not None:
            self.downsample(nSamples=nSamplesUse, percSamples=percSamplesUse)
            print('Number of samples for each event after downsamplng: %s' %self.Nsamples )
        print('We have %s observations' %self.Nobs)
        print('Number of samples: %s' %self.Nsamples )
        

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
    
    
    def _load_data(self, fname, nObsUse, nSamplesUse=None):
        print('Loading data...')
        if nObsUse is None:
            nObsUse=-1
        with h5py.File(fname, 'r') as phi: #observations.h5 has to be in the same folder as this code
               
                m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :]# m1
                m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :] # m2
                dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :]
                try:
                    snrs = np.array(phi['posteriors']['rho'])[:nObsUse, :]
                #print(dl_samples.shape)
                except:
                    print('SNRs not present for this dataset. Use the same SNR threshold as the original injections.')
                    snrs = np.zeros(dl_samples.shape)

                try:
                    bin_weights = np.array(phi['posteriors']['bin_weights'])[:nObsUse]
                except Exception as e:
                    print(e)
                    print('No bin weights.')
                    bin_weights = np.ones(dl_samples.shape[0])
        
        if self.dist_unit==u.Mpc:
            print('Using distances in Mpc')
            dl_samples*=1e03
        #theta =   np.array([m1det_samples, m2det_samples, dl_samples])
        
        #if nSamplesUse is None:
        #    nSamplesUse=dl_samples.shape[1]
        #m1z=np.empty((m1det_samples.shape[0],nSamplesUse) )
        #m2z=np.empty((m1det_samples.shape[0],nSamplesUse))
        #dL=np.empty((m1det_samples.shape[0],nSamplesUse))
        #print(m1z.shape)
        #for i in range(m1det_samples.shape[0]):
        #    vb= (i==0)
        #    m1z[i], m2z[i], dL[i] = self.downsample([m1det_samples[i], m2det_samples[i], dl_samples[i]], nSamplesUse, verbose=vb)
            
        #m1det_samples, m2det_samples, dl_samples = self.downsample([m1det_samples, m2det_samples, dl_samples], nSamplesUse)
        #return m1z, m2z, dL, np.count_nonzero(m1z, axis=-1)
        return m1det_samples, m2det_samples, dl_samples, snrs, np.count_nonzero(m1det_samples, axis=-1), bin_weights 
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        return np.zeros(self.dL.shape)
    



class GWMockInjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None,  dist_unit=u.Gpc, Tobs=2.5, snr_th=None ):
        
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.weights_sel, self.log_weights_sel, self.snr_sel, self.N_gen, self.snr_th = self._load_data(fname, nInjUse )
        self.logN_gen = np.log(self.N_gen)
        #self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<=self.m1z).all()
        self.condition=True
        
        self.Tobs=Tobs
        self.spins = []# np.zeros(self.m1z.shape)
        print('Obs time: %s' %self.Tobs )
        
        if snr_th is not None:
            self.set_snr_threshold(snr_th)
        
    def set_snr_threshold(self, snr_th):
        if self.snr_sel.sum()==0.:
            print('Snrs not present in this dataset.')
            return
        if snr_th<self.snr_th:
            #raise ValueError('New snr threshold is lower than original one !')
            print('warning: New snr threshold is lower than original one ! Using original')
            return

        print('Updating snr threshold to %s' %snr_th)
        self.snr_th=snr_th
        keep = self.snr_sel >= snr_th
        self.m1z = self.m1z[keep]
        self.m2z = self.m2z[keep]
        self.dL = self.dL[keep]
        try:
            self.weights_sel =  self.weights_sel[keep]
        except TypeError:
            pass
        self.log_weights_sel = self.log_weights_sel[keep]
        self.snr_sel = self.snr_sel[keep]
        
        print('New number of detected injections with snr>%s :  %s' %(self.snr_th, self.dL.shape[0]))

        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    def _load_data(self, fname, nInjUse=None):
        print('Loading injections...')
        with h5py.File(fname, 'r') as f:
        
            if nInjUse is not None:
                m1_sel = np.array(f['m1det'])[:nInjUse]
                m2_sel = np.array(f['m2det'])[:nInjUse]
                dl_sel = np.array(f['dl'])[:nInjUse]
                try:
                    weights_sel = np.array(f['wt'])[:nInjUse]
                    log_weights_sel = np.log(weights_sel)
                except KeyError:
                    log_weights_sel = np.array(f['logwt'])[:nInjUse]
                    weights_sel=None
                try:
                    snr_sel = np.array(f['snr'])[:nInjUse]
                except:
                    snr_sel = np.zeros(dl_sel.shape)
            else:
                m1_sel = np.array(f['m1det'])
                m2_sel = np.array(f['m2det'])
                dl_sel = np.array(f['dl'])
                try:
                    weights_sel = np.array(f['wt'])
                    log_weights_sel = np.log(weights_sel)
                except KeyError:
                    log_weights_sel = np.array(f['logwt'])[:nInjUse]
                    weights_sel = None
                try:
                    snr_sel = np.array(f['snr'])
                except:
                    snr_sel = np.zeros(dl_sel.shape)
            
            N_gen = f.attrs['N_gen']
            try:
                snr_th = f.attrs['snr_th']
            except:
                print('Threshold snr not saved in this dataset. Assuming 8.')
                snr_th=8.
        if self.dist_unit==u.Mpc:
            dl_sel*=1e03
            
        #self.max_z = np.max(z)
        self.max_z=z_at_value(Planck15.luminosity_distance, dl_sel.max()*self.dist_unit)
        
        # Drop points in the unlikely case of m1==m2, to avoid crashes
        keep = m1_sel!=m2_sel
        throw = ~keep
        print('Dropping %s points with exactly equal masses' %str(throw.sum()) )
        N_gen -= throw.sum()
        if weights_sel is not None:
            weights_sel=weights_sel[keep]
        
        print('Max redshift of injections assuming Planck 15 cosmology: %s' %self.max_z)
        print('Number of total injections: %s' %N_gen)
        print('Number of detected injections: %s' %dl_sel[keep].shape[0])
        
        return m1_sel[keep], m2_sel[keep], dl_sel[keep], weights_sel, log_weights_sel[keep] , snr_sel[keep], N_gen, snr_th
      
    
    def originalMassPrior(self):
        return np.ones(self.m1z.shape)

    def originalDistPrior(self):
        return np.ones(self.dL.shape)    
    
