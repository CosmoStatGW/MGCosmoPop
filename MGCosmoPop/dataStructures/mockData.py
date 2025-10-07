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
    
    def __init__(self, fname, 
                 nObsUse=None, 
                 nSamplesUse=None, 
                 percSamplesUse=None, 
                 dist_unit=u.Gpc, 
                 Tobs=2.5,  
                 SNR_th=8., 
                 events_use_idxs = None, 
                 events_not_use_idxs=None, 
                dLprior = None, 
                 inclination=False, 
                 which_spins='none'
                
                ):
        

        self.which_spins=which_spins
        self.inclination=inclination
        self.dLprior=dLprior
        print("dL prior is %s"%dLprior)
        
        self.SNR_th= SNR_th
        self.dist_unit = dist_unit
        self.m1z, self.m2z, self.dL, self.ra, self.dec, self.inclination, self.spins, self.snr, self.Nsamples, self.bin_weights = self._load_data(fname, nObsUse, events_use_idxs=events_use_idxs, events_not_use_idxs=events_not_use_idxs ) #nSamplesUse, )  
        print('snr shape: %s' %str(self.snr.shape))
        
        self.logNsamples = np.log(self.Nsamples)

        if nSamplesUse is not None or percSamplesUse is not None:
            self.downsample(nSamples=nSamplesUse, percSamples=percSamplesUse)
            print('Number of samples for each event after downsamplng: %s' %self.Nsamples )
        
        # Impose cut on SNR
        self.set_snr_threshold(SNR_th)
        

        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<=self.m1z).all()
        
        self.Tobs=Tobs
        #self.chiEff = np.zeros(self.m1z.shape)
        #self.spins = []
        self.events = [str(i) for i in range(len(self.dL))] #np.arange(len(self.dL))
        print('Obs time: %s' %self.Tobs )
        print('N. of events: %s' %len(self.events) )
        
    
    def set_snr_threshold(self, snr_th):
        if self.snr.sum()==0.:
            print('Snrs not present in this dataset.')
            setsnr = False
        if snr_th<self.SNR_th:
            #raise ValueError('New snr threshold is lower than original one !')
            print('warning: New snr threshold is lower than original one ! Using original')
            setsnr = False
        
        if setsnr:
            print('Setting snr threshold to %s' %snr_th)
            self.SNR_th=snr_th
            keep = self.snr >= snr_th

            self.m1z, self.m2z, self.dL, self.ra, self.dec, self.inclination, self.spins, self.snr, self.bin_weights = self.m1z[keep, :], self.m2z[keep, :], self.dL[ keep, :], self.ra[keep, :], self.dec[keep, :], self.inclination[keep, :], [self.spins[i][keep, :] for i in range(len(spins))], self.snr[keep], self.bin_weights[keep]
            self.logNsamples = self.logNsamples[keep]
            self.Nsamples = self.Nsamples[keep]
    
        self.Nobs=self.m1z.shape[0]
        
        print('We have %s observations with SNR>%s' %(self.Nobs, self.SNR_th))
        print('Number of samples: %s' %self.Nsamples )
        

    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    
    def _load_data(self, fname, nObsUse, nSamplesUse=None, events_use_idxs=None, events_not_use_idxs=None):
        
        if events_not_use_idxs is not None and events_use_idxs is not None:
            raise ValueError('You cannot pass events_not_use_idxs and events_use_idxs at the same time !')
        
        print('Loading data...')
        #if nObsUse is None:
        #    nObsUse=None
        with h5py.File(fname, 'r') as phi: #observations.h5 has to be in the same folder as this code
                
                print('List of available entries:')
                for pn in phi['posteriors'].keys():
                    print(pn)
            
                m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :]# m1
                m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :] # m2
                
                try:
                    dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :]
                except:
                    try:
                        dl_samples = np.array(phi['posteriors']['dL'])[:nObsUse, :]
                    except:
                        raise ValueError('Neither dL nor dl present as key in this dataset')
                
                print('dl samples loaded shape: %s' %str(dl_samples.shape))
                try:
                    snrs = np.array(phi['posteriors']['rho'])[:nObsUse]
                #print(dl_samples.shape)
                except Exception as e:
                    print(e)
                    print('SNRs not present for this dataset. Use the same SNR threshold as the original injections.')
                    snrs = np.zeros(dl_samples.shape[0])
                print('snrs loaded shape: %s' %str(snrs.shape))
                try:
                    bin_weights = np.array(phi['posteriors']['bin_weights'])[:nObsUse]
                except Exception as e:
                    print(e)
                    print('No bin weights.')
                    bin_weights = np.ones(dl_samples.shape[0])
                    
                try:
                    ra = np.array(phi['posteriors']['ra'])[:nObsUse, :]# m1
                    dec = np.array(phi['posteriors']['dec'])[:nObsUse, :] 
                except:
                    try:
                        ra = np.array(phi['posteriors']['phi'])[:nObsUse, :]# m1
                        dec = np.pi/2-np.array(phi['posteriors']['theta'])[:nObsUse, :]
                        print('Added ra, dec from theta, phi')
                    except:
                        print('No ra, dec')
                        ra=np.zeros(m1det_samples.shape)
                        dec=np.zeros(m1det_samples.shape)
                if self.inclination:
                    print('Also loading inclination')
                    try:
                        iota = np.array(phi['posteriors']['iota'])[:nObsUse, :]
                    except:
                        print('Inclination not found. Check input data!')
                        iota=np.zeros(m1det_samples.shape)
                else:
                    iota=np.zeros(m1det_samples.shape)

                if self.which_spins=='none':
                    spins=[ np.zeros(m1det_samples.shape) ]
                elif self.which_spins=='aligned':
                    chi1z = np.array(phi['posteriors']['chi1z'])[:nObsUse, :]
                    chi2z = np.array(phi['posteriors']['chi2z'])[:nObsUse, :]
                    spins=[ chi1z, chi2z ]
                elif which_spins=='default':
                    try:
                        s1x = posterior_samples['spin_1x']
                        s2x = posterior_samples['spin_2x']
                        s1y = posterior_samples['spin_1y']
                        s2y = posterior_samples['spin_2y']
                        s1z = posterior_samples['spin_1z']
                        s2z = posterior_samples['spin_2z']
                        s1 = np.sqrt(s1x**2+s1y**2+s1z**2)
                        s2 = np.sqrt(s2x**2+s2y**2+s2z**2)
                        cost1 = posterior_samples['cos_tilt_1']
                        cost2 = posterior_samples['cos_tilt_2']
                        spins = [s1, s2, cost1, cost2]
                    except Exception as e:
                        print(e)
                        print(posterior_samples.dtype.fields.keys())
                        try:
                            s1 = posterior_samples['a_1']
                            s2 = posterior_samples['a_2']
                            cost1 = np.cos(posterior_samples['tilt_1'])
                            cost2 = np.cos(posterior_samples['tilt_2'])
                            spins = [s1, s2, cost1, cost2]

                        except Exception as e:
                            print(e)
                            raise ValueError()
        
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
        
        if  events_not_use_idxs is not None and events_use_idxs is None:
            events_use_idxs = np.array([i for i in np.arange(m1det_samples.shape[0]) if i not in  events_not_use_idxs]) #np.where( np.arange(m1det_samples.shape[0])!=events_not_use_idxs)[0]
        elif events_not_use_idxs is None and events_use_idxs is None:
            events_use_idxs = np.arange(m1det_samples.shape[0])
        
        print('Excluding events %s' %str(events_not_use_idxs))
        print('We finally use %s events '%len(events_use_idxs))
        return m1det_samples[events_use_idxs, :], m2det_samples[events_use_idxs, :], dl_samples[events_use_idxs, :], ra[events_use_idxs, :], dec[events_use_idxs, :], iota[events_use_idxs, :], [spins[i][events_use_idxs, :] for i in range(len(spins))], snrs[events_use_idxs], np.count_nonzero(m1det_samples[events_use_idxs, :], axis=-1), bin_weights[events_use_idxs] 
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        if self.dLprior==None:
            return np.zeros(self.dL.shape)
        elif self.dLprior=='dLsq':
            return 2*np.log(self.dL)
        else:
            raise ValueError()
    



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
        
        
        self.Tobs=Tobs
        self.spins = []# np.zeros(self.m1z.shape)
        print('Obs time: %s' %self.Tobs )
        
        if snr_th is not None:
            self.set_snr_threshold(snr_th)

        self.condition=np.full(self.m1z.shape, True)
        
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
                
                try:
                    dl_sel = np.array(f['dl'])[:nInjUse]
                except:
                    try:
                        dl_sel = np.array(f['dL'])[:nInjUse]
                    except:
                        raise ValueError('Neither dL nor dl present as key in this dataset')
                
                
                
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
                
                
                try:
                    dl_sel = np.array(f['dl'])
                except:
                    try:
                        dl_sel = np.array(f['dL'])
                    except:
                        raise ValueError('Neither dL nor dl present as key in this dataset')
                
                
                try:
                    weights_sel = np.array(f['wt'])
                    log_weights_sel = np.log(weights_sel)
                except KeyError:
                    log_weights_sel = np.array(f['logwt'])#[:nInjUse]
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
            print('Threshold snr is %s'%snr_th)
        if self.dist_unit==u.Mpc:
            dl_sel*=1e03
            
        #self.max_z = np.max(z)
        try:
            self.max_z=z_at_value(Planck15.luminosity_distance, dl_sel.max()*self.dist_unit)
            print('Max redshift of injections assuming Planck 15 cosmology: %s' %self.max_z)
        except:
            pass
        
        # Drop points in the unlikely case of m1==m2, to avoid crashes
        
        
        keep = np.full(m1_sel.shape, True) #m1_sel!=m2_sel
        throw = ~keep
        print('Dropping %s points with exactly equal masses' %str(throw.sum()) )
        N_gen -= throw.sum()
        if weights_sel is not None:
            weights_sel=weights_sel[keep]
        
        
        print('Number of total injections: %s' %N_gen)
        print('Number of detected injections: %s' %dl_sel[keep].shape[0])
        
        return m1_sel[keep], m2_sel[keep], dl_sel[keep], weights_sel, log_weights_sel[keep] , snr_sel[keep], N_gen, snr_th
      
    
    def originalMassPrior(self):
        return np.ones(self.m1z.shape)

    def originalDistPrior(self):
        return np.ones(self.dL.shape)    
    
