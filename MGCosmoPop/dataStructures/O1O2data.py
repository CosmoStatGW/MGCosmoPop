#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

from .ABSdata import Data, LVCData

import numpy as np
import astropy.units as u
import h5py
import os
import sys


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from astropy.cosmology import Planck15
from cosmology.cosmo import Cosmo

import Globals

     
class O1O2Data(LVCData):
    
    def __init__(self, fname, which_metadata='GWOSC', force_BNS=False, **kwargs):#nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.post_file_extension='.hdf5'
        self.force_BNS=force_BNS
        
        import pandas as pd
        
        if which_metadata=='GWOSC':
            print('Using SNRS and far from the public version of the GWTC-3 catalog from the GWOSC')
            self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-1-confident.csv'))
        else:
            print('Using best SNRS and far from all pipelines as reported in the GWTC-3 catalog paper')
            self.metadata = pd.read_csv(os.path.join(Globals.dataPath, 'all_metadata_pipelines_best.csv'))
        
        LVCData.__init__(self, fname, **kwargs) #nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, events_use=events_use, which_spins=which_spins)
        
        
        
    def _set_Tobs(self):
        # The first observing run (O1) ran from September 12th, 2015 to January 19th, 2016 --> 129 days
        # From https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.041015: 
        # after data quality flags, the remaining coincident analysis time in O1 is 48.3 days with GSTLal analysis, 46.1 with pycbc
        
        # The second observing run (O2) ran from November 30th, 2016 to August 25th, 2017 --> 267 days
        # During the O2 run the duty cycles were 62% for LIGO Hanford and 61% for LIGO Livingston, 
        # so that two detectors were in observing mode 46.4% of the time and at least one detector 
        # was in observing mode 75.6% of the time.
        # From https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031040 :
        # During O2, the individual LIGO detectors had duty factors of approximately 60% with a LIGO 
        # network duty factor of about 45%. Times with significant instrumental disturbances are flagged and removed, 
        # resulting in about 118 days of data suitable for coincident analysis
        
        #self.Tobs= (48.3+118)/365.  # yrs
        self.Tobs= (129+267)/365.  # yrs
    

    
    def _get_not_BBHs(self):
        if not self.force_BNS:
            return ['GW170817', ]
        else:
            return []
        
    
    def _name_conditions(self, f ):
        return ( ( 'prior' not in f.split('.')[0] ) &  (f.split('_')[0][:2]=='GW') )
    
    
    def _get_name_from_fname(self, fname):
        return fname.split('.')[0].split('_')[0]
    
    
    def _load_data_event(self, fname, event, nSamplesUse, which_spins='skip'):
        
        data_path = os.path.join(fname,  event+'_GWTC-1'+self.post_file_extension)
        
        with h5py.File(data_path, 'r') as f:
            try:
                posterior_samples = f['Overall_posterior']
            except Exception as e:
                print(e)
                print(f.keys())
            _keys = ['m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'luminosity_distance_Mpc', 
                     'right_ascension', 'declination', 'costheta_jn']
            m1z, m2z, dL, ra, dec, costh = [posterior_samples[k] for k in _keys]
            try:
                w = posterior_samples['weights_bin']
            except Exception as e:
                print(e)
                w = np.ones(1)
                
            if which_spins=='skip':
                spins=[]
            elif which_spins=='chiEff':
                #print('chi_p not available for O1-O2 data ! ')
                s1 = posterior_samples['spin1']
                s2 = posterior_samples['spin2']
                cost1 = posterior_samples['costilt1']
                cost2 = posterior_samples['costilt2']
                sint1 = np.sqrt(1-cost1**2)
                sint2 = np.sqrt(1-cost2**2)
                chi1z = s1*cost1
                chi2z = s2*cost2
                q = m2z/m1z
                chiEff = (chi1z+q*chi2z)/(1+q)
                
                chiP = np.max( np.array([s1*sint1, (4*q+3)/(4+3*q)*q*s2*sint2 ]) , axis=0 )
                
                spins=[chiEff, chiP]
            elif which_spins=='s1s2':
                raise NotImplementedError()
                s1 = posterior_samples['spin1']
                s2 = posterior_samples['spin2']
                spins=[s1,s2]
            elif which_spins=='chi1zchi2z':
                s1 = posterior_samples['spin1']*posterior_samples['costilt1']
                s2 = posterior_samples['spin2']*posterior_samples['costilt2']
                spins=[s1,s2]
            elif which_spins=='default':
                s1 = posterior_samples['spin1']
                s2 = posterior_samples['spin2']
                cost1 = posterior_samples['costilt1']
                cost2 = posterior_samples['costilt2']
                spins = [s1, s2, cost1, cost2]
                
         
        
        iota = np.arccos(costh)
        # Downsample if needed
        #all_ds = self._downsample( [m1z, m2z, dL, w, *spins,], nSamplesUse)
        
        #m1z = all_ds[0]
        #m2z= all_ds[1]
        #dL =  all_ds[2]
        #spins = all_ds[4:]
        #ws = all_ds[3]
        
        return m1z, m2z, dL, ra, dec, iota, spins, w
    
 
    
 
    
 
    
 
class O1O2InjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None,  dist_unit=u.Gpc, ifar_th=1 , which_spins='skip', snr_th=None ):
        
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.spins, self.log_weights_sel, self.N_gen, self.Tobs = self._load_data(fname, nInjUse, which_spins=which_spins )        
        self.logN_gen = np.log(self.N_gen)
        #self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<self.m1z).all()
        
        #self.Tobs=0.5
        self.chiEff = np.zeros(self.m1z.shape)
        print('Obs time: %s yrs' %self.Tobs )
        
        self.ifar_th=ifar_th
        
        #gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar = conditions_arr
        self.condition = np.full(self.m1z.shape, True)
        #(gstlal_ifar > ifar_th) | (pycbc_ifar > ifar_th) | (pycbc_bbh_ifar > ifar_th)
        # np.full(self.m1z.shape, True) #
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    
    def _load_data(self, fname, nInjUse, which_spins='skip'):
        
        with h5py.File(fname, 'r') as f:
            print(f.attrs.keys())
            print(f.keys())
            
            Tobs = 1.084931506849315 #f.attrs['analysis_time_s']/(365.25*24*3600) # years  (48.3+118)/365. #
            Ndraw = 7.1e07 #f.attrs['total_generated']
    
            m1 = np.array(f['mass1_source'])
            m2 = np.array(f['mass2_source'])
            z = np.array(f['redshift'])
        #s1z = np.array(f['injections/spin1z'])
        #s2z = np.array(f['injections/spin2z'])
            if which_spins=='skip':
                spins=[]
            elif which_spins=='chiEff':
                chi1z = np.array(f['spin1z'])
                chi2z = np.array(f['spin2z'])   
                q = m2/m1
                chiEff = (chi1z+q*chi2z)/(1+q)
                print('chi_p not available for O2 selection effects ! ')
                spins=[chiEff, np.full(chiEff.shape, np.NaN)]
                #raise NotImplementedError()
            elif which_spins=='s1s2':
                raise NotImplementedError()
                s1 = np.array(f['spin1z'])
                s2 = np.array(f['spin2z'])
                spins=[s1,s2]
    
            p_draw = np.array(f['sampling_pdf'])
            if which_spins=='skip':
                print('Removing factor of 1/2 for each spin dimension from p_draw...')
                p_draw *= 4
            log_p_draw = np.log(p_draw)
            
            #gstlal_ifar = np.array(f['injections/ifar_gstlal'])
            #pycbc_ifar = np.array(f['injections/ifar_pycbc_full'])
            #pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
        
            m1z = m1*(1+z)
            m2z = m2*(1+z)
            dL = np.array(Planck15.luminosity_distance(z).to(self.dist_unit).value)
            #dL = np.array(f['injections/distance']) #in Mpc for GWTC2 !
            #if self.dist_unit==u.Gpc:
            #    dL*=1e-03
        
            print('Re-weighting p_draw to go to detector frame quantities...')
            myCosmo = Cosmo(dist_unit=self.dist_unit)
            #p_draw/=(1+z)**2
            #p_draw/=myCosmo.ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0) #z, H0, Om, w0, Xi0, n
            log_p_draw -=2*np.log1p(z)
            log_p_draw -= myCosmo.log_ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0. )
        

            print('Number of total injections: %s' %Ndraw)
            print('Number of injections that pass first threshold: %s' %p_draw.shape[0])
            
            self.max_z = np.max(z)
            print('Max redshift of injections: %s' %self.max_z)
            return m1z, m2z, dL , spins, log_p_draw , Ndraw, Tobs#, (gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar)

   
