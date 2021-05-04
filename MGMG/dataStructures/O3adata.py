#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:58:52 2021

@author: Michi
"""

from .ABSdata import Data, LVCData

import numpy as np
import astropy.units as u
import h5py
import os
import sys
#from pesummary.io import read
import glob
   

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from astropy.cosmology import Planck15
from cosmology.cosmo import Cosmo

     

class O3aData(LVCData):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.post_file_extension='.h5'
        LVCData.__init__(self, fname, nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, events_use=events_use, which_spins=which_spins)
        
        
    
    def _set_Tobs(self):
        self.Tobs= 183.3/365.
        # O3a data is taken between 1 April 2019 15:00 UTC and 1 October 2019 15:00 UTC.
        # The duty cycle for the three detectors was 76% (139.5 days) for Virgo, 
        # 71% (130.3 days) for LIGO Hanford, and 76% (138.5 days) for LIGO Livingston. 
        # With these duty cycles, the full 3-detector network was in observing mode 
        # for 44.5% of the time (81.4 days). 
        # Moreover, for 96.9% of the time (177.3 days) at least one detector was
        # observing and for 81.9% (149.9 days) at least two detectors were observing.
    
    
    def _get_not_BBHs(self):
        return ['GW190425', ]#'GW190426_152155', 'GW190426']
    
    
    def _name_conditions(self, f ):
        return ( ( 'prior' not in f.split('.')[0] ) & ('comoving' not in f.split('.')[0]) & (f.split('_')[0][:2]=='GW') )
    
    
    def _get_name_from_fname(self, fname):
        return fname.split('.')[0]
    
    
    def _load_data_event(self, fname, event, nSamplesUse, which_spins='skip'):
        
        ### Using pesummary read function
        #data = read(os.path.join(fname,  event+self.post_file_extension))
        #samples_dict = data.samples_dict
        #posterior_samples = samples_dict['PublicationSamples']
        #m1z, m2z, dL, chieff = posterior_samples['mass_1'], posterior_samples['mass_2'], posterior_samples['luminosity_distance'],  posterior_samples['chi_eff'] 

        
        # By hand:
        data_path = os.path.join(fname,  event+self.post_file_extension)
        with h5py.File(data_path, 'r') as f:
            dataset = f['PublicationSamples']
            posterior_samples = dataset['posterior_samples']
            
            m1z = posterior_samples['mass_1']
            m2z = posterior_samples['mass_2']
            dL = posterior_samples['luminosity_distance']
            if which_spins=='skip':
                spins=[]
            elif which_spins=='chiEff':
                chieff = posterior_samples['chi_eff']
                chiP = posterior_samples['chi_p']
                spins = [chieff, chiP]
            else:
                raise NotImplementedError()
        
        # Downsample if needed
        all_ds = self.downsample( [m1z, m2z, dL, *spins], nSamplesUse)
        
        m1z = all_ds[0]
        m2z= all_ds[1]
        dL =  all_ds[2]
        spins = all_ds[3:]
        
        return m1z, m2z, dL, spins
              
    
    

    
    
    
    
class O3InjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None,  dist_unit=u.Gpc, ifar_th=1., which_spins='skip' ):
        
        self.which_spins=which_spins
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.spins, self.log_weights_sel, self.N_gen, self.Tobs, conditions_arr = self._load_data(fname, nInjUse, which_spins=which_spins )        
        self.logN_gen = np.log(self.N_gen)
        #self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<self.m1z).all()
        
        self.Tobs=183.3/365. #0.5
        #self.chiEff = np.zeros(self.m1z.shape)
        print('Obs time: %s yrs' %self.Tobs )
        
        self.ifar_th=ifar_th
        gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar = conditions_arr
        self.condition = (gstlal_ifar > ifar_th) | (pycbc_ifar > ifar_th) | (pycbc_bbh_ifar > ifar_th)
        
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL, self.spins  ] )  
    
    
    def _load_data(self, fname, nInjUse, which_spins='skip'):
        
        with h5py.File(fname, 'r') as f:
        
            Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
            Ndraw = f.attrs['total_generated']
    
            m1 = np.array(f['injections/mass1_source'])
            m2 = np.array(f['injections/mass2_source'])
            z = np.array(f['injections/redshift'])
            
            if which_spins=='skip':
                spins=[]
            else:
                chi1z = np.array(f['injections/spin1z'])
                chi2z = np.array(f['injections/spin2z'])
                if which_spins=='chiEff':
                    q = m2/m1
                    chiEff = (chi1z+q*chi2z)/(1+q)
                    print('chi_p not available for O3 selection effects ! ')
                    spins=[chiEff, np.full(chiEff.shape, np.NaN)]
                elif which_spins=='s1s2':
                    raise NotImplementedError()
                    spins=[chi1z, chi2z]
    
            p_draw = np.array(f['injections/sampling_pdf'])
            if which_spins=='skip':
                print('Removing factor of 1/2 for each spin dimension from p_draw...')
                p_draw *= 4
            log_p_draw = np.log(p_draw)
        
            gstlal_ifar = np.array(f['injections/ifar_gstlal'])
            pycbc_ifar = np.array(f['injections/ifar_pycbc_full'])
            pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
        
            m1z = m1*(1+z)
            m2z = m2*(1+z)
            dL = np.array(Planck15.luminosity_distance(z).to(self.dist_unit).value)
            
                
            #dL = np.array(f['injections/distance']) #in Mpc for GWTC2 !
            #if self.dist_unit==u.Gpc:
            #    print('Converting original distance in Mpc to Gpc ...')
            #    dL*=1e-03
        
            print('Re-weighting p_draw to go to detector frame quantities...')
            myCosmo = Cosmo(dist_unit=self.dist_unit)
            #p_draw /= (1+z)**2
            #p_draw /= myCosmo.ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0) #z, H0, Om, w0, Xi0, n
            log_p_draw -=2*np.log1p(z)
            log_p_draw -= myCosmo.log_ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0. )
        

            print('Number of total injections: %s' %Ndraw)
            print('Number of injections that pass first threshold: %s' %p_draw.shape[0])
            
            self.max_z = np.max(z)
            print('Max redshift of injections: %s' %self.max_z)
            return m1z, m2z, dL , spins, log_p_draw , Ndraw, Tobs, (gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar)
      
  