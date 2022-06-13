#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

from .ABSdata import LVCData, O3InjectionsData

import numpy as np
#import astropy.units as u
import h5py
import os
import sys
#from pesummary.io import read
#import glob
   

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#from astropy.cosmology import Planck15
#from cosmology.cosmo import Cosmo
import Globals
     

class O3aData(LVCData):
    
    def __init__(self, fname,  which_metadata='GWOSC', **kwargs): #nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.GWTC2_1 = ['GW190403_051519', 'GW190426_190642', 'GW190725_174728','GW190805_211137', 'GW190916_200658', 'GW190917_114630','GW190925_232845', 'GW190926_050336']
        self.post_file_extension='.h5'
        import pandas as pd
        if which_metadata=='GWOSC':
            print('Using SNRS and far from the public version of the GWTC-2 and GWTC-2.1- catalog from the GWOSC')
            self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-2.csv'))
            metadata_21 = pd.read_csv(os.path.join(fname, 'GWTC-2.1-confident.csv'))
            self.metadata = self.metadata.append(metadata_21)
        else:
            print('Using best SNRS and far from all pipelines as reported in the GWTC-3 catalog paper')
            self.metadata = pd.read_csv(os.path.join(Globals.dataPath, 'all_metadata_pipelines_best.csv'))
        
        #print(self.metadata[[ 'commonName', 'catalog.shortName','mass_1_source',  'mass_2_source',  'network_matched_filter_snr','luminosity_distance','redshift','far' ]])
        LVCData.__init__(self, fname, **kwargs) #nObsUse=nObsUse, nSamplesUse=nSamplesUse, dist_unit=dist_unit, events_use=events_use, which_spins=which_spins)
        
        
    
    def _set_Tobs(self):
        self.Tobs= 183.375/365.
        # O3a data is taken between 1 April 2019 15:00 UTC and 1 October 2019 15:00 UTC.
        # The duty cycle for the three detectors was 76% (139.5 days) for Virgo, 
        # 71% (130.3 days) for LIGO Hanford, and 76% (138.5 days) for LIGO Livingston. 
        # With these duty cycles, the full 3-detector network was in observing mode 
        # for 44.5% of the time (81.4 days). 
        # Moreover, for 96.9% of the time (177.3 days) at least one detector was
        # observing and for 81.9% (149.9 days) at least two detectors were observing.
    
    
    def _get_not_BBHs(self):
        # return ['GW190425', 'GW190426_152155', 'GW190814', 'GW190917_114630' ] #'GW190426_152155', 'GW190426']
        return ['GW190425', 'GW190426_152155', 'GW190917_114630' ] #'GW190426_152155', 'GW190426']

    
    
    def _name_conditions(self, f ):
        return ( ( 'prior' not in f.split('.')[0] ) & ('comoving' not in f.split('.')[0]) & (f.split('_')[0][:2]=='GW') ) or ('IGWN-GWTC2p1' in f) 
    
    
    def _get_name_from_fname(self, fname):
        if 'IGWN-GWTC2p1' not in fname:
            return fname.split('.')[0]
        else:
            return '_'.join(fname.split('-')[-1].split('_')[:-1])
    
    def _load_data_event(self, fname, event, nSamplesUse, which_spins='skip'):
        if event in self.GWTC2_1:
            return self._load_data_event_GWTC2_1(fname, event, nSamplesUse, which_spins=which_spins)
        else:
            return self._load_data_event_GWTC2(fname, event, nSamplesUse, which_spins=which_spins)

    def _load_data_event_GWTC2_1(self, fname, event, nSamplesUse, which_spins='skip'):
        data_path = os.path.join(fname, 'IGWN-GWTC2p1-v1-'+event+'_PEDataRelease.h5')
        with h5py.File(data_path, 'r') as f:
            posterior_samples = f['IMRPhenomXPHM']['posterior_samples']
            _keys = ['mass_1', 'mass_2', 'luminosity_distance', 'right_ascension', 'declination']
            m1z, m2z, dL, ra, dec = [posterior_samples[k] for k in _keys]
            try:
                w = posterior_samples['weights_bin']
            except Exception as e:
                print(e)
                w = np.ones(1)
            if which_spins=='skip':
                spins=[]
            elif which_spins=='chiEff':
                chieff = posterior_samples['chi_eff']
                chiP = posterior_samples['chi_p']
                spins=[chieff, chiP]
            else:
                raise NotImplementedError()
            return m1z, m2z, dL, ra, dec, spins, w

    def _load_data_event_GWTC2(self, fname, event, nSamplesUse, which_spins='skip'):
        
        ### Using pesummary read function
        #data = read(os.path.join(fname,  event+self.post_file_extension))
        #samples_dict = data.samples_dict
        #posterior_samples = samples_dict['PublicationSamples']
        #m1z, m2z, dL, chieff = posterior_samples['mass_1'], posterior_samples['mass_2'], posterior_samples['luminosity_distance'],  posterior_samples['chi_eff'] 

        
        # By hand:
        data_path = os.path.join(fname,  event+self.post_file_extension)
        with h5py.File(data_path, 'r') as f:
            posterior_samples = f['PublicationSamples']['posterior_samples']      
            _keys = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec']
            m1z, m2z, dL, ra, dec = [posterior_samples[k] for k in _keys]
            print(dL, ra, dec)
            try:
                w = posterior_samples['weights_bin']
            except Exception as e:O1O2Data
                print(e)
                w = np.ones(1)
            if which_spins=='skip':
                spins=[]
            elif which_spins=='chiEff':
                chieff = posterior_samples['chi_eff']
                chiP = posterior_samples['chi_p']
                spins = [chieff, chiP]
            else:
                raise NotImplementedError()
        
        # Downsample if needed
        #all_ds = self._downsample( [m1z, m2z, dL, w, *spins], nSamplesUse)
        
        #m1z = all_ds[0]
        #m2z= all_ds[1]
        #dL =  all_ds[2]
        #w = all_ds[3]
        #spins = all_ds[4:]
        
        return m1z, m2z, dL, ra, dec, spins, w
              
   
    
   
    
class O3aInjectionsData(O3InjectionsData, ):
    
    def __init__(self, fname, **kwargs):
        self.Tobs=183.375/365.
        print('Obs time: %s yrs' %self.Tobs )
        
        O3InjectionsData.__init__(self, fname, **kwargs)
        
        