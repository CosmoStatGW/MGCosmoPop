#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

from .ABSdata import  LVCData, O3InjectionsData

import numpy as np
import h5py
import os
import sys
#from pesummary.io import read
#import glob
   

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#import astropy.units as u
#from astropy.cosmology import Planck15
#from cosmology.cosmo import Cosmo


import Globals
     

class O4aData(LVCData):
    
    def __init__(self, fname, metadata = 'GWTC-4.0.csv', **kwargs):#nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.suffix_name = ''
        import pandas as pd
        self.post_file_extension='.hdf5'
        if metadata is not None:
            self.metadata = pd.read_csv(metadata)
        else:
            try:
                self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-4.0.csv'))
            except:
                raise ValueError('Metadata for GWTC4 not found')
        # if which_metadata=='GWOSC':
        #     print('Using SNRS and far from the public version of the GWTC-3 catalog from the GWOSC')
        #     self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-4.0.csv'))
        # else:
        #     print('Using best SNRS and far from all pipelines as reported in the GWTC-3 catalog paper')
        #     self.metadata = pd.read_csv(os.path.join(Globals.dataPath, 'all_metadata_pipelines_best.csv'))
        LVCData.__init__(self, fname, **kwargs)
        
        
    
    def _set_Tobs(self):
        self.Tobs= (1389456018  - 1368975618)/60/60/24/365 # difference between the two GPS times below, in sec
        #O4 began on 2023 May 24 at 15:00:00 UTC.  GPS = 1368975618
        # This run is
        # again divided into parts: the first part of the fourth observing run (O4a) ended on 2024 January 16 at 16:00:00 UTC
        # GPS =1389456018
    
    
    def _get_not_BBHs(self):
        return ['GW230518_125908','GW230529_181500', ]
        # events in this line have secondary mass compatible with NS

        #+['GW190413_05954', 'GW190426_152155', 'GW190719_215514', 'GW190725_174728', 'GW190731_140936', 'GW190805_211137', 'GW190917_114630', 'GW191103_012549', 'GW200216_220804' ] 
        # events in the second list are those with ifar>=1yr, table 1 of 2111.03634
    
    
    def _name_conditions(self, f ):
        return self.suffix_name in f
    
    
    def _get_name_from_fname(self, fname):
        # IGWN-GWTC4p0-0f954158d_720-GW230627_015337-combined_PEDataRelease.hdf5
        return ('_').join(fname.split('-')[-2].split('_')[:2] )
    
    
    def _load_data_event(self, fname, event, nSamplesUse, which_spins='skip'):
        
        ### Using pesummary read function
        #data = read(os.path.join(fname,  event+self.post_file_extension))
        #samples_dict = data.samples_dict
        #posterior_samples = samples_dict['PublicationSamples']
        #m1z, m2z, dL, chieff = posterior_samples['mass_1'], posterior_samples['mass_2'], posterior_samples['luminosity_distance'],  posterior_samples['chi_eff'] 

        
        # By hand:
        data_path = os.path.join(fname,  'IGWN-GWTC4p0-0f954158d_720-'+event+'-combined_PEDataRelease'+self.post_file_extension)
        with h5py.File(data_path, 'r') as f:
            print('opening %s'%data_path)
            dataset = f['C00:IMRPhenomXPHM-SpinTaylor']
            posterior_samples = dataset['posterior_samples']
            
            _keys = ['mass_1', 'mass_2', 'luminosity_distance', 'iota']
            m1z, m2z, dL, iota = [posterior_samples[k] for k in _keys]
            
            #m1z = posterior_samples['mass_1']
            #m2z = posterior_samples['mass_2']
            #dL = posterior_samples['luminosity_distance']
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
                spins = [chieff, chiP]
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
                    raise ValueError()
            else:
                raise NotImplementedError()
            try:
                ra, dec  = posterior_samples['right_ascension'], posterior_samples['declination']
            except Exception as e:
                print(e)
                #ra, dec = np.empty(dL.shape), np.empty(dL.shape)
                ra = np.full( m1z.shape[0], 0.)
                dec = np.full( m1z.shape[0], 0.)
        
        # Downsample if needed
        #all_ds = self._downsample( [m1z, m2z, dL, w, *spins], nSamplesUse)
        
        #m1z = all_ds[0]
        #m2z= all_ds[1]
        #dL =  all_ds[2]
        #spins = all_ds[4:]
        #w = all_ds[3]
        
        return np.squeeze(m1z), np.squeeze(m2z), np.squeeze(dL), np.squeeze(ra), np.squeeze(dec), np.squeeze(iota), [np.squeeze(s) for s in spins], w
              
    
    
class O4aInjectionsData(O3InjectionsData, ):
    
    def __init__(self, fname, **kwargs):
        self.Tobs=(1389456018  - 1368975618)/60/60/24/365
        print('Obs time: %s yrs' %self.Tobs )
        
        O3InjectionsData.__init__(self, fname, **kwargs)

