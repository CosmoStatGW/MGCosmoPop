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
     

class O3bData(LVCData):
    
    def __init__(self, fname, suffix_name = 'nocosmo', which_metadata = 'GWOSC', **kwargs):#nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.suffix_name = suffix_name
        import pandas as pd
        self.post_file_extension='.h5'
        if which_metadata=='GWOSC':
            print('Using SNRS and far from the public version of the GWTC-3 catalog from the GWOSC')
            self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-3-confident.csv'))
        else:
            print('Using best SNRS and far from all pipelines as reported in the GWTC-3 catalog paper')
            self.metadata = pd.read_csv(os.path.join(Globals.dataPath, 'all_metadata_pipelines_best.csv'))
        LVCData.__init__(self, fname, **kwargs)
        
        
    
    def _set_Tobs(self):
        self.Tobs= 147.083/365. # difference between the two GPS times below, in sec
        # O3b dates: 1st November 2019 15:00 UTC (GPS 1256655618) to 27th March 2020 17:00 UTC (GPS 1269363618)
        # 147.083
        # 148 days in total
        # 142.0 days with at least one detector for O3b
        
        # second half of the third observing run (O3b) between 1 November 2019, 15:00 UTC and 27 March 2020, 17:00 UTC
        # for 96.6% of the time (142.0 days) at least one interferometer was observing,
        # while for 85.3% (125.5 days) at least two interferometers were observing
    
    
    def _get_not_BBHs(self):
        return ['GW200115_042309','GW200105_162426', 'GW191219_163120', 'GW200210_092254', 'GW200210_092255', 'GW190917_114630' ]
        # events in this line have secondary mass compatible with NS

        #+['GW190413_05954', 'GW190426_152155', 'GW190719_215514', 'GW190725_174728', 'GW190731_140936', 'GW190805_211137', 'GW190917_114630', 'GW191103_012549', 'GW200216_220804' ] 
        # events in the second list are those with ifar>=1yr, table 1 of 2111.03634
    
    
    def _name_conditions(self, f ):
        return self.suffix_name in f
    
    
    def _get_name_from_fname(self, fname):
        return ('_').join(fname.split('-')[-1].split('_')[:2] )
    
    
    def _load_data_event(self, fname, event, nSamplesUse, which_spins='skip'):
        
        ### Using pesummary read function
        #data = read(os.path.join(fname,  event+self.post_file_extension))
        #samples_dict = data.samples_dict
        #posterior_samples = samples_dict['PublicationSamples']
        #m1z, m2z, dL, chieff = posterior_samples['mass_1'], posterior_samples['mass_2'], posterior_samples['luminosity_distance'],  posterior_samples['chi_eff'] 

        
        # By hand:
        data_path = os.path.join(fname,  'IGWN-GWTC3p0-v1-'+event+'_PEDataRelease_mixed_'+self.suffix_name+self.post_file_extension)
        with h5py.File(data_path, 'r') as f:
            dataset = f['C01:IMRPhenomXPHM']
            posterior_samples = dataset['posterior_samples']
            
            m1z = posterior_samples['mass_1']
            m2z = posterior_samples['mass_2']
            dL = posterior_samples['luminosity_distance']
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
            else:
                raise NotImplementedError()
        
        # Downsample if needed
        #all_ds = self._downsample( [m1z, m2z, dL, w, *spins], nSamplesUse)
        
        #m1z = all_ds[0]
        #m2z= all_ds[1]
        #dL =  all_ds[2]
        #spins = all_ds[4:]
        #w = all_ds[3]
        
        return np.squeeze(m1z), np.squeeze(m2z), np.squeeze(dL), [np.squeeze(s) for s in spins], w
              
    
    
class O3bInjectionsData(O3InjectionsData, ):
    
    def __init__(self, fname, **kwargs):
        self.Tobs=147.083/365.
        print('Obs time: %s yrs' %self.Tobs )
        
        O3InjectionsData.__init__(self, fname, **kwargs)

      
  
