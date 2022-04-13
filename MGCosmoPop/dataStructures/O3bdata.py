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
#from pesummary.io import read
#import glob
   

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from astropy.cosmology import Planck15
from cosmology.cosmo import Cosmo

     

class O3bData(LVCData):
    
    def __init__(self, fname, suffix_name = 'nocosmo', **kwargs):#nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='skip' ):
        
        self.suffix_name = suffix_name
        import pandas as pd
        self.post_file_extension='.h5'
        self.metadata = pd.read_csv(os.path.join(fname, 'GWTC-3-confident.csv'))
        LVCData.__init__(self, fname, **kwargs)
        
        
    
    def _set_Tobs(self):
        self.Tobs= 147.083/365.
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
      
  
