#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:58:52 2021

@author: Michi
"""

from .ABSdata import Data

import numpy as np
import astropy.units as u
import h5py
import os
import sys
from pesummary.io import read
import glob
   

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from astropy.cosmology import Planck15
from cosmology.cosmo import Cosmo


O3BNS = ('GW190425', )
O3BHNS = ('GW190426_152155', 'GW190426' )

     
class O3aData(Data):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, dist_unit=u.Gpc, events_use=None ):
        
        
        self.events_use=events_use
        
        if events_use['use'] is not None:
            nObsUse=len(events_use['use'])
        
        self.dist_unit = dist_unit
        self.events = self._get_events(fname, events_use)
        
        self.m1z, self.m2z, self.dL, self.chiEff, self.Nsamples = self._load_data(fname, nObsUse, nSamplesUse, )  
        self.Nobs=self.m1z.shape[0]
        #print('We have %s observations' %self.Nobs)
        print('Number of samples used: %s' %self.Nsamples )
        
        self.logNsamples = np.log(self.Nsamples)
        #assert (self.m1z >= 0).all()
        #assert (self.m2z >= 0).all()
        #assert (self.dL >= 0).all()
        #assert(self.m2z<=self.m1z).all()
        
        self.Tobs= 0.5 #O3a data is taken between 1 April 2019 15:00 UTC and 1 October 2019 15:00 UTC.
        #self.chiEff = np.zeros(self.m1z.shape)
        print('Obs time (yrs): %s' %self.Tobs )
        
        self.Nobs=self.m1z.shape[0]
        
    
    def _get_events(self, fname, events_use):
        
        #dirlist = [ item for item in os.listdir(fname) if os.path.isdir(os.path.join(fname, item)) ]        
        #print(fname)
        #print(os.listdir(fname))
        #print(os.path.join(fname, '*.h5' ))
        allFiles = [f for f in os.listdir(fname) if f.endswith('.h5')] #glob.glob(os.path.join(fname, '*.h5' ))
        #print(allFiles)
        #print([len(f.split('.')[0].split('_')) for f in allFiles])
        #print([f.split('_')[0][:2] for f in allFiles])
        elist = [f.split('.')[0] for f in allFiles if ( ( 'prior' not in f.split('.')[0] ) & ('comoving' not in f.split('.')[0]) & (f.split('_')[0][:2]=='GW') ) ]
        
        
        list_BBH = [x for x in elist if x not in O3BNS+O3BHNS]
        print('In the O3a data we have the following BBH events, total %s (excluding %s):' %(len(list_BBH) ,str(O3BNS+O3BHNS)) )
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
        
    
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL , self.chiEff ] )  
    
    
    def _load_data(self, fname, nObsUse, nSamplesUse,):
        print('Loading data...')
    
        
        #events = self._get_events_names(fname)
        if nObsUse is None:
            nObsUse=len(self.events)
            
        
        #print('We have the following events: %s' %str(events))
        m1s, m2s, dLs, chiEffs = [], [], [], []
        allNsamples=[]
        for event in self.events[:nObsUse]:
                print('Reading data from %s' %event)
            #with h5py.File(fname, 'r') as phi:
                m1z_, m2z_, dL_, chiEff_  = self._load_data_event(fname, event, nSamplesUse)
                print('NUmber of samples in LVC data: %s' %m1z_.shape[0])
                m1s.append(m1z_)
                m2s.append(m2z_)
                dLs.append(dL_)
                chiEffs.append(chiEff_)
                assert len(m1z_)==len(m2z_)
                assert len(m2z_)==len(dL_)
                assert len(chiEff_)==len(dL_)
                nSamples = len(m1z_)
                
                allNsamples.append(nSamples)
            #print('ciao')
        print('We have %s events.'%len(allNsamples))
        max_nsamples = max(allNsamples) 
        
        fin_shape=(nObsUse,max_nsamples)
        
        m1det_samples= np.full(fin_shape, np.NaN)  #np.zeros((len(self.events),max_nsamples))
        m2det_samples=np.full(fin_shape, np.NaN)
        dl_samples= np.full(fin_shape, np.NaN)
        chiEff_samples= np.full(fin_shape, np.NaN)
        
        for i in range(nObsUse):
            
            m1det_samples[i, :allNsamples[i]] = m1s[i]
            m2det_samples[i, :allNsamples[i]] = m2s[i]
            dl_samples[i, :allNsamples[i]] = dLs[i]
            chiEff_samples[i, :allNsamples[i]] = chiEffs[i]
        
        if self.dist_unit==u.Gpc:
            print('Using distances in Gpc')   
            dl_samples*=1e-03
        
        return m1det_samples, m2det_samples, dl_samples, chiEff_samples, allNsamples
      
    
    
    def _load_data_event(self, fname, event, nSamplesUse):
        
        data = read(os.path.join(fname,  event+'.h5'))

        samples_dict = data.samples_dict
        posterior_samples = samples_dict['PublicationSamples']

        #parameters = sorted(list(posterior_samples.keys()))

        m1z, m2z, dL, chieff = posterior_samples['mass_1'], posterior_samples['mass_2'], posterior_samples['luminosity_distance'],  posterior_samples['chi_eff'] 
        
        m1z, m2z, dL, chieff = self.downsample( [m1z, m2z, dL, chieff], nSamplesUse)
        
        return m1z, m2z, dL, chieff
            
        
    
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        # dl^2 prior on dL
        return np.where( ~np.isnan(self.dL), 2*np.log(self.dL), 0)
    
    
    
    
    
class O3InjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None,  dist_unit=u.Gpc, ifar_th=1 ):
        
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.weights_sel, self.N_gen, self.Tobs, conditions_arr = self._load_data(fname, nInjUse )        
        self.logN_gen = np.log(self.N_gen)
        self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<self.m1z).all()
        
        #self.Tobs=0.5
        self.chiEff = np.zeros(self.m1z.shape)
        print('Obs time: %s' %self.Tobs )
        
        self.ifar_th=ifar_th
        gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar = conditions_arr
        self.condition = (gstlal_ifar > ifar_th) | (pycbc_ifar > ifar_th) | (pycbc_bbh_ifar > ifar_th)
        
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] )  
    
    
    def _load_data(self, fname, nInjUse,):
        
        with h5py.File(fname, 'r') as f:
        
            Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
            Ndraw = f.attrs['total_generated']
    
            m1 = np.array(f['injections/mass1_source'])
            m2 = np.array(f['injections/mass2_source'])
            z = np.array(f['injections/redshift'])
        #s1z = np.array(f['injections/spin1z'])
        #s2z = np.array(f['injections/spin2z'])
    
            p_draw = np.array(f['injections/sampling_pdf'])
    
            gstlal_ifar = np.array(f['injections/ifar_gstlal'])
            pycbc_ifar = np.array(f['injections/ifar_pycbc_full'])
            pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
        
            m1z = m1*(1+z)
            m2z = m2*(1+z)
            #dL = Planck15.luminosity_distance(z).to(Globals.which_unit).value
            dL = np.array(f['injections/distance']) #in Mpc for GWTC2 !
            if self.dist_unit==u.Gpc:
                dL*=1e-03
        
            print('Re-weighting p_draw to go to detector frame quantities...')
            myCosmo = Cosmo()
            p_draw/=(1+z)**2
            p_draw/=myCosmo.ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0) #z, H0, Om, w0, Xi0, n

        

            print('Number of total injections: %s' %Ndraw)
            print('Number of injections that pass first threshold: %s' %p_draw.shape[0])
            return m1z, m2z, dL , p_draw , Ndraw, Tobs, (gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar)
      
    
    def originalMassPrior(self):
        return np.ones(self.m1z.shape)

    def originalDistPrior(self):
        return np.ones(self.dL.shape)    
    
