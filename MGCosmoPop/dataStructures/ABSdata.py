#!/usr/bin/env python3
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

from abc import ABC, abstractmethod
from scipy.stats import ks_2samp
import numpy as np
try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
import astropy.units as u
import os
import pandas as pd
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))



import Globals


class Data(ABC):
    
    def __init__(self, ):
        pass
        
    @abstractmethod
    def _load_data(self):
        pass
    
    @abstractmethod
    def get_theta(self):
        pass
    
    def downsample(self, nSamples=None, percSamples=None, verbose=True):
        if nSamples is None:
            return self._downsample_perc(percSamples, verbose=verbose)
        elif percSamples is None:
            return self._downsample_n(nSamples, verbose=verbose)
        else:
            raise ValueError('One among nSamples and percSamples should not be None')

    def _downsample_perc(self, percSamples, verbose=True):
        try:
            samples = np.array([self.m1z, self.m2z, self.dL,  *self.spins ])#self.Nsamples                                       
        except:
            print('No spins in this data')
            samples = np.array([self.m1z, self.m2z, self.dL, ])
        Npar = samples.shape[0]
        print('Npar: %s' %Npar)
        Nobs = samples.shape[1]
        print('Nobs in downsample: %s' %Nobs)
        print('Reducing all samples by %s percent' %(percSamples*100))
        self.Nsamples = (np.array(self.Nsamples)*percSamples).astype(int)
        self.logNsamples=np.log(self.Nsamples)
        maxNsamples = self.Nsamples.max()
        m1zD, m2zD, dLD =  np.full((Nobs,maxNsamples), np.nan), np.full((Nobs,maxNsamples), np.nan), np.full((Nobs,maxNsamples), np.nan) #np.zeros((Nobs,maxNsamples)), np.zeros((Nobs,maxNsamples)), np.zeros((Nobs,maxNsamples))
        if Npar==5:
            s0D, s1D = np.full((Nobs,maxNsamples), np.nan), np.full((Nobs,maxNsamples), np.nan)#np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples))
        for o in range(Nobs):
            if Npar==5:
                m1zD[o, :self.Nsamples[o]], m2zD[o, :self.Nsamples[o]], dLD[o, :self.Nsamples[o]],  s0D[o, :self.Nsamples[o]], s1D[o, :self.Nsamples[o]] = self._downsample(samples[:,o, :], self.Nsamples[o], verbose=verbose)
            elif Npar==3:
                m1zD[o, :self.Nsamples[o]], m2zD[o, :self.Nsamples[o]], dLD[o, :self.Nsamples[o]], = self._downsample(samples[:,o, :], self.Nsamples[o], verbose=verbose)
        if Npar==5:
            self.spins = [s0D, s1D]

        self.m1z = m1zD
        self.m2z = m2zD
        self.dL = dLD


    def _downsample_n(self, nSamples, verbose=True):
        
        try:
            
            samples = np.array([self.m1z, self.m2z, self.dL, self.ra, self.dec, self.iota, *self.spins ]) #self.Nsamples
            #print(self.spins)
            Npar = samples.shape[0]
            if len(self.spins)>0:
                spins=True
                print('Spins in this data')
            else:
                spins=False
                print('No spins in this data')
        except Exception as e:
            print(e)
            print('No spins in this data')
            samples = np.array([self.m1z, self.m2z, self.dL, self.ra, self.dec,])
            Npar = 5
            spins=False
        print('Npar: %s' %Npar)
        Nobs = samples.shape[1]
        
        print('Nobs in downsample: %s' %Nobs)
        m1zD, m2zD, dLD, raD, decD, iotaD  =  np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples))
        #if not spins:
            # no spins
        s0D, s1D, s2D, s3D = np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples)), np.zeros((Nobs,nSamples))
        for o in range(Nobs):
            print('Samples shape is %s' %str(samples.shape))
            if Nobs>200:
                if o>0:
                    verbose=False
            if not spins:
                m1zD[o], m2zD[o], dLD[o], raD[o], decD[o], iotaD[o] = self._downsample(samples[:,o, :], nSamples, verbose=verbose)

            else:
                m1zD[o], m2zD[o], dLD[o], raD[o], decD[o], iotaD[o], s0D[o], s1D[o], s2D[o], s3D[o] = self._downsample(samples[:,o, :], nSamples, verbose=verbose)
            
        
        #elif Npar==3:
            #    m1zD[o], m2zD[o], dLD[o] = self._downsample(samples[:,o, :], nSamples, verbose=verbose)
        #if Npar==5:
        self.spins = [s0D, s1D, s2D, s3D]
        self.m1z = m1zD
        self.m2z = m2zD
        self.dL = dLD
        self.ra = raD
        self.dec = decD
        self.iota = iotaD
        self.logNsamples=np.where(self.logNsamples<np.log(nSamples), self.logNsamples, np.log(nSamples)) #np.full(Nobs, np.log(nSamples))
        self.Nsamples=np.where(np.array(self.Nsamples)<nSamples, self.Nsamples, nSamples)
        
        

    def _downsample(self, posterior, nSamples,  verbose=True):
        
        if nSamples is None:
            return posterior
        if verbose:
            print('Downsampling posterior to %s samples...' %nSamples)
        
        posterior = np.array(posterior)
        #nparams=posterior.shape[0]
        nOrSamples=(~np.isnan(posterior)[0]).sum()
        if verbose:
            print('Number of original samples: %s '%nOrSamples)
        #print('posterior shape: %s' %str(posterior.shape))
        if len(posterior) == 1:
            if verbose:
                print('Using inverse sampling method')
            n, bins = np.histogram(posterior, bins=50)
            n = np.array([0] + [i for i in n])
            cdf = cumtrapz(n, bins, initial=0)
            cdf /= cdf[-1]
            icdf = interp1d(cdf, bins)
            samples = icdf(np.random.rand(nSamples))
        else:
            if nOrSamples<nSamples:
                nans_to_fill = nSamples-nOrSamples
                if verbose:
                    print('Using all original samples and filling with %s nans' %nans_to_fill)
                samples = []
                for i,p in enumerate(posterior):
                    #print(i)
                    #print('p shape: %s' %str(p.shape))
                    or_samples_position =  ~np.isnan(p)
                    new_p = np.concatenate([p[or_samples_position], np.full( nans_to_fill, np.nan)])
                    assert np.isnan(new_p).sum()==nans_to_fill
                    samples.append(new_p)
            else:
                if verbose:
                    print('Randomly choosing subset of samples')
                keep_idxs = np.random.choice(nOrSamples, nSamples, replace=False)
                #samples = [i[keep_idxs] for i in posterior]
                samples=[]
                for i, p in enumerate(posterior):
                    or_samples_position =  ~np.isnan(p)
                    or_samples = p[or_samples_position]
                    print('or_samples shape: %s' %str(or_samples.shape))
                    #keep_idxs = np.random.choice(nOrSamples, nSamples, replace=False)
                    assert ~np.any(np.isnan(or_samples[keep_idxs]))
                    samples.append(or_samples[keep_idxs])
                #print(len(samples))
        #print(samples[0].shape)
        return samples
    
    
    
    
class LVCData(Data):
    
    def __init__(self, fname, nObsUse=None, nSamplesUse=None, percSamplesUse=None, dist_unit=u.Gpc, events_use=None, which_spins='chiEff', SNR_th=8., FAR_th=1., BBH_only=True, dLprior='dLsq' ):
        
        Data.__init__(self)
        
        
        
        #self.metadata = pd.read_csv(os.path.join(Globals.dataPath, 'all_metadata_pipelines_best.csv'))
        
        self.dLprior=dLprior
        self.FAR_th = FAR_th
        self.SNR_th = SNR_th
        print('FAR th in LVC data: %s' %self.FAR_th)
        self.events_use=events_use
        self.which_spins=which_spins
        self.BBH_only=BBH_only
        
        nObsUse=None
        if events_use is not None:
            try:
                nObsUse=len(events_use['use'])
            except:
                pass
        
        self.dist_unit = dist_unit
        self.events = self._get_events(fname, events_use)
        
        self.m1z, self.m2z, self.dL, self.ra, self.dec, self.iota, self.spins, self.Nsamples, self.bin_weights = self._load_data(fname, nObsUse, which_spins=which_spins)  
        self.Nobs=self.m1z.shape[0]
        # assert len(self.bin_weights)==self.Nobs
        #print(self.bin_weights)
        #print('We have %s observations' %self.Nobs)
        print('Number of samples for each event: %s' %self.Nsamples )
        self.logNsamples = np.log(self.Nsamples)

        if nSamplesUse is not None or percSamplesUse is not None:
            self.downsample(nSamples=nSamplesUse, percSamples=percSamplesUse)
            print('Number of samples for each event after downsamplng: %s' %self.Nsamples )
        
        

        #assert (self.m1z >= 0).all()
        #assert (self.m2z >= 0).all()
        #assert (self.dL >= 0).all()
        #assert(self.m2z<=self.m1z).all()
        
        # The first observing run (O1) ran from September 12th, 2015 to January 19th, 2016 --> 129 days
        # The second observing run (O2) ran from November 30th, 2016 to August 25th, 2017 --> 267 days
        self._set_Tobs() #Tobs=  (129+267)/365. # yrs
        
        print('Obs time (yrs): %s' %self.Tobs )
        
        self.Nobs=self.m1z.shape[0]
    
    @abstractmethod
    def _name_conditions(self, f ):
        pass
    
    @abstractmethod
    def _set_Tobs(self):
        pass
    
    @abstractmethod
    def _get_not_BBHs(self):
        pass

    @abstractmethod
    def _load_data_event(self, **kwargs):
        pass
    
    
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL , self.spins ] )  
    
    @abstractmethod
    def _get_name_from_fname(self, fname):
        pass
    
    
    def _get_events(self, fname, events_use, ):
        
        
        allFiles = [f for f in os.listdir(fname) if f.endswith(self.post_file_extension)] #glob.glob(os.path.join(fname, '*.h5' ))
        #print(allFiles)
        elist = [self._get_name_from_fname(f) for f in allFiles if self._name_conditions(f) ]
        
        
        # Exclude events not identified as BBHs
        if self.BBH_only:
            list_BBH_or = [x for x in elist if x not in self._get_not_BBHs() ]
            print('In the '+fname.split('/')[-1]+' data we have the following BBH events, total %s (excluding %s):' %(len(list_BBH_or) ,str(self._get_not_BBHs())) )
        else:
            list_BBH_or = [x for x in elist]
            print('In the '+fname.split('/')[-1]+' data we have the following BBH events, total %s (keeping all events independently):' %(len(list_BBH_or)) )

        list_BBH_or=list(np.unique(np.array(list_BBH_or)))
        
        print( np.sort(np.array(list_BBH_or)))
        # Impose cut on SNR
        print('Using only events with SNR>%s (round to 1 decimal digit)' %self.SNR_th)
        list_BBH_0 = [x for x in list_BBH_or if np.round(self.metadata[self.metadata.commonName==x].network_matched_filter_snr.values, 1)>=self.SNR_th  ]
        print('Events after SNR cut:%s'%len(list_BBH_0))
        list_BBH_excluded_0 = [(x, np.round(self.metadata[self.metadata.commonName==x].network_matched_filter_snr.values), 1)[0] for x in list_BBH_or if np.round(self.metadata[self.metadata.commonName==x].network_matched_filter_snr.values, 1)<self.SNR_th  ]
        print('Excluded the following events with SNR<%s: ' %self.SNR_th)
        print(list_BBH_excluded_0)
        #print('%s events remaining.' %len(list_BBH_0))

        # Impose cut on FAR
        print('Using only events with FAR<%s' %self.FAR_th)
        #print(self.metadata.far.values)
        list_BBH = [x for x in list_BBH_0 if self.metadata[self.metadata.commonName==x].far.values<=self.FAR_th  ]
        list_BBH_excluded = [(x, self.metadata[self.metadata.commonName==x].far.values) for x in list_BBH_0 if self.metadata[self.metadata.commonName==x].far.values>self.FAR_th  ]
        print('Excluded the following events with FAR>%s: ' %self.FAR_th)
        print(str(list_BBH_excluded))
        print('%s events remaining.' %len(list_BBH))
        
        
        # Exclude other events if this is required in the config file
        if events_use is not None and events_use['use'] is not None and events_use['not_use'] is not None:
            raise ValueError('You passed options to both use and not_use. Please only provide the list of events that you want to use, or the list of events that you want to exclude. ')
        elif events_use is not None and events_use['use'] is not None:
            # Use only events given in use
            print('Using only BBH events passed in the config file: ')
            print(events_use['use'])
            list_BBH_final = [x for x in list_BBH if x in events_use['use']]
        elif events_use is not None and events_use['not_use'] is not None:
            print('Also excluding BBH events passed in the config file: ')
            print(events_use['not_use'])
            list_BBH_final = [x for x in list_BBH if x not in events_use['not_use']]
        else:
            print('Using all BBH events')
            list_BBH_final=list_BBH
        
        print('\nFinal list of events used (total %s):'%len(list_BBH_final))
        print(str(list_BBH_final))
        print()
        return list_BBH_final
 
    
    def _load_data(self, fname, nObsUse, which_spins='chiEff'):
        print('Loading data...')
    
        
        #events = self._get_events_names(fname)
        if nObsUse is None:
            nObsUse=len(self.events)
            
        
        #print('We have the following events: %s' %str(events))
        m1s, m2s, dLs, ra, dec, iota, spins, weights = [], [], [], [], [], [], [], []
        allNsamples=[]
        for event in self.events[:nObsUse]:
                print('Reading data from %s' %event)
            #with h5py.File(fname, 'r') as phi:
                m1z_, m2z_, dL_, ra_, dec_, iota_, spins_, weights_  = self._load_data_event(fname, event, nSamplesUse=None, which_spins=which_spins)
                print('Number of samples in LVC data: %s' %m1z_.shape[0]) #%(~np.isnan(m1z_)).sum()) #%m1z_.shape[0])
                m1s.append(m1z_)
                m2s.append(m2z_)
                dLs.append(dL_)
                ra.append(ra_)
                dec.append(dec_)
                iota.append(iota_)
                spins.append(spins_)
                weights.append(weights_)
                assert len(m1z_)==len(m2z_)
                assert len(m2z_)==len(dL_)
                #assert len(weights_)==len(dL_)
                if which_spins!="skip":
                    if which_spins!='default':
                        assert len(spins_)==2
                    else:
                        assert len(spins_)==4
                    assert len(spins_[0])==len(dL_)
                else:  assert spins_==[]
                
                nSamples = len(m1z_)
                
                allNsamples.append(nSamples)
            #print('ciao')
        print('We have %s events.'%len(allNsamples))
        max_nsamples = max(allNsamples) 
        
        fin_shape=(nObsUse, max_nsamples)
        
        m1det_samples= np.full(fin_shape, np.NaN)  #np.zeros((len(self.events),max_nsamples))
        m2det_samples=np.full(fin_shape, np.NaN)
        dl_samples= np.full(fin_shape, np.NaN)
        ra_samples= np.full(fin_shape, np.NaN)
        dec_samples= np.full(fin_shape, np.NaN)
        iota_samples= np.full(fin_shape, np.NaN)
        if which_spins!="skip":
            if which_spins=='default':
                spins_samples= [np.full(fin_shape, np.NaN), np.full(fin_shape, np.NaN), np.full(fin_shape, np.NaN), np.full(fin_shape, np.NaN) ]
            else:
                spins_samples= [np.full(fin_shape, np.NaN), np.full(fin_shape, np.NaN) ]
        else: spins_samples=[]
        
        for i in range(nObsUse):
            
            m1det_samples[i, :allNsamples[i]] = m1s[i]
            m2det_samples[i, :allNsamples[i]] = m2s[i]
            dl_samples[i, :allNsamples[i]] = dLs[i]
            ra_samples[i, :allNsamples[i]] = ra[i]
            dec_samples[i, :allNsamples[i]] = dec[i]
            iota_samples[i, :allNsamples[i]] = iota[i]
            if which_spins!="skip":
                for k in range(len(spins_samples)):
                    spins_samples[k][i, :allNsamples[i]] = spins[i][k]
        
        if self.dist_unit==u.Gpc:
            print('Using distances in Gpc')   
            dl_samples*=1e-03
        
        return m1det_samples, m2det_samples, dl_samples, ra_samples, dec_samples, iota_samples, spins_samples, allNsamples, np.squeeze(np.array(weights))
    
    
    def logOrMassPrior(self):
        return np.zeros(self.m1z.shape)

    def logOrDistPrior(self):
        
    
        if self.dLprior==None:
                return np.zeros(self.dL.shape)
        elif self.dLprior=='dLsq':
                # dl^2 prior on dL
                return np.where( ~np.isnan(self.dL), 2*np.log(self.dL), 0)
        else:
                raise ValueError()
  

class O3InjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None, dist_unit=u.Gpc, ifar_th=1., snr_th=0.,  which_spins='skip', which_injections='GWTC-2' ):
        
        
        self.which_injections = which_injections
        self.which_spins=which_spins
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.spins, self.log_weights_sel, self.N_gen, self.Tobs, self.snrs, conditions_arr = self._load_data(fname, nInjUse, which_spins=which_spins )        
        self.logN_gen = np.log(self.N_gen)
        #self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<self.m1z).all()
        
        print('Loaded data shape: %s' %str(self.m1z.shape))
        print('Loaded weights shape: %s' %str(self.log_weights_sel.shape))
        
        
        
        self.ifar_th=ifar_th
        self.snr_th = snr_th
        
        
        
        
        if which_injections=='GWTC-2':
            gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar = conditions_arr
            self.condition = ((gstlal_ifar > ifar_th) | (pycbc_ifar > ifar_th) | (pycbc_bbh_ifar > ifar_th)) & (self.snrs > snr_th)
        elif which_injections=='GWTC-3':
            self.condition = (np.max(conditions_arr, axis=0) > ifar_th) & (self.snrs > snr_th)
        

        
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL, self.spins  ] )  
    
    
    def _load_data(self, fname, nInjUse, which_spins='skip'):
        
        #import astropy.units as u
        from astropy.cosmology import Planck15
        from cosmology.cosmo import Cosmo
        import h5py
        
        print('Reading injections from %s...' %fname)
        
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
                
            
            
        
            if self.which_injections=='GWTC-2':
                p_draw = np.array(f['injections/sampling_pdf'])
                if which_spins=='skip':
                    print('Removing factor of 1/2 for each spin dimension from p_draw...')
                    p_draw *= 4
                
                gstlal_ifar = np.array(f['injections/ifar_gstlal'])
                pycbc_ifar = np.array(f['injections/ifar_pycbc_full'])
                pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
                conditions_arr = (gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar)
                
                snrs = np.sqrt(np.array(f['injections/optimal_snr_l'])**2 + np.array(f['injections/optimal_snr_h'])**2)
                
            elif self.which_injections=='GWTC-3':
                
                p_draw = np.array(f['injections/mass1_source_mass2_source_sampling_pdf'])*np.array(f['injections/redshift_sampling_pdf'])
                if which_spins!='skip':
                    p_draw *= (np.array(f['injections/spin1x_spin1y_spin1z_sampling_pdf'])*np.array(f['injections/spin2x_spin2y_spin2z_sampling_pdf']) )
                
                conditions_arr = np.array( [ np.array(f['injections/ifar_cwb']), np.array(f['injections/ifar_gstlal']), np.array(f['injections/ifar_mbta']), np.array(f['injections/ifar_pycbc_bbh']), np.array(f['injections/ifar_pycbc_hyperbank'])] )
                snrs = np.array(f['injections/optimal_snr_net'])
             
            log_p_draw = np.log(p_draw)
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
            return m1z, m2z, dL , spins, log_p_draw , Ndraw, Tobs, snrs, conditions_arr


class GWTC3InjectionsData(Data):
    
    def __init__(self, fname, nInjUse=None, dist_unit=u.Gpc, ifar_th=1., snr_th=0.,  which_spins='skip' ):
        
        
        #self.which_injections = which_injections
        self.which_spins=which_spins
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.spins, self.log_weights_sel, self.N_gen, self.Tobs, self.snrs, self.ifar, self.runs = self._load_data(fname, nInjUse, which_spins=which_spins )        
        self.logN_gen = np.log(self.N_gen)
        #self.log_weights_sel = np.log(self.weights_sel)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<self.m1z).all()
        
        #print('Loaded data shape: %s' %str(self.m1z.shape))
        #print('Loaded weights shape: %s' %str(self.log_weights_sel.shape))
        
        
        
        self.ifar_th = ifar_th
        self.snr_th = snr_th
        print('Threshold on inverse false alarm rate (for O3a-O3b): %s'%str(ifar_th))
        print('Threshold on SNR (for O1-O2): %s'%str(snr_th))
        
        
        self.condition = np.where(self.runs == 'o3', self.ifar > ifar_th, self.snrs > self.snr_th)
        print('Number of injections that pass far/snr threshold: %s' %self.condition.sum())
        

          
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL, self.spins  ] )  
    
    
    def _load_data(self, fname, nInjUse, which_spins='skip'):
        
        #import astropy.units as u
        from astropy.cosmology import Planck15
        from cosmology.cosmo import Cosmo
        import h5py
        
        print('Reading injections from %s...' %fname)
        
        with h5py.File(fname, 'r') as f:
        
            T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
            Ndraw = f.attrs['total_generated']

            d = f['injections']
    
            ifars = [
                        d[par][:] for par in d.keys()
                        if ('ifar' in par) and ('cwb' not in par)
                        ]
            ifar = np.max(ifars, axis=0)
            snr = d['optimal_snr_net'][:]
            runs = d['name'][:].astype(str)
        
            all_injs = {k: np.array(f['injections'][k]) for k in f['injections'].keys()}

            zs = all_injs['redshift']
            m1s = all_injs['mass1_source']
            m2s = all_injs['mass2_source']

            log_jac_spin = np.zeros(zs.shape)
            if which_spins=='skip':
                spins=[]
                
            elif which_spins=='default':
                s1x = all_injs['spin1x']
                s1y = all_injs['spin1y']
                s1z = all_injs['spin1z']
                
                s2x = all_injs['spin2x']
                s2y = all_injs['spin2y']
                s2z = all_injs['spin2z']
    
    
                chi1sq = s1x**2+s1y**2+s1z**2
                
                chi2sq = s2x**2+s2y**2+s2z**2

                
                log_jac_spin = np.log(chi1sq) + np.log(chi2sq) + 2*np.log(2*np.pi)

                chi1 = np.sqrt(chi1sq)
                chi2 = np.sqrt(chi2sq)
                cost1 = s1z/chi1
                cost2 = s2z/chi2
                spins=[chi1, chi2, cost1, cost2]

            else:
                raise NotImplementedError()
                

            print('Re-weighting p_draw to go to detector frame quantities...')
            cosmo = Cosmo(dist_unit=self.dist_unit)

            dLs =  cosmo.dLGW(zs, 67.9, 0.3065, -1,1,0)
            log_p_draw_jac = 2*np.log1p(zs)+cosmo.log_ddL_dz( zs,  67.9, 0.3065, -1,1,0 )

            m1d = m1s*(1+zs)
            m2d = m2s*(1+zs)

            log_p_draw = np.log(all_injs['sampling_pdf'])

            log_p_draw_det = log_p_draw-log_p_draw_jac+log_jac_spin
            
            

            print('Number of total injections: %s' %Ndraw)
            print('Number of injections that pass first threshold: %s' %log_p_draw_det.shape[0])
            
            
            self.max_z = np.max(zs)
            print('Max redshift of injections: %s' %self.max_z)
            
            return m1d, m2d, dLs , spins, log_p_draw_det , Ndraw, T_obs, snr, ifar, runs
