#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:56:37 2021

@author: Michi
"""
import numpy as np
import time
import pycbc.waveform
import pycbc.filter
import pycbc.psd
from pycbc.types import FrequencySeries

from scipy.signal import savgol_filter
import h5py
from scipy.interpolate import RectBivariateSpline, interp1d
import os
import argparse
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import Globals



windows = { 'H1O3': 10.,
            'L1O3': 10.,
             'L1O2': 2,
            'H1O2': 4,  
          }


filenames = { 'H1O3': 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt',
            'L1O3': 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt',
             'L1O2': '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt',
            'H1O2': '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt',
            }


abbreviations = {v: k for k, v in filenames.items()}


class oSNR(object):
   
    '''
    
    Class to compute optimal SNR with a given PSD, that should be passed when initialising the class,
    by giving the path to the file where it is stored through the argument psd_path.
    Tabulates the results by using pycbc if pre-computed tables are not available.
    Usage:
    
    myosnr = oSNR(psd_path,  approximant='IMRPhenomXHM') 
    
    myosnr.make_interpolator()
    
    myosnr.get_oSNR(m1d, m2d, dl)
    
    psd_path should be the *full path* to the PSD, not just the name of the file
    
    '''
    
    def __init__(self, from_file=False, psd_path=None, psd_name ='aLIGOEarlyHighSensitivityP1200087', psd_base_path=None, approximant='IMRPhenomXAS', force_recompute = False, verbose=False):
        
        self.from_file=from_file
        if from_file:
            print('Using PSD from file %s ' %psd_path)
        
            self.psd_base_path = ('/').join(psd_path.split('/')[:-1])
            self.psd_file_name = psd_path.split('/')[-1]#.split('.')[0]
            self.name = abbreviations[self.psd_file_name]
            self.psd_path = psd_path
            self.path = psd_path+'_optimal_snr_'+approximant+'.h5'
        else:
            self.name = psd_name
            self.psd_path = psd_name
            self.path = os.path.join(psd_base_path,psd_name+'_optimal_snr_'+approximant+'.h5') #psd_path+'_optimal_snr_'+approximant+'.h5'
            #print('Output will be in %s' %self.path)
        
        self.approximant=approximant
        self.psd_computed=False
        self.verbose=verbose
        self.force_recompute=force_recompute
        if self.force_recompute:
            print('!! You choose to re-compute the PSD even if a pre-comuted one was present! This will overwrite already present files, if any. ')
        #self.interpolator = self._interpolateSNR(**kwargs)
        
        # This  will be the max length of hp , is this is computed. 
        # We track it in case we want to make a plot of the
        # extrapolated PSD
        self.maxlen=0
    
    def _computeSNR(self, m1, m2, npoints=200, mmin=1., mmax=1000., dL_pivot_Gpc=1., deltaf=1./40. , store=True):
        
        if not self.psd_computed and self.from_file:
            psdvals = np.loadtxt(self.psd_path)
            self.flow = min(psdvals[:,0])
        elif not self.from_file:
            self.flow = 10.
        
        hp, hc = pycbc.waveform.get_fd_waveform( approximant=self.approximant,
                                                    mass1=m1,
                                                    mass2=m2,
                                                    delta_f = deltaf,
                                                    f_lower=self.flow,
                                                    distance=1e03*dL_pivot_Gpc ) # get_fd_waveform wants distance in Mpc
        if len(hp)>self.maxlen:
            self.maxlen=len(hp)
            
        if not self.psd_computed:# or self.force_recompute:
                    
                    print('Computing PSD with spectral density from %s ...' %self.psd_path)
                    print('Wf length: %s ' %len(hp))
                    self.psd_computed=True
                    #psdvals = np.loadtxt(self.psd_path)
                    #print(psdvals)
                    #print('Min frequency in PSD: %s' %min(psdvals[:,0]))
                    #print('Max frequency in PSD: %s' %max(psdvals[:,0]))
                    #assert self.flow==min(psdvals[:,0])
        psd = self._get_psd(len(hp), deltaf, self.flow, is_asd_file=True, plot=False) #pycbc.psd.from_txt(self.psd_path, len(hp), deltaf, self.flow, is_asd_file=True)
                    
                    
        snr = pycbc.filter.sigma(hp, psd=psd, low_frequency_cutoff=self.flow)
    
        return snr
     
    
    def _get_psd(self, length, delta_f, low_freq_cutoff, is_asd_file=True, plot=False):
        if  self.from_file:
            return self._get_psd_from_file( length, delta_f, low_freq_cutoff, is_asd_file=True, plot=False)
        else:
            return pycbc.psd.analytical.from_string( self.name,  length, delta_f, low_freq_cutoff)
        
    
    def _get_psd_from_file(self, length, delta_f, low_freq_cutoff, is_asd_file=True, plot=False):
        '''
        This function adapts pycbc.psd.from_txt to extrapolate abose the max frequency of the 
        tabulated psd: it interpolates the values smoothly, and pads the original
        psd by extrapolating this smooth function.
        '''
        if not self.psd_computed:
            print('Readind PSD from %s...' %self.psd_path)
        file_data = np.loadtxt(self.psd_path)
        if (file_data < 0).any() or np.logical_not(np.isfinite(file_data)).any():
            raise ValueError('Invalid data in ' + self.psd_path)
        freq_data = file_data[:, 0]
        noise_data = file_data[:, 1]
        if is_asd_file:
            noise_data = noise_data ** 2
        
        kmin = int(low_freq_cutoff / delta_f)
        flow = kmin * delta_f
        data_start = (0 if freq_data[0]==low_freq_cutoff else np.searchsorted(freq_data, flow) - 1)

        # If the cutoff is exactly in the file, start there
        if freq_data[data_start+1] == low_freq_cutoff:
            if not self.psd_computed:
                print('Starting at +1')
            data_start += 1

        freq_data = freq_data[data_start:]
        noise_data = noise_data[data_start:]
        
        flog = np.log(freq_data)
        slog = np.log(noise_data)
        

        
        psd_interp = interp1d(flog, slog)
        kmin = int(low_freq_cutoff / delta_f)
        vals = np.log(np.arange(kmin, length) * delta_f)
        psd = np.zeros(length, dtype=np.float64)
        pad=True
        try:
            max_exact = np.argwhere(vals>=flog.max())[0][0]
            
            filtered = savgol_filter(slog, int(flog.shape[0]/windows[self.name])+1, 3, deriv=0)
            filter_interp = interp1d(flog, filtered, bounds_error=False, fill_value='extrapolate')
            
            switch=10
            psd[kmin:kmin+max_exact-switch] =   np.exp(psd_interp(vals[:max_exact-switch]))
            psd[kmin+max_exact-switch:] = np.exp(filter_interp(vals[max_exact-switch:]))
        except IndexError:
            pad=False
            psd[kmin:] =  np.exp(psd_interp(vals))
        
        
        if plot:
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = 'serif'
            plt.rcParams["mathtext.fontset"] = "cm"
            fig, ax = plt.subplots(1,2, figsize=(15, 4))

            if pad:
                ax[0].plot(vals[:max_exact-1], psd_interp(vals[:max_exact-1]), alpha=0.2,  color='green',label='Interpolation of the original')
                ax[0].plot(vals[8000:], filter_interp(vals[8000:]), alpha=1, label='Extrapolation', color='orange')
                ax[0].axvline(vals.min(), color='k', ls='--')
                ax[0].axvline(vals.max(), color='k', ls='--')
                ax[0].legend()



            ax[1].plot(vals, np.log(psd[kmin:]), alpha=1., label='Full')
            ax[1].legend()
            #plax[1].show()

            fig.suptitle(os.path.join(self.psd_base_path,self.name+', '+self.approximant) )
            fig.savefig( os.path.join(self.psd_base_path, self.name+'_interpolation.pdf') )


        
        
        return FrequencySeries(psd, delta_f=delta_f)


        
        
    
    def _computeSNRtable(self, npoints=200, mmin=1., mmax=1000., dL_pivot_Gpc=1., deltaf=1./40 ,  store=True):
    
        npoints=int(npoints)    
    
        self.dL_pivot_Gpc=dL_pivot_Gpc    
        self.mmin = mmin
        self.mmax = mmax
        # Grid of detector frame masses
        #ms = np.exp(np.linspace(np.log(mmin), np.log(mmax), npoints))
        ms = np.geomspace(mmin, mmax, npoints)
        print('Computing optimal SNR for detector-frame masses in (%s, %s ) solar masses at distance of %s Gpc...' %(mmin, mmax, dL_pivot_Gpc))
    
        osnrs = np.zeros((npoints, npoints))
        
        in_time=time.time()
    
        for i, m1 in enumerate(ms):
            for j in range(i+1):
                
                m2 = ms[j]
      
                snr_ = self._computeSNR( m1, m2, npoints=npoints, mmin=mmin, mmax=mmax, dL_pivot_Gpc=dL_pivot_Gpc, deltaf=deltaf , store=store)
            
                osnrs[i,j] = snr_
                osnrs[j,i] = snr_
            
                if i+j % 50 == 0 :
                    print('#', end='', flush=True)
    
        print('\nDone in %.2fs ' %(time.time() - in_time))
       
        if store:
            print('Saving result...')     
            with h5py.File(self.path, 'w') as out:
                out.create_dataset('ms', data=ms, compression='gzip', shuffle=True)
                out.create_dataset('SNR', data=osnrs, compression='gzip', shuffle=True)
                out.attrs['dL'] = dL_pivot_Gpc #'%s Gpc'%dL_pivot_Gpc
                out.attrs['npoints'] = npoints
                out.attrs['approximant'] = self.approximant
                out.attrs['mmin'] = mmin
                out.attrs['mmax'] = mmax
                out.attrs['deltaf'] = deltaf
                out.attrs['flow'] = self.flow
                
                
        return ms, osnrs


    def make_interpolator(self,  **kwargs) :
        
        if os.path.exists(self.path) and not self.force_recompute:
            if self.verbose:
                print('Pre-computed optimal SNR grid is present for this PSD. Loading...')
            with h5py.File(self.path, 'r') as inp:
                ms = np.array(inp['ms'])
                SNRgrid = np.array(inp['SNR'])
                self.dL_pivot_Gpc =  inp.attrs['dL']
                self.flow =  inp.attrs['flow']
                self.mmin = inp.attrs['mmin']
                self.mmax = inp.attrs['mmax']
                if self.verbose:
                    print('Attributes of pre-computed SNRs: ')
                    print([(k, inp.attrs[k]) for k in inp.attrs.keys() ])
                    #print(inp.attrs)
                
        else:
            print('Tabulating SNRs...')
            ms, SNRgrid = self._computeSNRtable(**kwargs)
        
        self.interpolator =  RectBivariateSpline(ms, ms, SNRgrid)
    
    
    def get_oSNR(self, m1det, m2det, dL):  
        '''
        m1det, m2det : masses in detector frame in units of solar mass
        dL : luminosity distance in Gpc
        '''
        
        if np.any(m2det>m1det):
            raise ValueError('Optimal SNR called with m2>m1')
        if np.any(m2det<self.mmin):
            raise ValueError('get_oSNR called with value of m2 below the interpolation range')
        if np.any(m1det>self.mmax):
            raise ValueError('get_oSNR called with value of m1 above the interpolation range')
        
        return self.interpolator.ev(m1det, m2det)*self.dL_pivot_Gpc/dL


    
ifos = {'L1': 'Livingston', 'H1': 'Hanford'}
        


def tabulate_SNR_from_files(approximant='IMRPhenomXHM', npoints=200, mmin=1., mmax=1000., dL_pivot_Gpc=1., deltaf=1./40 , store=True, force_recompute=False):
      
    

    for run in ('O3', 'O2'):
        for detectorname in ["L1", "H1"]:
            
            print('\n --------- %s obs run, %s interferometer' %(run, ifos[detectorname]))
            
            psd_path = os.path.join(Globals.detectorPath, filenames[detectorname+run])            
            myosnr = oSNR(from_file=True, psd_path=psd_path, approximant=approximant, force_recompute=force_recompute)
            
            myosnr.make_interpolator(npoints=npoints, mmin=mmin, mmax=mmax, dL_pivot_Gpc=dL_pivot_Gpc, deltaf=deltaf,  store=store)
            
            #make plot
            length = myosnr.maxlen
            _ = myosnr._get_psd( length, deltaf, myosnr.flow, is_asd_file=True, plot=True)
 
    
 
def tabulate_SNR_from_analytic(psd_name, approximant='IMRPhenomXHM', npoints=200, mmin=1., mmax=1000., dL_pivot_Gpc=1., deltaf=1./40 , store=True, force_recompute=False):
      
    

            #print('\n --------- %s obs run, %s interferometer' %(run, ifos[detectorname]))
            
            #psd_path = os.path.join(detectorPath, filenames[detectorname+run])            
            myosnr = oSNR(from_file=False,  psd_name =psd_name, psd_base_path=Globals.detectorPath, approximant=approximant, force_recompute=force_recompute)
            
            myosnr.make_interpolator(npoints=npoints, mmin=mmin, mmax=mmax, dL_pivot_Gpc=dL_pivot_Gpc, deltaf=deltaf,  store=store)
            
            ##make plot
            #length = myosnr.maxlen
            #_ = myosnr._get_psd( length, deltaf, myosnr.flow, is_asd_file=True, plot=True)
            
            
 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_file", default=0., type=float, required=True)
    parser.add_argument("--psd_name", default='aLIGOEarlyHighSensitivityP1200087', type=str, required=False)
    parser.add_argument("--approximant", default='IMRPhenomXAS', type=str, required=False)
    parser.add_argument("--npoints", default=200, type=float, required=False) 
    parser.add_argument("--mmin", default=1., type=float, required=False)
    parser.add_argument("--mmax", default=1000., type=float, required=False)
    parser.add_argument("--dL_pivot_Gpc", default=1., type=float, required=False)
    parser.add_argument("--deltaf", default=1./40, type=float, required=False)
    parser.add_argument("--force_recompute", default=0., type=float, required=False)
    FLAGS = parser.parse_args()
    
    print('Arguments: ')
    for key, value in vars(FLAGS).items():
        print(key, value )
    force_recompute=False
    if FLAGS.force_recompute==1.:
        force_recompute=True
    
    if FLAGS.from_file==1:
        print('Computing from file')
        tabulate_SNR_from_files(approximant = FLAGS.approximant, npoints=FLAGS.npoints,mmin=FLAGS.mmin, mmax=FLAGS.mmax, dL_pivot_Gpc=FLAGS.dL_pivot_Gpc, deltaf=FLAGS.deltaf ,  store=True, force_recompute=force_recompute)
    else:
        if FLAGS.psd_name is None:
            raise ValueError()
        tabulate_SNR_from_analytic(FLAGS.psd_name, approximant = FLAGS.approximant, npoints=FLAGS.npoints,mmin=FLAGS.mmin, mmax=FLAGS.mmax, dL_pivot_Gpc=FLAGS.dL_pivot_Gpc, deltaf=FLAGS.deltaf ,  store=True, force_recompute=force_recompute)
    