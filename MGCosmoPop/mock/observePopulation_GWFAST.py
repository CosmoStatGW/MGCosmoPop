#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:12:51 2022

@author: Michi
"""


import numpy as np
#import random
import h5py
import os 
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#from scipy.interpolate import interp1d
#from scipy.integrate import cumtrapz

#from .SNRtools import oSNR, NetworkSNR, Detector
#import scipy.stats as ss

#import Globals


from scipy.special import erfc, erfcinv

def sample_trunc_gaussian(mu = 1, sigma = 1, lower = 0, size = 1):

    sqrt2 = np.sqrt(2)
    Phialpha = 0.5*erfc(-(lower-mu)/(sqrt2*sigma))
    
    if np.isscalar(mu):
        arg = Phialpha + np.random.uniform(size=size)*(1-Phialpha)
        return np.squeeze(mu - sigma*sqrt2*erfcinv(2*arg))
    else:
        Phialpha = Phialpha[:,np.newaxis]
        arg = Phialpha + np.random.uniform(size=(mu.size, size))*(1-Phialpha)
        
        return np.squeeze(mu[:,np.newaxis] - sigma[:,np.newaxis]*sqrt2*erfcinv(2*arg))




class Observations(object):
    
    
    def __init__(self,
                 populations, 
                 detNet,
                 zmax, out_dir,
                 snr_th_dets = [],
                 add_noise=False,
                 seed=None
#                 **kwargs):
    ):        


        self.add_noise = add_noise
        self.seed=seed
        
        self.out_dir=out_dir
        self.detNet = detNet 
        
        self.allPops=populations
        
        self.zmax = zmax 
        
        #self.snr_th=snr_th
        self.snr_th_dets=snr_th_dets
    
        
        self.lambdaBase = self.allPops.get_base_values(self.allPops.params)
        self.LambdaCosmoBase, self.LambdaAllPopBase = self.allPops._split_params(self.lambdaBase)
        self.lambdaBBHrate, self.lambdaBBHmass, self.lambdaBBHspin = self.allPops._pops[0]._split_lambdas(self.LambdaAllPopBase)
        
        self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase = self.allPops.cosmo._get_values(self.LambdaCosmoBase, [ 'H0', 'Om', 'w0', 'Xi0', 'n'])
    
        self._find_Nperyear_expected()
        
    
    def _find_Nperyear_expected(self):  
         self.Nperyear_expected = self.allPops.Nperyear_expected(self.lambdaBase, zmax=self.zmax, verbose=True)
         print('Expected number per year between redshift 0 and %s: %s'%(self.zmax, self.Nperyear_expected) )

       
    
    def _generate_mergers(self, N, verbose=True):
        
        #np.random.seed(self.seed)
        #np.random.seed(np.random.randint(2**32 - 1, size=1))
        
        if N==1:
            verbose=False
        if verbose:
            print('Parameters used to sample events: %s' %str(self.lambdaBase))
        
        massSamples, zs, spinSamples = self.allPops.sample(N, self.zmax, self.lambdaBase)
        
        m1s, m2s = np.squeeze(massSamples)[:, 0], np.squeeze(massSamples)[:, 1]

        if self.allPops._pops[0].spinDist.__class__.__name__ =='UniformSpinDistChiz':
        
            spin1z, spin2z = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1]
            spins='flat'
        elif self.allPops._pops[0].spinDist.__class__.__name__ =='DefaultSpinModel':
            spins='default'
            chi1, chi2, cost1, cost2 = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1], np.squeeze(spinSamples)[:, 2], np.squeeze(spinSamples)[:, 3]
            spin1z = chi1*cost1
            spin2z = chi2*cost2
        elif self.allPops._pops[0].spinDist.__class__.__name__ =='UniformOnSphereSpin':
            spins='uniform_on_sphere'
            s1x, s1y, s1z, s2x, s2y, s2z = np.squeeze(spinSamples)[:, 0], np.squeeze(spinSamples)[:, 1], np.squeeze(spinSamples)[:, 2], np.squeeze(spinSamples)[:, 3], np.squeeze(spinSamples)[:, 4], np.squeeze(spinSamples)[:, 5]
            spin1z = s1z
            spin2z = s2z
        else:
            raise ValueError()
        
        costheta = np.random.uniform(-1, 1, N)
        cosiota = np.random.uniform(-1, 1, N)
            
        theta = np.arccos(costheta)
        iota = np.arccos(cosiota)
            
        phi = np.random.uniform(0, np.pi*2, N)
        phiCoal = np.random.uniform(0, np.pi*2, N)
        psi = np.random.uniform(0, np.pi, N)
            
        tcoal = np.random.uniform(size=N)
        
        Lambda1 = np.random.uniform(low=0, high=2000, size=N)
        Lambda2 = np.random.uniform(low=0, high=2000, size=N)
        
        ## Get quantities in detector frame
        
        dL = self.allPops.cosmo.dLGW(zs, self.H0base, self.Om0Base, self.w0Base, self.Xi0Base, self.nBase)
        mass1_det, mass2_det, = m1s*(1+zs), m2s*(1+zs)
        
        Mc_det = (mass1_det*mass2_det)**(3/5)/((mass1_det+mass2_det)**(1/5))
        eta = (mass1_det*mass2_det)/((mass1_det+mass2_det)**(2))
        
        
        events = { 'Mc':Mc_det,
          'eta':eta,
          'm1_source':m1s,
          'm2_source':m2s,
          'm1_det':mass1_det,
          'm2_det':mass2_det,
          'z':zs,
          'dL':dL,
          'theta':theta, 
          'phi':phi,
          
          'ra':phi,
          'dec':np.pi/2-theta,
          
          'iota':iota,
          'psi':psi, 
                  
          #'tGPS':gpsts,
          'Phicoal':phiCoal ,
                  
          'Lambda1': Lambda1,
          'Lambda2':  Lambda2,
                
                     #'chi1': chi1,
                     #'chi2':chi2,
                    #'cost1': cost1,
                    #'cost2':cost2,
            'chi1z': spin1z,
            'chi2z':spin2z,
          'tcoal':tcoal    
}

        if spins=='default':
            events['chi1'] = chi1
            events['chi2'] = chi2
            events['cost1'] = cost1
            events['cost2'] = cost2
        elif spins=='uniform_on_sphere'
            events['s1x'] = s1x
            events['s2x'] = s2x
            events['s1y'] = s1y
            events['s2y'] = s2y
            events['s1z'] = s1z
            events['s2z'] = s2z
        
        return events, spins #np.squeeze(m1s), np.squeeze(m2s), np.squeeze(zs), costhetas,  phis, cosiotas, ts_det
    
    
    
    def _generate_injection_chunk(self,  N, verbose=False ):
        
        # necessary !! otherwise get repeated events
        np.random.seed()
        
        #if verbose:
        #    print('Generating events in source frame and redshift...')
        events_injected, spins = self._generate_mergers(N, verbose=False)
        if verbose:
            print('Generated %s events .' %str(events_injected['dL'].shape))
      
        
        ## Get SNR
        if verbose:
            print('Computing snr...')
        try:
            allSNRsOpt = self.detNet.SNR_netFast(events_injected, return_all=True)
        except:
            allSNRsOpt = self.detNet.SNR(events_injected, return_all=True)
        
        SNRopt =  allSNRsOpt['net'] # network snr
        if verbose:
            print('Done.')
        
        
        if self.add_noise:
            if verbose:
                print('Adding noise...')
            # Get observed SNR (ie gaussian random variable w unit variance for each det). 
            SNRobsSq=np.zeros(allSNRsOpt['net'].shape)
            allSNRs = {}
            for d in allSNRsOpt.keys():
                if d!='net':
                    if verbose:
                        print('Adding %s' %d)
                    SNRobs_ = allSNRsOpt[d]+np.random.normal(loc=0, scale=1, size=len(allSNRsOpt[d]))
                    allSNRs[d] = SNRobs_
                    SNRobsSq += allSNRs[d]**2
            SNR = np.sqrt(SNRobsSq)
            allSNRs['net'] = SNR
        else:
            # Use optimal SNR. 
            if verbose:
                print('Using optimal SNR...')
            SNR = SNRopt
            allSNRs = allSNRsOpt
     
        
        # Select
        keep = np.full(SNR.shape, True)
       
        for d in allSNRs.keys():
            if verbose:
                print('Searching %s in %s' %(d , str(list(self.snr_th_dets.keys()))))
            found=False
            for d1 in self.snr_th_dets.keys():
                if d1 in d and not found:
                    th = self.snr_th_dets[d1]
                    if verbose:
                        print('th for %s: %s' %(d, th))
                    found=True
                        
            keep &= allSNRs[d]>th
            if verbose:
                print('Kept after %s snr cut: %s' %(d, keep.sum()))
        ndet = keep.sum()
        
        events_detected =  {k: events_injected[k][keep] for k in events_injected.keys()}
        
        # if not including spins in the population model, and using flat distribution for injections, 
        # here do not pass the spins. this way the log prob will be already correct
        # otherwise, pass spins here and include them in the pop model.
        
        if ndet>0:

            
            log_p_draw_nospin = self.allPops.log_dN_dm1zdm2zddL( events_detected['m1_source'], events_detected['m2_source'], events_detected['z'], 
                                                     #[events_detected['chi1z'], events_detected['chi2z']], 
                                                      [],
                                                     self.lambdaBase, 1., dL=events_detected['dL'])-np.log(self.Nperyear_expected)

            if spins=='default':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['chi1'], events_detected['chi2'], events_detected['cost1'], events_detected['cost2']], self.lambdaBBHspin)
            elif spins=='flat':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['chi1z'], events_detected['chi2z']], self.lambdaBBHspin)
            elif spins=='uniform_on_sphere':
                log_p_draw_spin = self.allPops._pops[0].spinDist.logpdf([events_detected['s1x'], events_detected['s1y'], events_detected['s1z'], events_detected['s2x'], events_detected['s2y'], events_detected['s2z']], self.lambdaBBHspin)
            else:
                log_p_draw_spin =  np.ones(events_detected['m1_source'].shape)

            log_p_draw =  log_p_draw_spin + log_p_draw_nospin
    
        
        else:
            log_p_draw=np.full( events_detected['m1_source'].shape, np.inf)
            log_p_draw_nospin=np.full( events_detected['m1_source'].shape, np.inf)
        
        events_detected['log_p_draw'] = log_p_draw
        events_detected['log_p_draw_nospin'] = log_p_draw_nospin
        
        for k in allSNRs.keys():
            events_detected['snr_%s'%k] = allSNRs[k][keep]
        
        events_detected['snr'] = SNR[keep]
        
        
        return events_detected
        
        
    def generate_injections(self, N_goal, chunk_size=int(1e05),):
        N_gen =0
        Nsucc=0
        m1s_det = []
        m2s_det = []
        dls_det = []
        logwts_det = []
        logwts_nosp_det = []
        enough=False
        #first_it=True
        success=False
        ntry=2
        maxit=3
        it=0
        all_evs_det = {}
        while not success:
            try:
                tmp_ = self._generate_injection_chunk( ntry, verbose=True)
                success=True
            except Exception as e:
                print(e)
                ntry+=2
                it+=1
                if it>=maxit:
                    #success=True
                    ntry=chunk_size
                #tmp_ = self._generate_injection_chunk( ntry, verbose=False)
        
        rhos_det_all = {k:[] for k in tmp_.keys() if 'snr' in k }
        print('\nStart...\n')
        is_first=True
        while not enough:

            #m1d, m2d, dls, logwts, rho = self._generate_injection_chunk( chunk_size, )
            events_chunk = self._generate_injection_chunk( chunk_size, verbose=True )
            
            if is_first:
                all_evs_det = events_chunk
                is_first=False
            else:
                for k in all_evs_det.keys():
                    all_evs_det[k] = np.append( all_evs_det[k], events_chunk[k])
                    if k=='dL':
                        print('Events detected cat len: %s' %len(all_evs_det[k]))
            
            m1d, m2d, dls, logwts, logwts_nosp = np.squeeze(events_chunk['m1_det']), np.squeeze(events_chunk['m2_det']), np.squeeze(events_chunk['dL']), np.squeeze(events_chunk['log_p_draw']), np.squeeze(events_chunk['log_p_draw_nospin'])
            
            
            try:
                Nsucc += len(m1d)
            except:
                pass
            
            if np.ndim(m1d)==0 or np.isscalar(m1d):
                m1d, m2d, dls, logwts, logwts_nosp = np.array([m1d]), np.array([m2d]), np.array([dls]), np.array([logwts]), np.array([logwts_nosp])
            
            N_gen += chunk_size
            
            #if first_it:
            #   print('Fisrt iteration !')
            #    for k in events_chunk.keys():
            #        if 'snr' in k:
            #            rhos_det_all[k] = events_chunk[k]
            #    first_it=False
            #else:
            for k in rhos_det_all.keys():
                    rhos_det_all[k].append(events_chunk[k])
                #first_it=False

            
            m1s_det.append(m1d)
            m2s_det.append(m2d)
            dls_det.append(dls)
            #rhos_det.append(rho)
            logwts_det.append(logwts)
            logwts_nosp_det.append(logwts_nosp)

            
            print('\nTotal kept: so far= %s\n\n' %Nsucc)
        
            if Nsucc > N_goal:
                enough=True
            
        m1s_det = np.concatenate(m1s_det)
        m2s_det = np.concatenate(m2s_det)
        dls_det = np.concatenate(dls_det)
        logwts_det = np.concatenate(logwts_det)
        logwts_nosp_det = np.concatenate(logwts_nosp_det)
        #rhos_det =  np.concatenate(rhos_det)
        for k in rhos_det_all.keys():
            rhos_det_all[k] = np.concatenate(rhos_det_all[k])
        print('Total generated: %s' %N_gen)
        print('Total kept: %s' %Nsucc)
        
        
        
        
        print('Saving catalog of detected events...')
        with h5py.File(os.path.join(self.out_dir,'catalog_detected.h5'), 'w') as out:
                        
            def cd(n, d):
                d = np.array(d)
                out.create_dataset(n, data=d, compression='gzip', shuffle=True)
            
            for key in all_evs_det.keys():
                cd(key, all_evs_det[key])
        
        
        with h5py.File(os.path.join(self.out_dir,'selected.h5'), 'w') as f:
            f.attrs['N_gen'] = N_gen
            for k in self.snr_th_dets.keys():
                f.attrs['snr_th_%s'%k] = self.snr_th_dets[k]
                if k=='net':
                    f.attrs['snr_th'] = self.snr_th_dets[k]
            f.create_dataset('m1det', data=m1s_det, compression='gzip', shuffle=True)
            f.create_dataset('m2det', data=m2s_det, compression='gzip', shuffle=True)
            f.create_dataset('dl', data=dls_det, compression='gzip', shuffle=True)
            f.create_dataset('logwt', data=logwts_det, compression='gzip', shuffle=True)
            f.create_dataset('logwt_nosp', data=logwts_nosp_det, compression='gzip', shuffle=True)
            #f.create_dataset('snr', data=rhos_det, compression='gzip', shuffle=True)
            for k in rhos_det_all.keys():
                print('Saving snr with key %s' %k)
                f.create_dataset(k, data=rhos_det_all[k], compression='gzip', shuffle=True)
    
 
