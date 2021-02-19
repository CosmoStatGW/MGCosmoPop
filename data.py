import numpy as np
import h5py
import os
from config import nObsUse, nSamplesUse, nInjUse
import Globals
import astropy.units as u
from astropy.cosmology import Planck15
import cosmo


def originalMassPrior(m1z, m2z):
    return np.ones(m1z.shape)

def originalDistPrior(dL):
    return np.ones(dL.shape)


def logOriginalMassPrior(m1z, m2z):
    return np.zeros(m1z.shape)

def logOriginalDistPrior(dL):
    return np.zeros(dL.shape)



###############################################
#  LOADING DATA
###############################################

def load_data(dataset_name):
    
    '''
    Returns a tuple (theta, Nsamples)
    -   theta is a np array of shape ( 3 x N_observations x max_n_samples )
        where  max_n_samples is the largest number of samples in all the observations;
        events with less samples should be filled with zeros (or -1)
    - Nsamples is an array of length N_observations, containing the number of samples for each observation
    
    '''
    
    if dataset_name=='mock':
        return load_mock_data()
    elif dataset_name=='O3':
        return load_O3_data()
    


def load_mock_data():
    with h5py.File(os.path.join(Globals.dataPath,'mock','observations.h5'), 'r') as phi: #observations.h5 has to be in the same folder as this code
        
        if nObsUse is None and nSamplesUse is None:
            m1det_samples = np.array(phi['posteriors']['m1det'])
            m2det_samples = np.array(phi['posteriors']['m2det'])
            dl_samples = np.array(phi['posteriors']['dl']) # dLm distance is given in Gpc in the .h5
            #theta_samples = np.array(phi['posteriors']['theta']) [:10,:100] # theta
            # Farr et al.'s simulations: our "observations"
        elif nObsUse is not None and nSamplesUse is None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :] 
            
        elif nObsUse is  None and nSamplesUse is not None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:, :nSamplesUse]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:, :nSamplesUse] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:, :nSamplesUse]
            
        elif nObsUse is not None and nSamplesUse is not None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :nSamplesUse]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :nSamplesUse] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :nSamplesUse] 
    
    if Globals.which_unit==u.Mpc:
        print('Using distances in Mpc')
        dl_samples*=1e03
    theta =   np.array([m1det_samples, m2det_samples, dl_samples])
    return theta, np.count_nonzero(m1det_samples, axis=-1)


############################
# Injections
############################

def load_injections_data(dataset_name_injections):
    if dataset_name_injections=='mock':
        return load_injections_data_mock()
    elif dataset_name_injections=='LVC':
        return load_injections_data_LVC()
    
  
    
def load_injections_data_LVC():
    with h5py.File(os.path.join(Globals.dataPath,'LVC_injections','o3a_bbhpop_inj_info.hdf'), 'r') as f:
        
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
        if Globals.which_unit==u.Gpc:
            dL*=1e-03
        
        print('Re-weighting p_draw to go to detector frame quantities...')
        p_draw/=(1+z)**2
        p_draw/=cosmo.ddL_dz(z, Planck15.H0.value, Planck15.Om0, -1., 1., 0) #z, H0, Om, w0, Xi0, n

        
    return np.array([m1z, m2z, dL]), p_draw , Ndraw, Tobs, (gstlal_ifar, pycbc_ifar, pycbc_bbh_ifar)

    

def load_injections_data_mock():   
    
    Tobs=2.5
    with h5py.File(os.path.join(Globals.dataPath,'mock','selected.h5'), 'r') as f:
        
        if nInjUse is not None:
            m1_sel = np.array(f['m1det'])[:nInjUse]
            m2_sel = np.array(f['m2det'])[:nInjUse]
            dl_sel = np.array(f['dl'])[:nInjUse]
            weights_sel = np.array(f['wt'])[:nInjUse]
        else:
            m1_sel = np.array(f['m1det'])
            m2_sel = np.array(f['m2det'])
            dl_sel = np.array(f['dl'])
            weights_sel = np.array(f['wt'])
        
        N_gen = f.attrs['N_gen']
    if Globals.which_unit==u.Mpc:
        dl_sel*=1e03
        
    return np.array([m1_sel, m2_sel, dl_sel]), weights_sel , N_gen, Tobs, np.NaN




def load_O3_data():
    return None
