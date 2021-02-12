import numpy as np
import h5py
import os
from config import nObsUse, nSamplesUse, nInjUse
import globals





def originalMassPrior(m1z, m2z):
    return np.ones(m1z.shape)


def originalDistPrior(dL):
    return np.ones(dL.shape)



###############################################
#  LOADING DATA
###############################################

def load_data(dataset_name):
    if dataset_name=='mock':
        return load_mock_data()
    elif dataset_name=='O3':
        return load_O3_data()
    


def load_mock_data():
    with h5py.File(os.path.join(globals.dataPath,'observations.h5'), 'r') as phi: #observations.h5 has to be in the same folder as this code
        
        if nObsUse is None and nSamplesUse is None:
            m1det_samples = np.array(phi['posteriors']['m1det'])
            m2det_samples = np.array(phi['posteriors']['m2det'])
            dl_samples = np.array(phi['posteriors']['dl'])*10**3 # dLm distance is given in Gpc in the .h5
            #theta_samples = np.array(phi['posteriors']['theta']) [:10,:100] # theta
            # Farr et al.'s simulations: our "observations"
        elif nObsUse is not None and nSamplesUse is None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :]*10**3 
            
        elif nObsUse is  None and nSamplesUse is not None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:, :nSamplesUse]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:, :nSamplesUse] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:, :nSamplesUse]*10**3 
            
        elif nObsUse is not None and nSamplesUse is not None:
            
            m1det_samples = np.array(phi['posteriors']['m1det'])[:nObsUse, :nSamplesUse]# m1
            m2det_samples = np.array(phi['posteriors']['m2det'])[:nObsUse, :nSamplesUse] # m2
            dl_samples = np.array(phi['posteriors']['dl'])[:nObsUse, :nSamplesUse]*10**3 
            
        
    return np.array([m1det_samples, m2det_samples, dl_samples])



def load_injections_data(dataset_name_injections):
    if dataset_name_injections=='mock':
        return load_injections_data_mock()
    elif dataset_name_injections=='LVC':
        return load_injections_data_LVC()
    
  
    
def load_injections_data_LVC():
    return None
    

def load_injections_data_mock():   
    
    
    with h5py.File(os.path.join(globals.dataPath,'selected.h5'), 'r') as f:
        
        if nInjUse is not None:
            m1_sel = np.array(f['m1det'])[:nInjUse]
            m2_sel = np.array(f['m2det'])[:nInjUse]
            dl_sel = np.array(f['dl'])[:nInjUse]*10**3
            weights_sel = np.array(f['wt'])[:nInjUse]
        else:
            m1_sel = np.array(f['m1det'])
            m2_sel = np.array(f['m2det'])
            dl_sel = np.array(f['dl'])*10**3
            weights_sel = np.array(f['wt'])
        
        N_gen = f.attrs['N_gen']
        
    return np.array([m1_sel, m2_sel, dl_sel]), weights_sel , N_gen




def load_O3_data():
    return None
