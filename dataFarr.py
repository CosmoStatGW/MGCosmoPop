import numpy as np
import h5py
import os


###############################################
#  GLOBAL VARIABLES
###############################################

TrueValues = {'H0':67.74, 'Xi0':1.0, 'n':1.91, 'lambdaRedshift':3.0,  'alpha':0.75, 'beta':0.0, 'ml':5.0, 'sl':0.1, 'mh':45.0, 'sh':0.1}

n= 1.91 # we will use that n. This isn't really nice :-/
alpha = 0.75 # param m1^-alpha
beta1 = 0.0 # param m2^beta
ml = 5.0 # [Msun] minimum mass
mh = 45.0 # [Msun] maximum mass
sl = 0.1 # standard deviation on the lightest mass
sh = 0.1 # standard deviation on the heavier mass 
R0 = 64.4
gamma1 = 3.0
Tobs=5./2
H0=67.74
Xi0 = 1.0

dirName  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
dataPath=os.path.join(dirName, 'data')

###############################################
#  LOADING DATA
###############################################

print('loading data')

with h5py.File(os.path.join(dataPath,'observations.h5'), 'r') as phi: #observations.h5 has to be in the same folder as this code
   
        m1det_samples = np.array(phi['posteriors']['m1det']) # m1
        m2det_samples = np.array(phi['posteriors']['m2det']) # m2
        dl_samples = np.array(phi['posteriors']['dl'])*10**3 # dLm distance is given in Gpc in the .h5
        theta_samples = np.array(phi['posteriors']['theta'])  # theta
        # Farr et al.'s simulations: our "observations"

with h5py.File(os.path.join(dataPath,'selected.h5'), 'r') as f:
        m1_sel = np.array(f['m1det'])
        m2_sel = np.array(f['m2det'])
        dl_sel = np.array(f['dl'])*10**3
        weights_sel = np.array(f['wt'])
        N_gen = f.attrs['N_gen']

print('done data')

n = m1det_samples.shape[0]
print('We have %s observations' %n)


theta = np.array([m1det_samples, m2det_samples, dl_samples])
theta_sel = np.array([m1_sel, m2_sel, dl_sel])
Lambda_ntest = np.array([n, gamma1, alpha, beta1, ml, sl, sh])
    
Delta=[140-20, 10-0, 150-20]
beginDelta = [20, 0, 20]

labels_param=[r"H_0", r"$\Xi_0$", r"m_h"]
trueValues = [67.74, 1, 45]

priorLimits  = ((20, 140), (0, 10), (20, 150)) # (-10, 250), (-10, 250), (-10, 250), (-10, 250), (-10, 250), (-10, 250))
##### New prior limits


