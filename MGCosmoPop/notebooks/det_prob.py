
# %%
import os
import sys
sys.path.append("/home/debian/software/MGCosmoPopPrivate/MGCosmoPop")

import numpy as np
import scipy.stats as sp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
%matplotlib widget

from population.astro.astroMassDistribution import BrokenPowerLawMass, PowerLawPlusPeakMass
from dataStructures.O1O2data import O1O2Data
from dataStructures.O3adata import O3aData
from cosmology.cosmo import Cosmo
cosmo = Cosmo()
import Globals

# events = {'use': ["GW150914", ],
#           'not_use': None }
# fname_data = os.path.join(Globals.dataPath, 'O1O2')
# gw = O1O2Data(fname_data, events_use=events)

# events = {'use': ["GW170814", ],
#           'not_use': None }
# fname_data = os.path.join(Globals.dataPath, 'O1O2')
# gw = O1O2Data(fname_data, events_use=events)

# events = {'use': ["GW190412", ],
#           'not_use': None }
# fname_data = os.path.join(Globals.dataPath, 'O3a')
# gw = O3aData(fname_data, events_use=events)

events = {'use': ["GW190814", ],
          'not_use': None }
fname_data = os.path.join(Globals.dataPath, 'O3a')
gw = O3aData(fname_data, events_use=events)


def sky2xyz(r, ra, dec):
    return r*np.cos(ra)*np.cos(dec), r*np.sin(ra)*np.cos(dec), r*np.sin(dec)

def xyz2sky(x1, x2, x3):
    r = np.sqrt(x1**2+x2**2+x3**2)
    return r, np.arctan2(x2,x1), np.arcsin(x3/r)

def jac_sky2xyz(r, ra, dec):
    return r**2*np.cos(dec)


def rejection_sampling( x1, x2, x3, wt ):
    wt /= np.max(wt)
    r = np.random.rand(len(x1))
    s = r < wt
    print(s.sum())
    return x1[s], x2[s], x3[s] 


# %%
idx = np.random.choice(np.arange(np.shape(gw.dL)[-1]), 10000)

dL, ra, dec = (gw.dL[:,idx]).flatten(), gw.ra[:,idx].flatten(), gw.dec[:,idx].flatten()
z           = cosmo.z_from_dLGW_fast(dL, H0=70, Om=0.3, w0=1, Xi0=1, n=0)

xs1, ys1, zs1 = sky2xyz(dL, ra, dec) 
xs2, ys2, zs2 = sky2xyz(z, ra, dec) 



# %%
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
ax.scatter3D(xs2, ys2, zs2, c='k', marker='.', s=0.5)

ax.view_init(30, 40);
plt.show()
# %%
