# %%
import sys
import numpy as np
sys.path.append("/home/debian/software/MGCosmoPopPrivate/MGCosmoPop/cosmologyc")
import cosmo
# %%


c = cosmo.fwCDM()

zs = np.linspace(0,2,10**6)


# %%

%%time
a = [c.dC(z) for z in zs]

#%%