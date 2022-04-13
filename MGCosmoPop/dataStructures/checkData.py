#!/usr/bin/env python3
# -*- coding: utf-8 -*-                                          

import os
import sys
import numpy as np

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import Globals

import astropy.units as u

from cosmology.cosmo import Cosmo
import astropy.units as u
#from dataStructures.mockData import GWMockData, GWMockInjectionsData
from dataStructures.O3adata import O3aData, O3InjectionsData
from dataStructures.O1O2data import O1O2Data, O1O2InjectionsData
from dataStructures.O3bdata import O3bData


SNR_th = 8.

H0range = [67.74, ] #[20., 140.]
Xi0range = [0.1, 1., 10.]
Omrange = [0.31, ] #[0.05, 1.]

w0=1.
n=2.

def main():
    
    myCosmo = Cosmo(dist_unit = u.Gpc)   
    
    fnameO3b = os.path.join(Globals.dataPath, 'O3b')
    DataTestO3b = O3bData(fnameO3b, SNR_th=SNR_th)
    print(' ')
    fnameO3a = os.path.join(Globals.dataPath, 'O3a')
    DataTestO3a = O3aData(fnameO3a, SNR_th=SNR_th)
    print(' ')
    fnameO1O2 = os.path.join(Globals.dataPath, 'O1O2')
    DataTestO1O2 = O1O2Data(fnameO1O2, SNR_th=SNR_th)
    
    print('\nTotal events with SNR>%s: %s in O3b, %s in O3a, %s in O1-O2, %s in total.' %(SNR_th, DataTestO3b.Nobs, DataTestO3a.Nobs, DataTestO1O2.Nobs, DataTestO3b.Nobs+DataTestO3a.Nobs+DataTestO1O2.Nobs))
    print(' ')
    zmaxO3b=0
    zmaxO3a=0
    zmaxO1O2=0
    for H0 in H0range:
        for Xi0 in Xi0range:
            for Om in Omrange:
    
                zzO3b = myCosmo.z_from_dLGW_fast(DataTestO3b.dL[~np.isnan(DataTestO3b.dL)].max(), H0, Om, w0, Xi0, n)
                print('Redshift corresponding to max dL of %s Gpc in O3b with H0=%s, Om=%s, Xi0=%s: %s' %(DataTestO3b.dL[~np.isnan(DataTestO3b.dL)].max(), H0, Om, Xi0, zzO3b))
                if zzO3b>zmaxO3b:
                    zmaxO3b = zzO3b

                zzO3a = myCosmo.z_from_dLGW_fast(DataTestO3a.dL[~np.isnan(DataTestO3a.dL)].max(), H0, Om, w0, Xi0, n)
                print('Redshift corresponding to max dL of %s Gpc in O3a with H0=%s, Om=%s, Xi0=%s: %s' %(DataTestO3a.dL[~np.isnan(DataTestO3a.dL)].max(), H0, Om, Xi0, zzO3a))
                if zzO3a>zmaxO3a:
                    zmaxO3a = zzO3a

                zzO1O2 = myCosmo.z_from_dLGW_fast(DataTestO1O2.dL[~np.isnan(DataTestO1O2.dL)].max(), H0, Om, w0, Xi0, n)
                print('Redshift corresponding to max dL of %s Gpc in O1-O2 with H0=%s, Om=%s, Xi0=%s: %s' %(DataTestO1O2.dL[~np.isnan(DataTestO1O2.dL)].max(), H0, Om, Xi0, zzO1O2))
                if zzO1O2>zmaxO1O2:
                    zmaxO1O2 = zzO1O2


    print('\nMax redshifts final:')
    print('O3b: %s' %zmaxO3b)
    print('O3a: %s' %zmaxO3a)
    print('O1-O2: %s' %zmaxO1O2)



if __name__=='__main__':
    main()
