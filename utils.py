import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from astropy import constants as const
import astropy.units as u
import sys


which_unit=u.Mpc

H0GLOB=67.9 # In km/s/Mpc
Om0GLOB=0.3
Xi0GLOB=1.
nGLOB=1.91

cosmoglob = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

zGridGLOB = np.logspace(start=-10, stop=5, base=10, num=1000)
dLGridGLOB = cosmoglob.luminosity_distance(zGridGLOB).to(which_unit).value

clight=const.c.value*1e-03 # c in km/s


######################
# FUNCTIONS FOR COSMOLOGY
######################

def uu(z):
    '''
    Dimensionless comoving distance. Does not depend on H0
    '''
    return 70/clight*FlatLambdaCDM(H0=70, Om0=Om0GLOB).comoving_distance(z).value

def E(z):
    '''
    E(z). Does not depend on H0
    '''
    return FlatLambdaCDM(H0=70, Om0=Om0GLOB).efunc(z)

def j(z):
    '''
    Jacobian of comoving volume, dimensioneless. Does not depend on H0
    '''
    return FlatLambdaCDM(H0=70, Om0=Om0GLOB).differential_comoving_volume(z).value*(70/clight)**3

####### For j(z), differential_comoving_volume(z) instead of comovin_volume(z)

def dV_dz(z, H0):
    '''
    Jacobian of comoving volume, with correct dimensions [Mpc^3]. Depends on H0
    '''
    return FlatLambdaCDM(H0=H0, Om0=Om0GLOB).differential_comoving_volume(z).value #*(70/clight)**3


def s(z, Xi0, n):
    return (1+z)*Xi(z, Xi0, n)

def sPrime(z, Xi0, n):
    return Xi(z, Xi0, n)-n*(1-Xi0)/(1+z)**n


### This is not used; if using, double-check units
#def Jac(z, Xi0, n):
#    return (sPrime(z, Xi0, n)*u(z)+s(z Xi0, n)/E(z))

def ddL_dz(z, H0, Xi0, n):
    '''
     Jacobian d(DL)/dz  [Mpc]
    '''
    return (sPrime(z, Xi0, n)*uu(z)+s(z, Xi0, n)/E(z))*(clight/H0)


def dLGW(z, H0, Xi0, n):
    '''                                                                                                          
    Modified GW luminosity distance in units set by utils.which_unit (default Mpc)                                                                           
    '''
    cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
    return (cosmo.luminosity_distance(z).to(which_unit).value)*Xi(z, Xi0, n=n)


def Xi(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n


######################
# SOLVERS FOR DISTANCE-REDSHIFT RELATION
######################

def z_from_dLGW_fast(r, H0, Xi0, n):
    '''
    Returns redshift for a given luminosity distance r (in Mpc by default). Vectorized
    '''
    z2dL = interpolate.interp1d(dLGridGLOB/H0*H0GLOB*Xi(zGridGLOB, Xi0, n=n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
    return z2dL(r)



def z_from_dLGW(dL_GW_val, H0, Xi0, n):
    '''Returns redshift for a given luminosity distance dL_GW_val (in Mpc by default)                                         '''
    from scipy.optimize import fsolve
    func = lambda z : dLGW(z, H0, Xi0, n=n) - dL_GW_val
    z = fsolve(func, 0.5)
    return z[0]


######################
# OTHER
######################

def flatten2(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten2(list_of_lists[0]) + flatten2(list_of_lists[1:])
    return list_of_lists[:1] + flatten2(list_of_lists[1:])

###### flatten2 instead of flatten (there is a conflict)



class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
