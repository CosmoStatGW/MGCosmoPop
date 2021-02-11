import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from astropy import constants as const
import astropy.units as u
import sys
import requests
from scipy.optimize import fsolve
import os


dirName  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
dataPath=os.path.join(dirName, 'data')


which_unit=u.Mpc

H0GLOB=67.9 # In km/s/Mpc
Om0GLOB=0.3
Xi0GLOB=1.
nGLOB=1.91

cosmoglob = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

eps=1e-15

zGridGLOB = np.concatenate([ np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000), np.logspace(start=np.log10(8), stop=5, base=10, num=100)])

#zGridGLOB = np.sort(zGridGLOB)
#zGridGLOB=np.unique(zGridGLOB, return_counts=False)  


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
    return 4*np.pi*FlatLambdaCDM(H0=70, Om0=Om0GLOB).differential_comoving_volume(z).value*(70/clight)**3


def dV_dz(z, H0):
    '''
    Jacobian of comoving volume, with correct dimensions [Mpc^3]. Depends on H0
    '''
    return 4*np.pi*FlatLambdaCDM(H0=H0, Om0=Om0GLOB).differential_comoving_volume(z).value


def s(z, Xi0, n):
    return (1+z)*Xi(z, Xi0, n)

def sPrime(z, Xi0, n):
    return Xi(z, Xi0, n)-n*(1-Xi0)/(1+z)**n



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
    return (cosmo.luminosity_distance(z).to(which_unit).value)*Xi(z, Xi0, n)


def Xi(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n


######################
# SOLVERS FOR DISTANCE-REDSHIFT RELATION
######################

def z_from_dLGW_fast(r, H0, Xi0, n):
    '''
    Returns redshift for a given luminosity distance r (in Mpc by default). Vectorized
    '''
    #dLeps=dLGW(eps, H0, Xi0, n)
    z2dL = interpolate.interp1d( dLGridGLOB/H0*H0GLOB*Xi(zGridGLOB, Xi0, n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,0.), assume_sorted=True)
    return z2dL(r)#np.where(r>dLeps, z2dL(r), 0.) #np.where(z2dL(r)>eps, z2dL(r), 0.)


def z_from_dLGW_fast_1(r, H0, Xi0, n):
    #print('r shape: %s' %str(r.shape))
    #if not np.isscalar(r):
    #    if np.ndim(r)==1:
    #        return  np.array([z_from_dLGW(dL_GW_val, H0, Xi0, n) for dL_GW_val in r] )
    #    else:
    #        
    #else:
    #    return np.array([z_from_dLGW(r, H0, Xi0, n),] )
    
    return np.vectorize(z_from_dLGW)(r, H0, Xi0, n)

def z_from_dLGW(dL_GW_val, H0, Xi0, n):
    '''Returns redshift for a given luminosity distance dL_GW_val (in Mpc by default)                                         '''
    func = lambda z : dLGW(z, H0, Xi0, n) - dL_GW_val
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
        
        
####################################
# TELEGRAM bot
####################################


def telegram_bot_sendtext(bot_message):
    
    bot_token = '1656874236:AAE_oNQLTEYxCcj0352LU1gikG0CNdpjnUg'
    bot_chatID = '463660975'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
