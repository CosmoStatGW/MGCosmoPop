#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:00:02 2021

@author: Michi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:52:04 2021

@author: Michi
"""


import time
import os
import sys

from utils import *


#######################
# Global variables
dirName  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
dataPath=os.path.join(dirName, 'data')

fout='test'

maxNtaus = 150
checkTauStep = 100
# Nobs=100

########################


def main():
    
    in_time=time.time()
    
    
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--observingRun", default='O2', type=str, required=True)
    #parser.add_argument("--wf_model", default='', type=str, required=True)
    #FLAGS = parser.parse_args()
    

    
    #####
    # Out file
    ####
    
    # St out path and create out directory
    out_path=os.path.join(dirName, 'results', fout)
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    else:
       print('Using directory %s for output' %out_path)
       
       
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    
    #shutil.copy('config.py', os.path.join(out_path, 'config_original.py'))
    
    #####
    #####
    
    #######
    
    
    # With gaussian :
    # - save chanin in .hdf5
    # - check autocorrelation time
    # - check convergence & stop 
    
    
    
    # Example: save plot 
    #plt.plot(a,b )
    #plt.savefig(os.path.join(out_path,'my_plot.png'))
    #plt.close();
    
    
    ## Save corner plot
    # fig1 = corner.corner( flat_samples, labels=labels,  #truths=[]);
    # fig1.savefig(os.path.join(out_path, 'corner.pdf'))
    
    
    ######
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    myLog.close() 




#######################################################

if __name__=='__main__':
    main()

    
