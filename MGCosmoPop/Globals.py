
import os

#dirName  = os.path.join( os.path.dirname(os.path.abspath(__file__)), '..' )
dirName  = os.path.dirname( os.path.dirname(os.path.abspath(__file__)))
#print('Directory: %s' %dirName)
dataPath=os.path.join(dirName, 'data')
#print('Data Directory: %s' %dataPath)

detectorPath=os.path.join(dataPath, 'detectors')
#print('SNR Directory: %s' %detectorPath)


thetaPath=os.path.join(dataPath, 'theta')
#print('theta Directory: %s' %thetaPath)