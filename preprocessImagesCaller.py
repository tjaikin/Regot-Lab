'''



'''
import sys
sys.path.append("/ImageAnalysisPython/")

from src.preprocessImages import *


inputStruct = {}
inputStruct['inputParentPath'] = '/RawImages'
inputStruct['outputParentPath'] = '/Images'

# if empty, all subfolders in the inputPath will be preprocessed
inputStruct['subfolderNames'] = [] 

# these must exactly match the channel names in micro-manager. 
# the first channel will be used for stitching (if applicable) and for image registration.
inputStruct['channels'] = ['Far-red','CFP','YFP','TRITC'] 

inputStruct['applyFlatFielding'] = True
inputStruct['prefix'] = 'Pos'
inputStruct['UseSingleFrame'] = True
# If UseSingleFrame == False, then define the start position and end position
inputStruct['startFlatfieldPos'] = 0
inputStruct['endFlatfieldPos'] = 1


inputStruct['applyImageRegistration'] = True

# if processFrames is empty, all frames will be processed.
# if processFrames has values greater than or equal to zero, only those frames will be processed.
# if processFrames has values less than zero, those frames will be ignored.
inputStruct['processFrames'] = []

# TRUE if no print statement
inputStruct['quite'] = False

# number of CPUs; 0 will consider all CPUs
inputStruct['CPU_NUMs'] = 0

#### call the mother function
preprocessImagesCaller(inputStruct)
