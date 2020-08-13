from PIL import Image
from scipy import ndimage
import scipy
import numpy as np
import re
import os

def exitProgram(msg):
    print msg
    exit(0)

def flatFieldingOneImage(inputStruct, inputDir, outputDir, NumFrames):
    Channels = inputStruct['channels']
    startFlatfieldPos = int(inputStruct['startFlatfieldPos']) 
    endFlatfieldPos = int(inputStruct['endFlatfieldPos']) 

    FFImages = {}
    Imgstr = str('%09d' %(0))
    for b in range (0, len(Channels)): #for each channel
        FF = []
        for a in range(startFlatfieldPos, endFlatfieldPos + 1): #for each Flatfield position
            try:
                img = ndimage.imread(os.path.join(inputStruct['inputParentPath'],inputStruct['prefix']+str(a),'img_'+Imgstr+'_'+Channels[b]+'_000.png')) # open image
                FF.append(np.array(img).astype(float)) #generates a stack of images.
            except:
                exitProgram('Cannot read images for flatFielding at '+ os.path.join(inputDir,'img_'+Imgstr+'_'+Channels[b]+'_000.png'))
        FF= np.array(FF)

        FFImages[b] = ndimage.uniform_filter(np.median(FF, axis =0), size=5, mode="nearest") #calculates the median image smoothed
        FFImages[b]=np.double(FFImages[b])/np.max(np.max(FFImages[b], axis = 0), axis = 0) # Normalizes FFimage

    #main loop
    for ii in NumFrames: # for each frame in 0th Channel ??? consider the frames of zero channel
        Imgstr = str('%09d' %(ii)) #creates the img number as a str of 9 digits with leading zeros
       
        for jj in range(0, len(Channels)): #for each channel **make it ||
            try:
                Img = np.double(ndimage.imread(os.path.join(inputDir,'img_'+Imgstr+'_'+Channels[jj]+'_000.png'))) # open image
                FFImg = Img/ FFImages[jj] #Flatfield image
                FFImg = np.round(FFImg).astype(int)
                path = os.path.join(outputDir,'img_'+Imgstr+'_'+Channels[jj]+'_000.png')
                scipy.misc.toimage(FFImg, mode = 'I', high = np.max(FFImg), low = np.min(FFImg)).save(path)
            except:
                exitProgram('Cannot read/write images for flatFielding')

def flatFielding(inputStruct, inputDir, outputDir, NumFrames):
    Channels = inputStruct['channels']
    startFlatfieldPos = int(inputStruct['startFlatfieldPos']) 
    endFlatfieldPos = int(inputStruct['endFlatfieldPos']) 

    #main loop
    for ii in NumFrames: # for each frame in 0th Channel ??? consider the frames of zero channel
        Imgstr = str('%09d' %(ii)) #creates the img number as a str of 9 digits with leading zeros

        FFImages = {}
        for b in range (0, len(Channels)): #for each channel
            FF = []
            for a in range(startFlatfieldPos, endFlatfieldPos + 1): #for each Flatfield position
                try:
                    img = ndimage.imread(os.path.join(inputStruct['inputParentPath'],inputStruct['prefix']+str(a),'img_'+Imgstr+'_'+Channels[b]+'_000.png')) # open image
                    FF.append(np.array(img).astype(float)) #generates a stack of images.
                except:
                    exitProgram('Cannot read images for flatFielding at '+ os.path.join(inputDir,'img_'+Imgstr+'_'+Channels[b]+'_000.png'))
            FF= np.array(FF)

            FFImages[b] = ndimage.uniform_filter(np.median(FF, axis =0), size=5, mode="nearest") #calculates the median image smoothed
            FFImages[b]=np.double(FFImages[b])/np.max(np.max(FFImages[b], axis = 0), axis = 0) # Normalizes FFimage
       
        for jj in range(0, len(Channels)): #for each channel **make it ||
            try:
                Img = np.double(ndimage.imread(os.path.join(inputDir,'img_'+Imgstr+'_'+Channels[jj]+'_000.png'))) # open image
                FFImg = Img/ FFImages[jj] #Flatfield image
                FFImg = np.round(FFImg).astype(int)
                path = os.path.join(outputDir,'img_'+Imgstr+'_'+Channels[jj]+'_000.png')
                scipy.misc.toimage(FFImg, mode = 'I', high = np.max(FFImg), low = np.min(FFImg)).save(path)
            except:
                exitProgram('Cannot read/write images for flatFielding')

