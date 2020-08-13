import re
import os
import scipy
import scipy.io as sio
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/home/regotlab/anaconda/lib/python2.7/site-packages/")
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp


def initObjectFeatures(data, featureNames, featureSize, objectIdx):

	objNamesNow = data['objectSetNames']
	if objectIdx != None: 
		objNamesNow = [data['objectSetNames'][objectIdx]]
		
	for i in range(0,len(objNamesNow)):
		if objNamesNow[i] not in data: data[objNamesNow[i]] = {}
		for j in range(0,len(featureNames)):
			data[objNamesNow[i]][featureNames[j]] = np.zeros(featureSize)

	return data

def initIntensityFeatures(data, handles, baseSize, intensityGroupName):

	numObjects = len(data['objectSetNames'])
	objectHasIntensity = np.zeros(numObjects)
	intensityFeatureNames = []

	imageSetNamesPer = []
	imageSetNames = []

	for i in range(0, numObjects):
		fieldsNow = handles.Measurements.__dict__[data['objectSetNames'][i]]._fieldnames
		intensityFeatureNamesFull = []
	
		for j in fieldsNow:
			temp = re.split('_', j)
			if intensityGroupName == temp[0]:
				intensityFeatureNamesFull.append(j)
	
		if len(intensityFeatureNamesFull): objectHasIntensity[i] = 1	
	
		if objectHasIntensity[i]:
			intensityFeatureNamesFull.sort()		
			endName = []
			middleName = []
			for i in intensityFeatureNamesFull:
				temp = re.split('_', i)
				if temp[2] not in endName:	
					endName.append(temp[2])
					if temp[2] not in imageSetNames: imageSetNames.append(temp[2])
				if temp[1] not in middleName: middleName.append(temp[1])
	
			imageSetNamesPer.append(endName)
	
			if len(intensityFeatureNames) == 0:
				middleName.sort()
				intensityFeatureNames = middleName

	data['imageSetNames'] = imageSetNames

	for i in range(0, numObjects):
		if objectHasIntensity[i]:
			data[data['objectSetNames'][i]]['imageSetIdx'] = []
			data[data['objectSetNames'][i]]['imageSetNames'] = []		
			for j in range(0, len(imageSetNamesPer[i])):
				try: 
					index = imageSetNames.index(imageSetNamesPer[i][j])
					data[data['objectSetNames'][i]]['imageSetIdx'].append(index+1) #index from 1 as matlab
					data[data['objectSetNames'][i]]['imageSetNames'].append(data['imageSetNames'][index]) 
				except:
					print imageSetNames,imageSetNamesPer[i][j]
					continue								
			for j in range(0, len(intensityFeatureNames)):
				data[data['objectSetNames'][i]][intensityFeatureNames[j]] = np.zeros( baseSize +[len(imageSetNamesPer[i])] )
			
	return (data, intensityFeatureNames, objectHasIntensity)


def getImageFileData(data, handles, trueFramesIdx):
	sChar = '_'
	fileKey = 'FileName'

	imageFieldSplit = handles.Measurements.Image._fieldnames
	imageSetNamesAll = []
	for i in imageFieldSplit:
		temp = re.split('_',i)
		if fileKey == temp[0]: imageSetNamesAll.append(temp[1])

	imageSetChannels = []
	imageFilenamesNest = []

	imageSetIdx = []
	for i in range(0, len(data['imageSetNames'])):
		flag = 1
		for j in range(0, len(imageSetNamesAll)):
			if data['imageSetNames'][i] == imageSetNamesAll[j]:
				imageSetIdx.append(j)
				flag = 0
				break
		if flag: imageSetIdx.append(-1)

	for i in range(0,len(data['imageSetNames'])):
		if imageSetIdx[i] > -1:
			temp = handles.Measurements.Image.__dict__[fileKey+sChar+imageSetNamesAll[imageSetIdx[i]]]
			temp1 = []
			for j in trueFramesIdx: temp1.append(temp[j])
			imageFilenamesNest.append(temp1)
			filenameSplit = re.split('_',temp[0])
			imageSetChannels.append(filenameSplit[2])

	imageSetFilenames = []
	for i in range(0,len(data['imageSetNames'])):
		imageSetFilenames.append(imageFilenamesNest[i])

	frames = [int(x[4:13]) for x in imageSetFilenames[0]]

	return (imageSetChannels, imageSetFilenames, frames)

def linkResultSets(data, handles): 
	# can only deal with objects/images created by the following modules
	moduleNamesPossible = ['IdentifyPrimaryObjects', 'IdentifySecondaryObjects', 'IdentifyTertiaryObjects', 'ExpandOrShrinkObjects', 'ConvertObjectsToImage', 'Morph', 'ApplyThreshold', 'EnhanceOrSuppressFeatures']
	varvalIdxForResult = [1, 1, 2, 1, 1, 1, 1, 1]
	varvalIdxForParent = [0, 3, 0, 0, 0, 0, 0, 0]

	moduleIdx = []
	moduleTypeIdx = []
	index = 0
	for i in handles.Settings.ModuleNames:
		temp = re.split('\.',i)
		moduleNames = str(temp[len(temp)-1])
		try:
			location = moduleNamesPossible.index(moduleNames)
			moduleIdx.append(index)
			moduleTypeIdx.append(location)
		except:	
			location = 0
		index = index + 1

	# list of list. where a row represents each module [col 1 result, col 2 firstParent, col 3 rootParent]
	resultSetChain = []
	for i in range(0, len(moduleIdx)):
		col_1 = handles.Settings.VariableValues[moduleIdx[i]][varvalIdxForResult[moduleTypeIdx[i]]]
		firstParent = handles.Settings.VariableValues[moduleIdx[i]][varvalIdxForParent[moduleTypeIdx[i]]]
		col_2 = firstParent	
		idxFirstParent = -1
		for j in range(0, len(resultSetChain)):
			if firstParent == resultSetChain[j][0]:
				idxFirstParent = j
				break
		if idxFirstParent > -1: col_3 = resultSetChain[idxFirstParent][2]
		else: col_3 = firstParent
		resultSetChain.append([col_1, col_2, col_3])

	rootImageSetIdxForObjectSets = []
	for i in data['objectSetNames']:
		for j in resultSetChain:
			if i == j[0]:
				for k in range(0, len(data['imageSetNames'])):
					if data['imageSetNames'][k] == j[2]: rootImageSetIdxForObjectSets.append(k+1)
				break

	return (resultSetChain, rootImageSetIdxForObjectSets)


def getFeatures(objectSetNames, handles, excludeFeatureGroups, excludeFeatureNamesFull):

	objectHasGroup = []
	for i in objectSetNames:
		groupsWithFeatures = {}
		names = handles.Measurements.__dict__[i]._fieldnames
		
		for j in names:
			temp = re.split('_', j)
			if (temp[0] not in excludeFeatureGroups) and (j not in excludeFeatureNamesFull): 
				feature = j[j.index('_')+1 :]
				if temp[0] not in groupsWithFeatures: groupsWithFeatures[temp[0]] = []
				groupsWithFeatures[temp[0]].append(feature)
		objectHasGroup.append(groupsWithFeatures)

	return objectHasGroup


def getThresholds(handles, trueFramesIdx):
	threshKey = 'Threshold_FinalThreshold_'
	fieldsTmp = handles.Measurements.Image._fieldnames
	thresholdResultSetNames = []
	for i in fieldsTmp:
		if i.startswith(threshKey):
			thresholdResultSetNames.append(i[len(threshKey):])

	thresholdIntensity = []
	thresholdResultSetNames.sort()
	for i in range(0, len(thresholdResultSetNames)):	
		temp = handles.Measurements.Image.__dict__[threshKey+thresholdResultSetNames[i]] 
		temp1 = []
		for j in trueFramesIdx: temp1.append(temp[j])
		thresholdIntensity.append(temp1)

	return (thresholdResultSetNames, thresholdIntensity)


def linkThresholdsObjectSets(data, handles):

	thresholdIdxForObjectSets = np.zeros(len(data['objectSetNames']))

	for i in range(0, len(data['objectSetNames'])):
		for j in range(0, len(data['resultSetChain'])):
			if data['resultSetChain'][j][0] == data['objectSetNames'][i]:
				objectSetNameChainIdx = j
				break
		fields = handles.Measurements._fieldnames
		sameRootParent = []
		isObject = []
		for j in range(0, objectSetNameChainIdx):
			if data['resultSetChain'][objectSetNameChainIdx][2] == data['resultSetChain'][j][2]: 
				sameRootParent.append(1)
			else: 
				sameRootParent.append(0)
			if data['resultSetChain'][j][0] in fields: isObject.append(1)	
		chainIdx = -1
		for k in range(0,len(sameRootParent)): 
			if (sameRootParent[k] == 1) and (isObject[k] == 1):
				chainIdx = k
				break	
		if chainIdx> -1:
			try:
				ind = data['thresholdResultSetNames'].index(data['resultSetChain'][chainIdx][0])
				thresholdIdxForObjectSets[i] = ind+1 #+1 to match with matlab
			except:
				continue

	return thresholdIdxForObjectSets


def dataFrameToDictForTracking(tracks, nFrames):

	newTrack  = {}
	if len(tracks) == 0: return []
	for j in range(0, nFrames):
		index = 0
		x = list(tracks['x'][j])
		y = list(tracks['y'][j])
		for i in tracks['particle'][j]:
			if (int(i)+1) not in newTrack: newTrack[int(i)+1] = {'x': [],'y':[],'frame':[]}
			newTrack[(int(i)+1)]['x'].append(x[index]) 
			newTrack[(int(i)+1)]['y'].append(y[index]) 
			newTrack[(int(i)+1)]['frame'].append(j) 
			index = index + 1

	index = 0
	newContTracks = {}
	for i in newTrack:
		newContTracks[index] = newTrack[i]
		index = index + 1

	return newContTracks

def convertNpArray(data, numObjects):

	for i in range(0, numObjects): 
		data[data['objectSetNames'][i]]['imageSetIdx'] = np.array(np.matrix(data[data['objectSetNames'][i]]['imageSetIdx']).T)#, dtype = 'double')
		data[data['objectSetNames'][i]]['imageSetNames'] = np.array(np.matrix(data[data['objectSetNames'][i]]['imageSetNames']).T, dtype = 'object')

	data['inputStruct']['objectSetNames'] = np.array([data['inputStruct']['objectSetNames']], dtype='object')
	data['inputStruct']['subfolderNames'] = np.array(np.matrix([data['inputStruct']['subfolderNames']]).T, dtype='object')
	
	data['objectSetNames'] = np.array(np.matrix([data['objectSetNames']]).T, dtype='object')
	
	data['frames'] = np.array(data['frames'])#, dtype='double')
	data['imageSetChannels'] = np.array(np.matrix(data['imageSetChannels']).T, dtype='object')
	data['imageSetFilenames'] = np.array(np.matrix(data['imageSetFilenames']), dtype='object')
	data['imageSetNames'] = np.array(np.matrix(data['imageSetNames']).T, dtype='object')

	data['rootImageSetIdxForObjectSets'] = np.array(np.matrix(data['rootImageSetIdxForObjectSets']).T)#, dtype='double')
	
	data['resultSetChain'] = np.array(data['resultSetChain'], dtype='object')
	
	data['thresholdResultSetNames'] = np.array(np.matrix(data['thresholdResultSetNames']).T, dtype='object')
	data['thresholdIntensity'] = np.array(data['thresholdIntensity'], dtype='double')
	data['thresholdIdxForObjectSets'] = np.array(np.matrix(data['thresholdIdxForObjectSets']).T)
	
	return data

####################################################################### 

def trackOrganizeCpData(inputStruct, subfolderNow):

	# currently ignores the 'Location' fields produced by the intensity module currently ignores 'RadialDistribution' fields
	data = {}
	data['subfolderMetadata'] = {'subfolderName' : subfolderNow}
	data['inputStruct'] = inputStruct
	trackStatus = 'failure'

	# load the cpData file (only variable is 'handles')
	dirPath = os.path.join(inputStruct['parentPath'],subfolderNow)
	filePath = os.path.join(dirPath, inputStruct['cpDataFilename'])
	mat = sio.loadmat(filePath, struct_as_record=False, squeeze_me=True)
	handles = mat['handles']

	# check objectSetNames
	numObjects = len(inputStruct['objectSetNames'])

	objectExists = []
	for i in inputStruct['objectSetNames']:
		if i not in ['Experiment', 'Image']:
			if i in handles.Measurements._fieldnames:
				objectExists.append(i)
			else:
				print 'This object set name is invalid:', i, 'Abort trackOrganizeCpData!'
				return trackStatus
		else:
			print 'This object set name is invalid:', i,'Abort trackOrganizeCpData!'
			return trackStatus
	data['objectSetNames'] = inputStruct['objectSetNames']

	# define base object features
	objectFeatureNamesBase = ['label', 'x', 'y']
	labelFieldname = ['Number_Object_Number']
	xyFieldnames = ['Location_Center_X', 'Location_Center_Y']

	objectFeaturePointersBase = labelFieldname + xyFieldnames

	# in case the first frame that CP analyzed was not the first in the directory
	# cellprofiler seems to make empty cells only for skipped frames in the front, 
	# not in the back, but this should work no matter what

	nFrames = len(handles.Measurements.Image.Group_Index)
	if nFrames == 0: trueFramesIdx = []
	else: trueFramesIdx = [i[0] for i in enumerate(handles.Measurements.Image.Group_Index)]

	# get and format centroids of nuclei for tracking
	trackObjName = data['objectSetNames'][0]
	temp = handles.Measurements.__dict__[trackObjName].__dict__[xyFieldnames[0]] 
	x = []
	for i in temp:
		for j in i: x.append(j)
	temp = handles.Measurements.__dict__[trackObjName].__dict__[xyFieldnames[1]] 
	y = []
	nObjectsFrame = [] # get frames for each object for tracking
	frameIdxForObjects = [] # get frames for each object for tracking
	index = 0
	for i in temp:
		for j in i:
			y.append(j)
			frameIdxForObjects.append(index)
		nObjectsFrame.append(len(i)) #length is same for both X and Y position #Sajia
		index = index + 1

	# run the tracking
	if nFrames > 1:
		i = 0
		while (i < len(inputStruct['trackParam']['maxdisp'])) and (trackStatus == 'failure'):
			#call trackpy for tracking
			try:
				frameData = DataFrame(columns=['x','y', 'frame'])
				frameData['x'] = x
				frameData['y'] = y
				frameData['frame'] = frameIdxForObjects
				t = tp.link_df(frameData, inputStruct['trackParam']['maxdisp'][i], memory = inputStruct['trackParam']['mem'])
				tracks = tp.filter_stubs(t,inputStruct['trackParam']['good'])
				trackStatus = 'success'
			except:
				trackStatus = 'failure'
			i = i + 1         

		data['inputStruct']['trackStatus'] = [trackStatus]
		if trackStatus == 'failure': 
			print 'Tracking status: ', trackStatus
			return trackStatus
		else: data['inputStruct']['maxdispSuccess'] = inputStruct['trackParam']['maxdisp'][i-1]
		allTracks = dataFrameToDictForTracking(tracks, nFrames)		

	else: 
		data['inputStruct']['trackStatus'] = 'success'
		tracks = DataFrame(columns=['x','y', 'frame', 'particle'])
		tracks['x'] = x
		tracks['y'] = y
		tracks['frame'] = frameIdxForObjects
		tracks['particle'] = range(0, nObjectsFrame[0])

		allTracks  = {}
		for i in range(1, nObjectsFrame[0] + 1): #reassign particle number from 1,2,3...
			newTrack[i] = {'x' : x[i], 'y' : y[i], 'frame' : i}

	if tracks.shape[0] > 0: nTracks = tracks['particle'].nunique()
	else: 
		print 'No tracking particle exists! Aborting!!'
		return trackStatus

	# initialize the struct for base object features (rows are cells/tracks, cols are frames)
	data = initObjectFeatures(data, objectFeatureNamesBase, [nTracks, nFrames], None)

	# initialize intensity features
	intensityGroupName = 'Intensity'
	(data, intensityFeatureNames, objectHasIntensity) = initIntensityFeatures(data, handles, [nTracks, nFrames], intensityGroupName)

	# find image channels and image filenames for each imageSet
	(data['imageSetChannels'], data['imageSetFilenames'], data['frames']) = getImageFileData(data, handles, trueFramesIdx)

	# find the root imageSet for each object
	(data['resultSetChain'], data['rootImageSetIdxForObjectSets']) = linkResultSets(data, handles) 

	#check for and initialize other object features (AreaShape, Neighbors)
	excludeFeatureGroups = ['RadialDistribution', 'Location', 'Number', 'Intensity', 'Parent', 'Children']
	excludeFeatureNamesFull = ['AreaShape_Center_X', 'AreaShape_Center_Y']

	objectHasGroupOther = getFeatures(data['objectSetNames'], handles, excludeFeatureGroups, excludeFeatureNamesFull)

	for i in range(0, len(objectHasGroupOther)):
		for j in objectHasGroupOther[i]:
			data = initObjectFeatures(data, objectHasGroupOther[i][j], [nTracks, nFrames], i)

	# get the threshold data
	(data['thresholdResultSetNames'], data['thresholdIntensity']) = getThresholds(handles, trueFramesIdx)

	# find what is probably the correct threshold index for each objectSet assuming the thresholds you want are based on imageSets from the LoadImages module
	data['thresholdIdxForObjectSets'] = linkThresholdsObjectSets(data, handles)

	# construct matrices
	sChar = '_'

	for currentTrack in range(0, nTracks): #for each tracked cell it was ii
		# get the relevant part of tracks
		trackNow = allTracks[currentTrack]
		
		framePosition = -1
		for frameIdxNow in trackNow['frame']: # for each frame for which the cell is tracked ; it was jj
			framePosition = framePosition + 1
			if frameIdxNow not in trueFramesIdx: continue
			# for calculating label
			xValues = handles.Measurements.__dict__[trackObjName].__dict__[xyFieldnames[0]][frameIdxNow]
			yValues = handles.Measurements.__dict__[trackObjName].__dict__[xyFieldnames[1]][frameIdxNow]
			label = -1
			for k in range(0, len(xValues)):
				if (trackNow['x'][framePosition] == xValues[k]) and (trackNow['y'][framePosition] == yValues[k]): 
					label = k
			if label == -1:
				print 'Something went Wrong, X or Y position doesn\'t match with tracked particles. Aborting... '
				return trackStatus		 	

			for k in range(0, numObjects): # for each object
				
				# base features
				for m in range(0, len(objectFeatureNamesBase)):
					data[data['objectSetNames'][k]][objectFeatureNamesBase[m]][currentTrack][frameIdxNow] = float(handles.Measurements.__dict__[data['objectSetNames'][k]].__dict__[objectFeaturePointersBase[m]][frameIdxNow][label])

				# other object features
				for m in objectHasGroupOther[k]:
					feature = objectHasGroupOther[k][m]
					for n in feature:
						data[data['objectSetNames'][k]][n][currentTrack][frameIdxNow] = float(handles.Measurements.__dict__[data['objectSetNames'][k]].__dict__[m+sChar+n][frameIdxNow][label])

				# intensity features
				if objectHasIntensity[k]:
					indexM = 0
					for m in data[data['objectSetNames'][k]]['imageSetIdx']:
						for n in intensityFeatureNames:
							fullFieldStr = intensityGroupName + sChar + n + sChar + data['imageSetNames'][m-1] # ??? index starts at 1, matlab
							data[data['objectSetNames'][k]][n][currentTrack][frameIdxNow][indexM] = float(handles.Measurements.__dict__[data['objectSetNames'][k]].__dict__[fullFieldStr][frameIdxNow][label])
						indexM = indexM + 1
	# save
	ratio, status = clacRatio(data)
	if status:
		data['nuclei']['MedianIntensityRatio'] = ratio
	#import pdb; pdb.set_trace()

	indexForDot = inputStruct['cpDataFilename'].index('.')
	filenameOrig = inputStruct['cpDataFilename'][: indexForDot]
	extensionForFile = inputStruct['cpDataFilename'][indexForDot:]
	outputFilepath = os.path.join(inputStruct['parentPath'], subfolderNow, filenameOrig+inputStruct['outputSuffix']+extensionForFile)
	data = convertNpArray(data, numObjects)
	sio.savemat(outputFilepath, {'data':data})
	return trackStatus

def clacRatio(data):
	if 'cytoring' in data and 'nuclei' in data:
		channels = data['nuclei']['imageSetIdx']
		allSize = data['nuclei']['MedianIntensity'].shape
		ratioMatrix = np.zeros(allSize)

		thresholdIndex = -1
		try:
			thresholdIndex = data['thresholdResultSetNames'].index('nuclei')
		except:
			print 'Ignoring background noise while calculating MedianIntensityRatio'

		indexForCytoring = 0
		for index in channels:
			if (index+1) not in data['cytoring']['imageSetIdx']: continue

			for row in range(0,allSize[0]):
				for col in range(0,allSize[1]):
					n=data['nuclei']['MedianIntensity'][row][col][index]
		   			c=data['cytoring']['MedianIntensity'][row][col][indexForCytoring]
		   			if thresholdIndex>-1:
						thr=data['thresholdIntensity'][thresholdIndex][col] ##background
						c=c-(thr*0.8)
						n=n-(thr*0.8)
					ratio=c/n
		    		#pJUN=Smoothdata(pJUN,3);
					ratioMatrix[row][col][index]=ratio
			indexForCytoring = indexForCytoring + 1
	
		return ratioMatrix, 1 
	else:
		return 0,0

#check input parameters validity
def trackOrganizeCpDataDir(inputStruct): 

	# in case extension isn\'t supplied
	if inputStruct['cpDataFilename'].endswith('.mat') == False: inputStruct['cpDataFilename'] = inputStruct['cpDataFilename'] + '.mat'

	# check existence of subfolders
	if os.path.isdir(inputStruct['parentPath']) == False:
		print 'ERROR: ',inputStruct['parentPath'],' is not a folder on the path'
		return

	dirListPre = os.listdir(inputStruct['parentPath'])
	dirListFull = []
	for i in dirListPre:
		fullPath = os.path.join(inputStruct['parentPath'], i)
		if (i.startswith('.') == False) and os.path.isdir(fullPath): dirListFull.append(i)
	 
	if len(inputStruct['subfolderNames']) == 0: dirList = dirListFull
	else:
		dirList = []
		for i in inputStruct['subfolderNames']:
			if i in dirListFull: dirList.append(i)
			else: print 'Warning: ',i, 'subfolder does not exist in ', inputStruct['parentPath']
			
	# check existence of cpData files
	for i in dirList:
		dirPath = os.path.join(inputStruct['parentPath'],i)
		filePath = os.path.join(dirPath, inputStruct['cpDataFilename'])
		if os.path.exists(filePath) == False:
			print 'Warning: The ',dirPath,' subfolder does not contain ', inputStruct['cpDataFilename']

	# call the main function
	failureDirList = []
	for i in dirList:
		temp = inputStruct.copy()
		status = trackOrganizeCpData(temp, i)
		
		if status == 'failure': failureDirList.append(i)

	#write failure directories list
	if len(failureDirList):
		try:
			fullPath = os.path.join(inputStruct['parentPath'], 'failedTracking.txt')
			fw = open(fullPath, 'w')
		except:
			print 'Cannot write file at ', inputStruct['parentPath']
			return

		for i in failureDirList: fw.write(i+'\n')
		fw.close()






