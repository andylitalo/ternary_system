# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:39:16 2018

@author: Andy
"""

# import packages
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import Functions as Fun
import cv2
import skimage.morphology
import skimage.filters
import skimage.feature
from scipy import ndimage
import ImageProcessingFunctions as IPF
import UserInputFunctions as UIF

# User parameters
folder = '..\\DATA\\2018-06-12\\segmentation\\low_threshold\\' # folder containing videos
fileString = 'sheath_cap_glyc_0100_*.png' # filestring of videos to analyze
hdr = 'sheath_cap_glyc_' # header of filename, including "_"
widthCapMicron = 556 # inner diameter of capillary in um
outerConversion = 0.86/0.76 # conversion to get actual inner flowrate
pixPerMicron = 1.4 # pixels per micron in image; set to 0 to calculate from image by clicking
uncertainty = 15 # uncertainty in streamwidth, given in pixels
#sigmaStreamWidth = 15 # pixels of uncertainty in stream width
eps = 0.2 # small parameter determining meaning of <<
# saving parameters
saveData = True
saveFolder = '..\\DATA\\2018-06-12\\segmentation\\'
dataFile = 'stream_width_vs_inner_flowrate_glyc.pkl'
# viewing parameters
viewIms = False
maskMsg = 'Click two opposing vertices to define rectangle around portion' + \
' of inner stream desired for analysis.' # message for clicking on mask
# plot parameters
A_FS = 16 # fontsize of axis titles
T_FS = 20 # fontsize of title
MS = 4 # markersize

def actual_flow_rate(innerFlowrate, innerConversion):
    """
    Converts the purported inner flowrate on the syringe pump to the actual 
    inner flowrate using the conversion factor innerConversion = actual/purported
    """
    return innerFlowrate*innerConversion
    
def stream_width(innerFlowrate, outerFlowrate, innerDiameter):
    """
    Gives expected radius of inner stream inside a cylindrical tube of a given
    inner diameter given the inner and outer flowrates
    """
    streamWidth = innerDiameter*np.sqrt(innerFlowrate/(innerFlowrate + outerFlowrate))
    
    return streamWidth
    
def get_flow_rates(fileName, hdr):
    """Extracts just the flowrates of the inner and outer streams from the file
    name. The header (hdr) must include all text up to and including the "_" 
    before the flow rates
    """
    # get length of header for appropriate offset
    l = len(hdr)
    # index for first character after header
    i = fileName.find(hdr) + l
    # strings for outer and inner flowrates based on naming convention
    outerStr = fileName[i:i+4]
    innerStr = fileName[i+5:i+9]
    outerFlowRate = int(outerStr) # ul/min
    innerFlowRate = int(innerStr)/10 # ul/min
    return innerFlowRate, outerFlowRate

###############################################################################
# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of images to consider
nIms = len(fileList)
# initialize data structures to store results
innerFlowRateList = np.zeros([nIms,1])
outerFlowRateList = np.zeros([nIms,1])
streamWidthList = np.zeros([nIms,1])
sigmaList = np.zeros([nIms,1])

# Loop through all videos
for i in range(nIms):
    # Parse the filename to get image info
    imPath = fileList[i]
    # Get inner flow rate and store
    innerFlowRateList[i], statedOuterFlowRate = get_flow_rates(imPath, hdr)
    # convert outer flow rate
    outerFlowRateList[i] = actual_flow_rate(statedOuterFlowRate, outerConversion)
    # Load image
    im = plt.imread(imPath)
    imCopy = np.copy(im) # copy
    # convert copy to 0-255 uint8 image
    imCopy *= 255 
    imCopy = imCopy.astype('uint8')
    # calculate pixels per micron by clicking on first image if no conversion given
    if i == 0 and pixPerMicron == 0:
        pixPerMicron = UIF.pixels_per_micron(imCopy, widthCapMicron)
    # find edges by determining columns with most saturated (255) pixels
    isEdgeX = np.where(imCopy[:,:,1]==255)[1]
    start = np.min(isEdgeX)
    print start
    end = np.max(isEdgeX)
    print end
    # show edges to check that they were found properly
    if viewIms:
        plt.figure()
        plt.imshow(imCopy[:,:start,:])
        plt.title('left edge')
        plt.figure()
        plt.imshow(imCopy[:,end:,:])
        plt.title('right edge')
    
    # user-defined mask for determining where to search
    # obtain vertices of mask from clicks
    maskVertices = UIF.define_outer_edge(imCopy,'rectangle',
                                     message=maskMsg)
    # create mask from vertices
    mask, maskPts = IPF.create_polygon_mask(imCopy, maskVertices)
    # mask image so only region around inner stream is shown
    imMasked = IPF.mask_image(imCopy, mask)
    # show masked edges and mask to ensure they were determined properly
    if viewIms:
        plt.figure()
        plt.imshow(imMasked[:,:start,:])
        plt.title('left edge')
        plt.figure()
        plt.imshow(imMasked[:,end:,:])
        plt.title('right edge')
    if viewIms:
        plt.figure()
        plt.imshow(mask)
        plt.title('mask')
    # sum all stream widths, average later by dividing by number of summations
    streamWidthSum = 0
    streamWidthSqSum = 0
    colCt = 0
    cutCols = [] # list of columns that were cut off by mask
    # loop through column indices to determine average stream width in pixels
    print 'looping.'
    for p in range(start, end):
        # extract current column from masked image
        col = imMasked[:,p,1]
        # skip if column is masked
        if np.sum(col) == 0:
            continue
        # locate saturated pixels
        is255 = col==255
        # if more saturated pixels than just the upper and lower bounds of the contour, stop analysis
        if np.sum(is255) > 2:
            print 'Error: more than 2 entries = 255.'
            continue
        elif np.sum(is255) < 2:
            cutCols += [p]
            continue
        # if only upper and lower bounds of contour are saturated, measure separation in pixels
        else:
            # count columns in proper format
            colCt += 1
            # calculate width of stream by taking difference of locations of saturated pixels
            streamWidth = np.diff(np.where(is255)[0])[0]
            print ' streamWidth = ' + str(streamWidth)
            streamWidthSum += streamWidth
            streamWidthSqSum += streamWidth**2
    # print range of columns cut off by mask
    if len(cutCols) > 0:
        print 'Error: part of contour cut out by mask, columns from ' + \
        str(min(cutCols)) + ' to ' + str(max(cutCols))

    # divide sum by number of elements to calculate the mean width
    print 'columns counted = ' + str(colCt)
    if colCt == 0:
        streamWidthMean = 0
        streamWidthStDev = 0
    else:
        streamWidthMean = float(streamWidthSum) / colCt
        streamWidthStDev = np.sqrt(float(streamWidthSqSum) / colCt - streamWidthMean**2)
    print 'mean stream width is ' + str(streamWidthMean) + \
    ' and stdev of stream width is ' + str(streamWidthStDev)
    # store stream width
    streamWidthList[i] = streamWidthMean
    sigmaList[i] = np.max(streamWidthStDev, uncertainty) # max between statistical noise and uncertainty
     
# Plot stream width as a function of inner flowrate on linear axes
pos = streamWidthList > 0
streamWidthList = streamWidthList[pos]
innerFlowRateList = innerFlowRateList[pos]
outerFlowRateList = outerFlowRateList[pos]
sigmaList = sigmaList[pos]
streamWidthMicron = streamWidthList/pixPerMicron
sigmaMicronList = sigmaList / pixPerMicron
plt.figure()
plt.loglog(innerFlowRateList, streamWidthMicron, 'b^', markersize=MS)
plt.errorbar(innerFlowRateList, streamWidthMicron, yerr=sigmaMicronList, fmt=None) # error bars
plt.grid()
plt.xlabel('Inner Flowrate [ul/min]', fontsize=A_FS)
plt.ylabel('Stream width [um]', fontsize=A_FS)
plt.title('Stream width vs. inner flow rate', fontsize=T_FS)
# power-law fit
# only consider small flow rates for which there is a predicted power-law dependence
small = innerFlowRateList < eps*outerFlowRateList #streamWidthMicron > 0
x = innerFlowRateList[small]
y = streamWidthMicron[small]
sigmaList = sigmaList[small]
outerFlowRateList = outerFlowRateList[small]
xlog = np.log(x)
ylog = np.log(y)
sigmaYLog = sigmaList / y
p, V = np.polyfit(xlog, ylog, 1, w=1/sigmaYLog, cov=True)
m, b = p
sigmaM = np.sqrt(V[0,0])
sigmaB = np.sqrt(V[1,1])
A = np.exp(b)
sigmaA = A*sigmaB
yFit = A*x**m
plt.plot(x, yFit, 'r--')
# compare to theory
yPredMicron = stream_width(innerFlowRateList[small], outerFlowRateList[small], widthCapMicron)
#yPredPix = pixPerMicron*yPredMicron
plt.plot(x, yPredMicron, 'b-')
plt.legend(['data','y = (' + str(round(A)) + '+/-' + str(round(sigmaA)) + \
')*x^(' + str(round(m,2)) + '+/-' + str(round(sigmaM,2)) + ')',
            'theory'], loc='best')
# ^what is the problem with the fit?     
       
### THIS THEORY IS NOT RIGHT ###
# linear fit based on exact theory
xTheory = (streamWidthList/pixPerMicron)**(-2) #convert width to um
yTheory = 1/innerFlowRateList
mTheory, bTheory = np.polyfit(xTheory, yTheory, 1)
outerFlowRateTheory = -1/bTheory
widthCapMicronTheory = np.sqrt(-mTheory/bTheory)
print 'theoretical outer flow rate is ' + str(outerFlowRateTheory) + ' ul/min.'
print 'theoretical width of capillary is ' + str(widthCapMicronTheory) + ' um.'
plt.figure()
plt.plot(xTheory, yTheory, 'b^', markersize=MS)
yTheoryFit = mTheory*xTheory + bTheory
plt.plot(xTheory, yTheoryFit, 'r--')
plt.xlabel('1/Inner Flowrate^2 [ul/min]', fontsize=A_FS)
plt.ylabel('1/Stream width [um]', fontsize=A_FS)
plt.title('Fit theory', fontsize=T_FS)
plt.legend(['data', 'Qin^-1 = ' + str(round(mTheory,2)) + 'Din^-2 + ' + str(round(bTheory,3))],
            loc='best')