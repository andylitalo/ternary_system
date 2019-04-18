# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:44:10 2018

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
import UserInputFunctions as UIF
import ImageProcessingFunctions as IPF

# User Parameters
# data for video
folder = '..\\DATA\\2018-06-12\\' # folder containing videos
fileString = 'sheath_cap_glyc_0100*.jpg' # filestring of videos to analyze
maskMsg = 'Click opposing corners of rectangle to include desired section of image.'
maskDataFile = 'maskData_glyc_180620.pkl'
# analysis parameters
meanFilter = True
kernel = np.ones((5,5),np.float32)/25 # kernel for gaussian filter
bPct = 45 # percentile of pixels from blue channel kept
rNegPct = 60 # percentile of pixels from negative of red channel kept
# Structuring element is radius 2 disk
selem = skimage.morphology.disk(10)
nDilations = 0
showIm = False
minSize = 250
# saving parameters
saveIm = True
saveFolder = '..\\DATA\\2018-06-12\\segmentation\\low_threshold\\'

def show_im(im, title):
    """
    Shows image in new figure with given title
    """
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    
def get_contour(imBin, showIm):
    """
    returns image showing only a 1-pixel thick contour enclosing largest object
    in the binary image given.
    """
    #find edge using contour finding
    contours, hierarchy = cv2.findContours(imBin.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # return blank image if no contours
    if len(contours) == 0:
        return np.zeros_like(imBin,dtype='uint8')
    cntMax = sorted(contours, key=cv2.contourArea, reverse=True)[0] # get largest contour
    nPtsCnt = int(4*np.sqrt(imBin.size)) # 4*length scale, length scale = sqrt num pixels
    # generate continuous array of points of largest contour
    x,y = Fun.generate_polygon(cntMax[:,0,0], cntMax[:,0,1], nPtsCnt)
    cntMaxZip = np.array([zip(x,y)],dtype=int)
    # create image of contour
    imCnt = np.zeros_like(imBin,dtype='uint8')
    cv2.drawContours(imCnt, [cntMaxZip], -1, 255, 1)
    
    return imCnt
    
def clean_up_bw_im(imBin, selem, minSize):
    """
    cleans up given binary image by closing gaps, filling holes, smoothing
    fringes, and removing specks
    """
    # close region in case of gaps
    closed = skimage.morphology.binary_closing(imBin, selem=selem)
    # fill holes
    filled = ndimage.morphology.binary_fill_holes(closed)
    # remove fringes
    noFringes = skimage.morphology.binary_opening(filled, selem=selem)
    # remove small objects
    imClean = skimage.morphology.remove_small_objects(noFringes, min_size=minSize)
    
    return imClean
    
def union_blue_neg_red(im, showIm, bPct, rNegPct):
    """
    Returns a binary image that is the union of the thresholded blue channel
    and the thresholded negative of the red channel of the given image. Used
    for locating blue stream in images of sheath flow.
    Threshold is done by taking upper percentile of each channel, with cutoffs 
    provided by user (bPct and rNegPct).
    """
    # extract blue channel
    imB = im[:,:,2]
    if showIm:
        show_im(imB, 'Blue channel')
    # threshold
    ret, imBThresh = cv2.threshold(imB,np.percentile(imB, bPct),255,cv2.THRESH_BINARY)
    if showIm:
        show_im(imBThresh, 'Blue channel Threshold')   
    # extract red channel
    imR = im[:,:,0]
    # get negative
    imRNeg = 255-imR
    if showIm:
        show_im(imRNeg, 'Negative of Red Channel')
    # threshold
    ret, imRNegThresh = cv2.threshold(imRNeg,np.percentile(imRNeg, rNegPct),255,cv2.THRESH_BINARY)
    if showIm:
        show_im(imRNegThresh, 'Negative of Red channel Threshold')    
    # combine negative of red channel with blue channel
    imBRNeg = np.multiply(imBThresh, imRNegThresh)
    if showIm:
        show_im(imBRNeg, 'Union of Blue and Negative Red Thresholding')
        
    return imBRNeg

###############################################################################
# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of images to consider
nIms = len(fileList)

# Loop through all videos
for i in range(nIms):
    ### EXTRACT AND SMOOTH IMAGE ###
    # Parse the filename to get image info
    imPath = fileList[i]
    # Load image
    im = plt.imread(imPath)
    imCopy = np.copy(im)
    # apply mean filter to each channel of image
    for j in range(3):
        imCopy[:,:,j] = skimage.filters.rank.mean(imCopy[:,:,j], selem)
    
    ### MASK IMAGE ###
    # user-defined mask for determining where to search
    maskData = UIF.get_rect_mask_data(imCopy, maskDataFile)
    mask = maskData['mask']
    xMin, xMax, yMin, yMax = maskData['xyMinMax']
    imInterest = imCopy[xMin:xMax,yMin:yMax,:]

    ### THRESHOLD BLUE CHANNEL AND NEGATIVE OF RED CHANNEL TO IDENTIFY STREAM ###
    imSeg = union_blue_neg_red(imInterest, showIm, bPct, rNegPct)
        
    ### CLEAN UP BINARY IMAGE ###
    imSegFilled = clean_up_bw_im(imSeg, selem,minSize)
    if showIm:
        show_im(imSegFilled,'Filled holes of Segmented image')
    
    ### TRACE CONTOUR OF LARGEST OBJECT ###
    imCntInterest = get_contour(imSegFilled, showIm)
    # place contour in full size image
    imCnt = np.zeros_like(imCopy, dtype='uint8')[:,:,0]
    imCnt[xMin:xMax,yMin:yMax] = imCntInterest
    # skip images with no contour
    if np.sum(imCnt) == 0:
        print 'no contour found in ' + imPath
        continue
    if showIm:
        show_im(imCnt,'Largest contour')
    # add thick contours
# cv2.drawContours(dilated_rgb, cnt_sorted, i, (0, 255, 0), thickness=5)
    imROutline = imCopy[:,:,1]
    imROutline[imCnt.astype(bool)] = 255
    im[:,:,1] = imROutline
    # show final result
    if showIm:
        show_im(im,'Image with outline')
        
    ### SAVE IMAGE ###
    # save image with edge overlayed
    if saveIm:
        # save image with outline overlaid for viewing
        dirTokens = imPath.split('\\')
        fileName = dirTokens[-1]
        saveName = saveFolder + fileName[:-4] + '.png'
        # save in save folder as .png rather than previous file extension
        cv2.imwrite(saveName, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print 'Saved ' + str(i+1) + ' of ' + str(nIms) + ' images.'
#        # save image of outline only
#        cv2.imwrite(saveFolder + fileName[:-4] + '_contour.png', imCnt)