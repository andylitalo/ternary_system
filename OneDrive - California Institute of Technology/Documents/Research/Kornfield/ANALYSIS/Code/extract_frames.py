# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:44:10 2018

@author: Andy
"""

# import packages
import glob
import os
import VideoFunctions as VF
import matplotlib.pyplot as plt
import numpy as np
import Functions as Fun
import cv2
import skimage.morphology
import skimage.filters
import skimage.feature
from scipy import ndimage

# User Parameters
# data for video
folder = '..\\Videos\\' # folder containing videos
fileString = '8819fps_108us_16mLmin.mp4' # filestring of videos to analyze
fps = 8819 # set to 0 if unknown and program will obtain fps from video object
# analysis parameters
startFrame = 1
endFrame = 100
subtractRef = True
meanFilter = True
kernel = np.ones((5,5),np.float32)/25 # kernel for gaussian filter
thresh = 5
# Structuring element is radius 2 disk
selem = skimage.morphology.disk(10)
nDilations = 0
showIms = False
minSize = 100
# saving parameters
saveIms = True
saveFolder = '..\\Videos\\8819fps_108us_16mLmin_video\\'


# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of videos to consider
nVideos = len(fileList)
print nVideos

# Loop through all videos
for v in range(nVideos):
    # Parse the filename to get video info
    videoPath = fileList[v]
    indExposure = videoPath.find('us_')
    if indExposure == -1:
        exposureTime = 1000/fps # approximate exposure time as 1s/fps
    else:
        exposureTime = int(videoPath[indExposure-3:indExposure])
    # Create video "object" for manipulation
    vid = VF.get_video_object(videoPath)
    nFrames = int(vid.get(7)) # number of frames in video
    # save mean intensity of each frame
    meanIntensity = np.zeros([nFrames,nVideos])
    # Loop through all frames
    lastFrame = np.min([nFrames, endFrame])
    if startFrame >= lastFrame:
        print 'start frame is larger than last frame.'
        continue
    # save a reference frame for image subtraction
    vid.set(1,0)
    ret, refFrame = vid.read()
    plt.figure()
    plt.imshow(refFrame)
    # smooth reference frame to reduce noise
    if meanFilter:
        for i in range(3):
            refFrame[:,:,i] = skimage.filters.rank.mean(refFrame[:,:,i], selem)
    # loop through all frames
    for f in range(startFrame, lastFrame):
        # Extract frame from video--frame is a matrix of pixel values
        vid.set(1,f)
        ret, frame = vid.read()
        if showIms:
            plt.figure()
            plt.imshow(frame)
        if meanFilter:
            for i in range(3):
                frame[:,:,i] = skimage.filters.rank.mean(frame[:,:,i], selem)
        if subtractRef:
            frame1 = frame.astype(int)
            refFrame1 = refFrame.astype(int)
            frame1 = refFrame1 - frame1
            frame1[frame1<0] = 0
            frameSubtracted = frame1.astype('uint8')
            frameG = frameSubtracted[:,:,1]
        # threshold
        ret, bubble = cv2.threshold(frameG,thresh,255,cv2.THRESH_BINARY)
        bubble = bubble.astype(bool)
        bubbleClosed = skimage.morphology.binary_closing(bubble,selem=selem)
        for i in range(nDilations):
            bubble = bubbleClosed
            bubbleClosed = skimage.morphology.binary_closing(bubble,selem=selem)
        # fill holes
        bubbleFilled = ndimage.morphology.binary_fill_holes(bubbleClosed)
        # open
        bubbleFiled = skimage.morphology.binary_opening(bubbleFilled, selem=selem)
        # remove small objects
        bubbleFilled = skimage.morphology.remove_small_objects(bubbleFilled, min_size=minSize)
        bubbleEdge = skimage.feature.canny(bubbleFilled.astype(float), 1.4)
        # overlay edge with image
        frameR = frame[:,:,0]
        frameR[bubbleEdge] = 255
        frame[:,:,0] = frameR
        bubbleFilled= bubbleFilled.astype('uint8')
        bubbleFilled *= 255
        if showIms:
            plt.figure()
            plt.imshow(bubbleFilled)
        if saveIms:
            cv2.imwrite(saveFolder + fileString[:-4] + '_' + str(f) + '.png', frame)
#        # compute mean intensity
#        meanIntensity[f] = np.mean(np.mean(frameSubtracted))
        ################## DISPLAY FRAME #####################
#        # Open new figure for new video; same for new frame
#        plt.figure(v+1)
#        # Clear previous frame
#        plt.cla()
#        # converts black-and-white image to RGB (still shows up black and white)
#        frame = np.dstack((frame,frame,frame))
#        # Display frame
#        Fun.plt_show_image(frame)
#        # Pause before showing next frame
#        plt.pause(.001)

    time = 1000*np.linspace(0,float(nFrames-1)/fps,nFrames)
    plt.plot(time, meanIntensity, label=str(exposureTime) + ' us')
    
plt.xlabel('time (ms)')
plt.ylabel('mean intensity (a.u.)')
plt.title('Mean intensity over time at ' + str(fps) + ' fps')