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

# User Parameters
folder = '..\\Videos\\' # folder containing videos
fileString = '*.mp4' # filestring of videos to analyze
fps = 2111
viewWidth = 50 # number of pixels width of viewing window

# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of videos to consider
nVideos = len(fileList)

# Loop through all videos
for v in range(nVideos):
    # Parse the filename to get video info
    videoPath = fileList[v]
    indExposure = videoPath.find('us_')
    exposureTime = int(videoPath[indExposure-3:indExposure])
    # Create video "object" for manipulation
    vid = VF.get_video_object(videoPath)
    nFrames = int(vid.get(7)) # number of frames in video
    # save mean intensity of each frame
    frameMeanIntensity = np.zeros(100)
    # Loop through all frames
    for f in range(100):
        # Extract frame from video--frame is a matrix of pixel values
        frame = VF.extract_frame(vid, f, removeBanner=False)
        # dimensions of frame (pixels)
        frameHeight = len(frame[:,0])
        frameWidth = len(frame[0,:])
        # compute mean intensity
        midWidth = frameWidth/2
        frameMeanIntensity[f] = np.mean(np.mean(frame[:,midWidth:midWidth+viewWidth]))

    time = np.linspace(0,100-1,100)/fps*1000 # ms
    plt.plot(time, frameMeanIntensity)    
    plt.xlabel('time (ms)',fontsize=16)
    plt.ylabel('mean intensity (a.u.)',fontsize=16)
    plt.title('Mean intensity over Time at ' + str(fps) + ' fps',fontsize=18)