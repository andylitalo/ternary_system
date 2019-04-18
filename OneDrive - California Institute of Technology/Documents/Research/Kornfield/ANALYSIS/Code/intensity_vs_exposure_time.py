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
folder = '..\\High-speed Camera\\exposure_time\\' # folder containing videos
fileString = '*.mp4' # filestring of videos to analyze
fps = 2111
maxFrames = 200 # maximum number of frames to analyze from each video

# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of videos to consider
nVideos = len(fileList)

timeMeanIntensity = np.zeros(nVideos)
exposureTimeList = np.zeros(nVideos)

# Loop through all videos
for v in range(nVideos):
    # Parse the filename to get video info
    videoPath = fileList[v]
    indExposure = videoPath.find('us_')
    exposureTime = int(videoPath[indExposure-3:indExposure])
    exposureTimeList[v] = exposureTime
    # Create video "object" for manipulation
    vid = VF.get_video_object(videoPath)
    nFrames = int(vid.get(7)) # number of frames in video
    nFramesToAnalyze = max(nFrames,maxFrames)
    # save mean intensity of each frame
    frameMeanIntensity = np.zeros(nFramesToAnalyze)
    # Loop through all frames
    for f in range(nFramesToAnalyze):
        # Extract frame from video--frame is a matrix of pixel values
        frame = VF.extract_frame(vid, f, removeBanner=False)
        # compute mean intensity
        frameMeanIntensity[f] = np.mean(np.mean(frame))
        
    timeMeanIntensity[v] = np.mean(frameMeanIntensity)
    
# We prepare the plot  
fig = plt.figure(1)
# We define a fake subplot that is in fact only the plot.  
plot = fig.add_subplot(111)
# We change the fontsize of major ticks label 
plot.tick_params(axis='both', which='major', labelsize=20)
# make the plot
plt.plot(exposureTimeList, timeMeanIntensity)    
# create axis labels
plt.xlabel('exposure time (us)',fontsize=20)
plt.ylabel('mean intensity (a.u.)',fontsize=20)
# add title
plt.title('Mean intensity vs. Exposure Time at ' + str(fps) + ' fps',fontsize=20)