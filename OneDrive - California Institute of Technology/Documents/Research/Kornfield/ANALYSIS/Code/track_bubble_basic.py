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

# User Parameters
folder = '..\\Videos\\' # folder containing videos
fileString = '2018_04_10_freshsoda_11000fps_1.6mLmin.mp4' # filestring of videos to analyze
fps = 11000 # fps that video was filmed at

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
#    indExposure = videoPath.find('us_')
#    exposureTime = int(videoPath[indExposure-3:indExposure])
#    print exposureTime
    # Create video "object" for manipulation
    vid = VF.get_video_object(videoPath)
    nFrames = int(vid.get(7)) # number of frames in video
    # Loop through all frames
    for f in range(nFrames):
        # Extract frame from video--frame is a matrix of pixel values
        frame = VF.extract_frame(vid, f, removeBanner=False)
        edges = cv2.Canny(frame, )
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