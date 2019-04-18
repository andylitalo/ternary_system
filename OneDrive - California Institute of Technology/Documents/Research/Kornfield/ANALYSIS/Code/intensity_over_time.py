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
fileString = '2111fps_469us_00mLmin.mp4' # filestring of videos to analyze
fps = 2111 # Hz
light_freq = 999.508 # Hz
shift = 0.9
t_i = 0 # ms
t_f = 50 # ms

# Cleanup existing windows
plt.close('all')
   
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
# Number of videos to consider
nVideos = len(fileList)

# We prepare the plot  
fig = plt.figure(1)
# We define a fake subplot that is in fact only the plot.  
plot = fig.add_subplot(111)
# We change the fontsize of major ticks label 
plot.tick_params(axis='both', which='major', labelsize=20)

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
    meanIntensity = np.zeros([nFrames,nVideos])
    # Loop through all frames
    for f in range(nFrames):
        # Extract frame from video--frame is a matrix of pixel values
        frame = VF.extract_frame(vid, f, removeBanner=False)
        # compute mean intensity
        meanIntensity[f] = np.mean(np.mean(frame))

    # create time vector in ms
    time = 1000*np.linspace(0,float(nFrames-1)/fps,nFrames)
    # plot mean intensity over time of signal
    plt.plot(time, meanIntensity, label='actual signal')
    # replicate signal artificially
    baseline = np.mean(meanIntensity)
    amplitude = np.max(meanIntensity)-baseline    
    # make plot of artifical signal
    light_intensity = baseline + amplitude*np.cos(2*np.pi*light_freq*time+shift)
    plt.plot(time, light_intensity, label='artifical signal, LED=' + str(light_freq) + ' Hz')
      
plt.xlabel('time (ms)',fontsize=20)
plt.xlim(t_i, t_f)
plt.ylabel('mean intensity (a.u.)',fontsize=20)
plt.title('Mean intensity over time at ' + str(fps) + ' fps',fontsize=20)
plt.legend(loc=1)

### decompose signal with FFT