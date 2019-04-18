# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:37:50 2018

@author: Andy
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# User parameters
nPts = 10
plungerAreaRange = [1E-6, 1E-2] # m^2
plungerAreaFixed = 6E-4 # m^2
channelDim1Range = [300E-6, 2E-3] # m
channelDim1Fixed = 0.5E-3 # m
channelDim2Range = [500E-6, 5E-3] # m
channelDim2Fixed = 5E-3 # m
durationExperiments = 3600 # s
pAtm = 1.013E5 # atmospheric pressure, Pa
pMax = 1E7 # max pressure in channel we can reasonably expect device to achieve
u = 1 # average speed of fluid in channel, m/s
mu = 5 # viscosity, Pa.s
L = 0.1 # length of channel, m

# unit conversions
barPerPa = 1E-5

# Fixed plunger area
# Derived parameters
channelDim1List = np.exp(np.linspace(np.log(channelDim1Range[0]),
                                     np.log(channelDim1Range[1]), nPts))
channelDim2List = np.exp(np.linspace(np.log(channelDim2Range[0]),
                                     np.log(channelDim2Range[1]), nPts))
channelDim1Mat = np.tile(channelDim1List,(nPts,1))
channelDim2Mat = np.transpose(np.tile(channelDim2List,(nPts,1)))
minDimMat = np.multiply(channelDim1Mat,channelDim1Mat<=channelDim2Mat)+np.multiply(channelDim2Mat,channelDim2Mat<channelDim1Mat)
channelAreaMat = np.multiply(channelDim1Mat, channelDim2Mat)
DeltaP = 12*mu*u*L/minDimMat**2 # Pa                                   
Q = u*channelAreaMat # flow rate, m^3/s
speedSyringe = Q/plungerAreaFixed # speed of syringe, m/s
forceSyringe = (DeltaP+pAtm)*plungerAreaFixed
volumeTank = Q*durationExperiments # volume of tank, m^3
# plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = channelDim1Mat
Y = channelDim2Mat
Z1 = DeltaP*barPerPa # bar
ax.plot_surface(np.log10(X),Y,Z1)
ax.set_zlim(bottom=0, top=100)
ax.set_xlabel('channel width [m]')
ax.set_ylabel('channel height [m]')
ax.set_zlabel('pressure [bar]')

# trying out contour plot instead of 3D plot