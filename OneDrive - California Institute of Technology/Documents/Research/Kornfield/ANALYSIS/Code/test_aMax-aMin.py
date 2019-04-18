# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:45:07 2018

@author: Andy
"""

import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

# User parameters
folder = 'Data\\'
epFile = 'ep.pkl'


# Load ep file
with open(folder + epFile, 'rb') as f:
    allEPData = pkl.load(f)
    
keys = allEPData.keys()

for key in keys:
    ePData = allEPData[keys]
    aMax = ePData['aMax95']
    aMin = ePData['aMin2']
    time = ePData['time']
    t0 = ePData['t0']
    time -= t0
    plt.figure()
    plt.plot(time, aMax-aMin)