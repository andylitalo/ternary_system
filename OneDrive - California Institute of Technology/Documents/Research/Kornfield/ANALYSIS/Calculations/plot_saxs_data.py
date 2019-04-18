# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:19:52 2018

@author: Andy
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os


def get_mu(fileName):
    """
    Returns the value for the parameter mu from the file name, which includes
    1000*exp(mu).
    """
    hdr = '_mu_'
    iMuStart = fileName.find(hdr) + len(hdr)
    iMuEnd = fileName.find('_sigma_')
    mu = np.log(float(fileName[iMuStart:iMuEnd])/1000.)
    
    return mu

def get_sigma(fileName):
    """
    Returns the value for the parameter sigma from the file name, which includes
    1000*sigma.
    """
    hdr = '_sigma_'
    iSigmaStart = fileName.find(hdr) + len(hdr)
    iSigmaEnd = fileName.find('.csv')
    sigma = float(fileName[iSigmaStart:iSigmaEnd])/1000.
    
    return sigma

    
# User parameters
fileString = 'saxs_data_mu_*_sigma_*.csv'
folder = 'Data/'
muList = [np.log(0.5)]
sigmaList = [0.1]
ytickVals = 10.**np.arange(-12,4,4)
ytickLabels = ["$10^{-12}$", "$10^{-8}$","$10^{-4}$", "$10^0$"]
wavenumberMax = 5E-4 # max wavenumber
# Plot parameters
axLabFS = 20
tkFS = 16
saveFolder = "Figures/"
savePlots = False

plt.close('all')
# Get file path
pathToFiles = os.path.join(folder,fileString)
# Create list of files to analyze
fileList = glob.glob(pathToFiles)
nFiles = len(fileList)

for i in range(nFiles):
    currFileName = fileList[i]
    mu = get_mu(currFileName)
    sigma = get_sigma(currFileName)
    if mu in muList and sigma in sigmaList:
        # first column is s (wavenumber), second is intensity
        dataFile = np.genfromtxt(currFileName, delimiter=',')
        wavenumber = dataFile[:,0]
        intensity = dataFile[:,1]
        # convert wavenumber to 1/A as is convention
        wavenumber = wavenumber/10000 # 1/um to 1/A
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.loglog(wavenumber, intensity, 'b.')
        plt.xlabel("Wavenumber [1/$\AA$]", fontsize=axLabFS)
        plt.ylabel('Intensity [a.u.]', fontsize=axLabFS)
        plt.title('Scattering Intensity for mu = ' + str(round(mu,2)) + ', sigma = ' + str(round(sigma, 3)))
        ticks = []
        plt.yticks(ytickVals,ytickLabels)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tkFS) 
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(tkFS)
        # reshape figure boundaries to fit around resized axis and tick labels
        plt.tight_layout()
        if savePlots:
            saveName = "saxs_data_mu_" + str(int(np.exp(mu)*1000)) + "_sigma_" + str(int(sigma*1000)) + ".pdf"
            plt.savefig(saveFolder + saveName, format='pdf')
        
        # Kratky plot (ln(I) vs. s^2)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        plt.plot(wavenumber**2, np.log(intensity), 'b.', label='data')
        plt.xlim((0,wavenumberMax**2))
        plt.ylim((-2,-1))
        # fit
        inds = wavenumber < wavenumberMax
        x = wavenumber[inds]**2
        y = np.log(intensity[inds])
        m, b = np.polyfit(x, y, 1)
        Rg = np.sqrt(-3*m)/10 # Rg in nm
        plt.plot(x, m*x+b, 'r--',label='m = ' + '%.1E' % (m))
        plt.xlabel("$q^2$ [1/$\AA^2$]", fontsize=axLabFS)
        plt.ylabel('log(Intensity) [a.u.]', fontsize=axLabFS)
        plt.title('Rg = %d' % (Rg) +' nm, mu = ' + str(round(mu,2)) + ', sigma = ' + str(round(sigma, 3)))
        plt.legend(loc=1)
#        ticks = []
#        plt.yticks(ytickVals,ytickLabels)
#        for tick in ax.yaxis.get_major_ticks():
#            tick.label.set_fontsize(tkFS) 
#        for tick in ax.xaxis.get_major_ticks():
#            tick.label.set_fontsize(tkFS)
        # reshape figure boundaries to fit around resized axis and tick labels
        plt.tight_layout()
        if savePlots:
            saveName = "kratky_mu_" + str(int(np.exp(mu)*1000)) + "_sigma_" + str(int(sigma*1000)) + ".pdf"
            plt.savefig(saveFolder + saveName, format='pdf')
                   