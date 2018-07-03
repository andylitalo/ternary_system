"""
plot_surface_tension.py produces a plot of the surface tension of the
surface between the liquid and gas phases of a van der Waals fluid in
a Cartesian coordinate system (which eliminates the effects of curvature)
near the critical temperature along the binodal curve.

It is based on the MATLAB script plot_vdW_surface_tension_fixed_midpts.m.

It produces a .png of the plot of the surface tension as a function of
difference from the critical temperature on a log-log plot with a
power-law fit to the data compared to the known power-law fit of gamma=(1-T)**1.5
It also saves the data in a .pkl file.
"""

import numpy as np
import csv


# User Parameters
nParams2Save = 8 # number of parameters to save to data file
N = 100 # number of points in grid
TMin = 0.97 # non-dimensional temperature
TMax = 0.998 # non-dimensional temperature
numT = 20 # number of temperature to consider
Cl = 1.0 # "clustering" Number
LRatio = 50.0 # ratio of interface width to length of domain (should be ~50)
# data
dataFile = 'rho_G_rho_L_T.csv'
dataFolder = '../Data/'
# save parameters
savePlots = False
savePlotsFolder = 'Figures/plot_surface_tension/'
saveData = True
saveDataFolder = 'Data/plot_surface_tension/'

def create_TList_near_Tc(TMin, TMax, numT):
    """
    Returns a numpy array of temperatures such that they are logarithmically
    spaced (linearly spaced on a log-log plot) when plotting as a function of
    1-T, where T is non-dimensionalized by the critical temperature, Tc
    """
    # logarithmic spacing relative to Tc
    logTDiff = np.linspace(np.log(1-TMax), np.log(1-TMin), numT)
    # convert to list of temperatures
    TListRev = 1 - np.exp(logTDiff)
    # reverse left to right
    TList = np.flip(TListRev, 0)
    return TList

def csv_to_matrix(csvfile, type=float):
    """
    reads data from a csv file into a numpy array
    by default converts 10-char formatted data into floats
    """
    # initialize list to hold data as it is read
    dataList = []
    # open data file and read
    with open(dataFolder + dataFile) as csvfile:
        csvdata = csv.reader(csvfile)
        # store all rows of file into a list
        for row in csvdata:
            dataList += [row]
    return np.array(dataList).astype(type)

def pressure_vdW(v, T):
    """
    Gives pressure of van der Waals fluid based on equation of state
    """
    return 8.0*np.divide(T,3.0*v-1)- 3.0*np.divide(np.ones_like(v),v**2.0)

def interface_width(T):
    """
    FILL OUT BASED ON FUNCTION IN CRITICAL RADIUS SCRIPT
    """
    iwMultiplier = 1.0/15
    return iwMultiplier*(1-T)**(-2.0/3)
# what is the best way to solve the free energy minimization problem?

###################################### DERIVED PARAMETERS ######################
# Compute state variables along binodal
# convert list of data in 10-char format into numpy array of floats
dataMatrix = csv_to_matrix(dataFolder + dataFile)
# parse columns of data matrix
rhoGData = dataMatrix[:,0]
rhoLData = dataMatrix[:,1]
TData = dataMatrix[:,2]
# create list of logarithmically spaced temperatures b/w given max and min
TList = create_TList_near_Tc(TMin, TMax, numT)
# organize data in increasing order
iSort = np.argsort(TData)
rhoGData = rhoGData[iSort]
rhoLData = rhoLData[iSort]
TData = TData[iSort]
# interpolate corresponding values of gas and liquid phase densities
rhoGList = np.interp(TList, TData, rhoGData)
rhoLList = np.interp(TList, TData, rhoLData)

########################### CALCULATE SURFACE TENSION ##########################
# initialize array to store surface tension values
gamma = np.zeros_like(TList)
# loop through each temperature and calculate surface tension
for i in range(numT):
    # set corresponding values of state variables
    T = TList[i]
    rhoG = rhoGList[i]
    rhoL = rhoLList[i]
    # define grid
    iw = interface_width(T) # width of interface
    L = LRatio * iw # length of domain
    x = np.linspace(-L/2.0, L/2.0, N+1) #x-coordinates of grid points
    delta = float(L)/N  # grid spacing
    isGas = x < 0 # designates grid points located in gas phase
    isLiquid = x > 0 # designates grid points located in liquid phase
    # initial guess for density profile (sigmoid)
    rhoGuess = rhoG + (rhoL - rhoG)*(1+np.exp(-x/iw))**(-1)
    # remove end points because we only need to fit inner points
    rhoGuess = rhoGuess[1:-1]
    # guess lagrange multiplier for condition on fixed midpoint
    lambdaGuess = 1.0

    # midpoint parameters
    rhoMid = (rhoG + rhoL)/2.0
    iMid = N/2.0
