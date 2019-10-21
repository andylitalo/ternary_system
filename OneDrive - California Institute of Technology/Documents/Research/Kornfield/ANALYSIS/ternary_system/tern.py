"""
tern is a library of methods used for analyzing data from GC sampling of ternary
systems composed of cyclopentane, polyol, and CO2.

Author - Andy Ylitalo
Created 16:20, 10/2/19, CDT
"""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d



def get_calib_data(signal, compound, quantity):
    """Returns calibration data of co2 and c5 in GC."""
    # dictionary to convert code words to column names in dataframe
    col_dict = {'rho':'density [g/mL]', 'w':'weight fraction [w/w]'}
    # load data in pandas dataframe
    df = pd.read_csv('calib_' + signal + '_' + compound + '.csv')
    pa = df['peak area [a.u.]'].to_numpy(dtype=float)
    value = df[col_dict[quantity]].to_numpy(dtype=float)

    return pa, value


def pa_conv(pa, signal, compound, quantity='rho', average=True):
    """
    Loads data from calibration curves generated for the 7890 Agilent gas
    chromatograph at Dow Chemical Co., Lake Jackson, TX, and converts to a mass
    fraction or density.
    PARAMETERS:
        pa : float or numpy array
            GC peak area [a.u.] desired to be converted into a physical quantity.
        signal : string
            GC signal measured. Either 'f' (HPLIS dense-phase sampling) or
            'b' (GC light-phase sampling).
        compound : string
            Compound detected in GC. Either 'co2' (carbon dioxide) or 'c5' (cyclopentane).
        quantity : string, default='rho'
            Quantity to conver peak area to. Either 'rho' for density [g/mL] or
            'w' for weight fraction [w/w].
        average : bool, default=True
            If True, the peak areas provided will be averaged.
    RETURNS:
        result : float or numpy array
            Physical quantity requested with "quantity" parameter, same type as
            the input "pa" parameter unless averaged (in which case the
            result will be a float).
    """
    assert signal in ['f', 'b'], "Invalid signal. Choose ''f'' or ''b''."
    assert compound in ['co2', 'c5'], "Invalid compound. Choose ''co2'' or ''c5''."
    x, y = get_calib_data(signal, compound, quantity)
    # perform linear fit to get calibration conversion
    a, b = np.polyfit(x, y, 1)
    # estimate quantity for given peak area
    result = a*pa + b
    # average results
    if average:
        result = np.mean(result)

    return result


def rho_co2(p, T, eos_file_hdr='eos_co2_', ext='.csv', psi=False):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at 30.5 C.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.
    PARAMETERS:
        p : int (or array of ints)
            pressure in kPa of CO2 (unless psi==True)
        T : float
            temperature in Celsius (only to one decimal place)
        eos_file_hdr : string, default='eos_co2_'
            File header for equation of state data table
        ext : string, default='.csv'
            Extension for file, including period (4 characters).
        psi : bool, default=False
            If True, treats input pressure as psi.
    RETURNS:
        rho : same as p
            density in g/mL of co2 @ 30.5 C
    """
    # convert to kPa if pressure is passed as psi
    if psi:
        p *= 100/14.5
    # get decimal and integer parts of the temperature
    dec, integ = np.modf(T)
    # create identifier string for temperature
    T_tag = '%d-%dC' % (integ, 10*dec)
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(eos_file_hdr + T_tag + ext, header=0)
    # get list of pressures of all data points [kPa]
    p_co2_kpa = df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [g/mL]
    rho_co2 = df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # remove repeated entries
    p_co2_kpa, inds_uniq = np.unique(p_co2_kpa, return_index=True)
    rho_co2 = rho_co2[inds_uniq]
    # create interpolation function and interpolate density [g/mL]
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")
    rho = f_rho(p)

    return rho


def rho_v2110b(T):
    """
    Estimates density of VORANOL 2110B polyol at given temperature based on
    data collected at Dow for P-1000 (also 1000 g/mol, difunctional polyol).
    PARAMETERS:
        T : float
            Temperature [C].
    RETURNS:
        rho : float
            Density [g/mL]
    """
    T_data = np.array([25, 50, 75, 100])
    rho_data = np.array([1.0015, 0.9826, 0.9631, 0.9438])
    # estimate density with linear regression
    a, b = np.polyfit(T_data, rho_data, 1)
    rho = a*T + b

    return rho


def rho_liq(rho_c, T_c, A, B, C, D, T):
    """
    Empirical equation for the density of a liquid from equation 1 of Chapter D3,
    "Properties of Pure Fluid Substances," by Michael Kleiber and Ralph Joh,
    Section 1, "Liquids and Gases," of the VDI Heat Atlas (2010).
    PARAMETERS:
        rho_c : float
            Critical density [kg/m^3].
        T_c : float
            Critical temperature [K].
        A, B, C, D : floats
            fitting parameters of equation
        T : float or (N x 1) numpy array
            Desired temperature(s) [K].
    RETURNS:
        rho_liq : float or (N x 1) numpy array (same as T)
            Density of desired temperature(s) [kg/m^3].
    """
    rho_liq = rho_c + A*(1-T/T_c)**(0.35) + B*(1-T/T_c)**(2/3) + C*(1-T/T_c) + \
                D*(1-T/T_c)**(4/3)
    return rho_liq


def rho_c5(T, empirical_formula=False):
    """
    Estimates density of cyclopentane below critical temperature with empirical
    equation for liquid-phase densities in Chapter D3, "Properties of Pure Fluid
    Substances," by Michael Kleiber and Ralph Joh, Section 1, "Liquids and Gases,"
    of the VDI Heat Atlas (2010). The equation is of the form:

    rho_liq = rho_c + A*(1-T/T_c)**(0.35) + B*(1-T/T_c)**(2/3) + C*(1-T/T_c) +
                D*(1-T/T_c)**(4/3)

    SOURCE:
    Michael Kleiber, Ralph Joh, Roland Span (2010)
    SpringerMaterials
    D3 Properties of Pure Fluid Substances
    VDI-Buch
    (VDI Heat Atlas)
    https://materials-springer-com.clsproxy.library.caltech.edu/lb/docs/sm_nlb_978-3-540-77877-6_18
    10.1007/978-3-540-77877-6_18 (Springer-Verlag © 2010)
    Accessed: 04-10-2019

    PARAMETERS:
        T : float or (N x 1) numpy array
            Temperature(s) [C].
    RETURNS:
        rho_c5 : float or (N x 1) numpy array (same as T)
            Density of cyclopentane at desired temperature(s) [g/mL].
    """
    if empirical_formula:
        A, B, C, D = [450.0150, 692.7447, -1131.7809, 724.4630]
        rho_c = 270 # [kg/m^3]
        T_c = 511.7 # [K]
        rho_c5 = rho_liq(rho_c, T_c, A, B, C, D, T+273.15)/1000 # [g/mL]
    else:
        T_data = np.array([0, 20, 50, 100]) # [C]
        rho_data = np.array([0.7652, 0.7465, 0.7176, 0.6652]) #[g/mL]
        # a, b = np.polyfit(T_data, rho_data, 1)
        # rho_c5 = a*T + b # [g/mL]
        rho_c5_f = interp1d(T_data, rho_data)
        rho_c5 = rho_c5_f(T)

    return rho_c5


def rho_v_co2_c5(p, T, w_v_co2, w_v_c5, filename='co2_c5_vle.csv', thresh=0.4):
    """
    Estimates density of co2-c5 binary vapor mixture by using a PC-SAFT
    model fitted to experimental measurements of the CO2-C5 VLE from
    Eckert and Sandler (1986) to estimate the expected total density of the
    vapor phase of the mixture.
    """
    # load dataframe of co2-c5 VLE
    df = pd.read_csv(filename)
    # select temperature
    T_arr = df['temperature [C]'].to_numpy(dtype=float)
    # remove duplicate temperatures
    T_uniq = np.unique(T_arr)
    # choose the temperature in the array that is closest to the current temperature
    T_nearest = T_uniq[int(np.argmin(np.abs(T_uniq-T)))]
    # extract entries for selected temperature
    inds = np.where(T_arr==T_nearest)[0]
    p_arr = df['pressure [psi]'].to_numpy(dtype=float)[inds]
    w_v_co2_arr = df['w co2 vap [w/w]'].to_numpy(dtype=float)[inds]
    w_v_c5_arr = df['w c5 vap [w/w]'].to_numpy(dtype=float)[inds]
    rho_v_co2_arr = df['density co2 vap [g/mL]'].to_numpy(dtype=float)[inds]
    rho_v_c5_arr = df['density c5 vap [g/mL]'].to_numpy(dtype=float)[inds]
    # interpolate weight fractions and densities
    w_v_co2 = np.interp(p, p_arr, w_v_co2_arr)
    w_v_c5 = np.interp(p, p_arr, w_v_c5_arr)
    rho_v_co2 = np.interp(p, p_arr, rho_v_co2_arr)
    rho_v_c5 = np.interp(p, p_arr, rho_v_c5_arr)
    # ensure that the weight fractions match what the GC measured
    assert np.max(np.abs(w_v_co2-w_v_co2)) < thresh and \
            np.max(np.abs(w_v_c5-w_v_c5)) < thresh, "weight fractions too different." + \
            "{0:.3f} vs. {1:.3f} for {2:d} psi and {3:d} C.".format(w_v_co2, w_v_co2, p, T)

    return rho_v_co2, rho_v_c5


def rho_polyol_co2(p, T):
    """Estimates density of polyol-CO2 mixture based on Naples data."""
    T_list = [30.5, 60]
    df_30 = pd.read_csv('1k2f_30c.csv')
    df_60 = pd.read_csv('1k2f_60c.csv')
    p_arr_list = [df_30['p actual [kPa]'].to_numpy(dtype=float),
                    df_60['p actual [kPa]'].to_numpy(dtype=float)]
    spec_vol_arr_list = [df_30['specific volume [mL/g]'].to_numpy(dtype=float),
                    df_60['specific volume [mL/g]'].to_numpy(dtype=float)]
    spec_vol_list = [np.interp(p, np.array(p_arr_list[i]),
                    np.array(spec_vol_arr_list[i])) for i in range(len(p_arr_list))]
    spec_vol = np.interp(T, np.array(T_list), np.array(spec_vol_list))
    rho = 1/spec_vol

    return rho
