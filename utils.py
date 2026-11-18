import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tcbp.reduction.dataset import get_calib_path
from scipy.interpolate import interp1d

"""
Utility functions for optical bench data analysis.
"""

def get_data_run(run, doplot=True):
    """
    Compute and plot charges ratio between sphere and receiver photodiodes for one run

    run : LSSTRun
        LSSTRun object containing the data for the run
    doplot : bool
        If True, the results are plotted
    """

    ### Get data
    wls = run["set_wl"]
    
    Q_sphere = run['pd_charge_per_burst'][wls > 0.]
    Q_receiver = run['calib_charge_per_burst'][wls > 0.]

    Q_sphere_err = run['pd_charge_per_burst_err'][wls > 0.]
    Q_receiver_err = run['calib_charge_per_burst_err'][wls > 0.]

    ratio = np.abs(Q_receiver / Q_sphere)
    ratio_err = np.abs(ratio) * np.sqrt((Q_sphere_err / Q_sphere)**2 + (Q_receiver_err / Q_receiver)**2)

    wls_valid = wls[wls > 0.]

    # Reorder the arrays by wavelength
    sort_indices = np.argsort(wls_valid)
    wls_valid = wls_valid[sort_indices]
    ratio = ratio[sort_indices]
    ratio_err = ratio_err[sort_indices]

    ### Plot
    if doplot:
        fig, (ax1,ax2,ax3) = plt.subplots(3, sharex="all", figsize=(8,8))

        ax1.errorbar(wls_valid, Q_sphere, yerr=Q_sphere_err, c='b', marker='.', ls="")
        ax1.set_ylabel('Charges Sphere [C]')
            
        ax2.errorbar(wls_valid, Q_receiver, yerr=Q_receiver_err, c='r', marker='.', ls="")
        ax2.set_ylabel('Charges Receiver [C]')

        ax3.errorbar(wls_valid, ratio, yerr=ratio_err, c='indigo', marker='.', ls="")
        ax3.set_ylabel('Ratio Receiver / Sphere [No unit]')
        ax3.set_xlabel('wavelengths [nm]')

        fig.tight_layout()
        plt.show()

    return wls_valid, ratio, ratio_err


def read_Elana_lens_measurements(file_path):
    """
    Read Elana lens measurements from a CSV file and return the data.

    Parameters:
    file_path : str
        Path to the CSV file containing the Elana lens measurements.

    Returns:
    wls, transmission, error : numpy arrays
        wls : Wavelengths (nm)
        transmission : Transmission values (unitless)
        error : Error values (unitless)
    """


    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    wls = df['Wavelength'].values
    transmission = df['lens_ratio'].values
    error = df['ratio_error'].values

    return wls, transmission, error

def load_thierry_measurements():
    """
    Load Thierry's measurements of the CBP lens transmission and return the wavelength, transmission and error arrays.
    """
    tcbp = np.load(get_calib_path("cbp_laser_response_dr5.npy"), allow_pickle=True)
    wl = tcbp['cbp_wl']
    tr = tcbp['cbp_tr']
    tr_err = tcbp['cbp_tr_err']

    return wl, tr, tr_err

def load_kelian_measurements():
    """
    Load Kelian's measurements of the CBP lens transmission and return the wavelength, transmission and error arrays.
    """
    solarcell = np.loadtxt(get_calib_path("SC_QE_NewCell_20220401_LongQECurve-2_reduced_filtered.txt"), skiprows=1, delimiter=",")
    pd_df_enhanced = pd.read_feather(get_calib_path("tCBP_v0_pd_df_fit.feather"))
    sc_df_enhanced = pd.read_feather(get_calib_path("tCBP_v0_sc_df_fit.feather"))

    sigma_A_SC = sc_df_enhanced['sigma_amplitude']
    A_SC = sc_df_enhanced['amplitude']

    sigma_A_PD = pd_df_enhanced['sigma_amplitude']
    A_PD = pd_df_enhanced['amplitude']

    wl = sc_df_enhanced['wl'] # wavelengths in nm
    T_CBP = A_SC / A_PD / np.interp(wl, solarcell[:, 0], solarcell[:, 1]) # Transmission of the TCBP

    sigma_T_CBP = T_CBP * np.sqrt((sigma_A_SC/A_SC)**2 + (sigma_A_PD/A_PD)**2) # Error on the transmission

    return wl, T_CBP, sigma_T_CBP

def interp_photodiode_response(file_path):
    """
    Interpolate the photodiode response to a given wavelength grid.

    Parameters
    ----------
    file_path : str
        Path to the file containing the photodiode response data.

    Returns
    -------
    interp_func : callable
        Interpolation function for the photodiode response.
    interp_func_err : callable
        Interpolation function for the photodiode response error.
    """
    data = np.load(file_path, allow_pickle=True)

    # Extract the wavelength, charge, and error arrays from the loaded data
    wl = data[0, :]
    Q = data[1, :]
    Q_err = data[2, :]

    # Interpolate the charges and errors
    interp_func = interp1d(wl, Q, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_err = interp1d(wl, Q_err, kind='linear', bounds_error=False, fill_value="extrapolate")

    return interp_func, interp_func_err