import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    ### Plot
    if doplot:
        fig, (ax1,ax2,ax3) = plt.subplots(3, sharex="all", figsize=(8,8))

        ax1.errorbar(wls[wls > 0.], Q_sphere, yerr=Q_sphere_err, c='b', marker='.', ls="")
        ax1.set_ylabel('Charges Sphere [C]')
            
        ax2.errorbar(wls[wls > 0.], Q_receiver, yerr=Q_receiver_err, c='r', marker='.', ls="")
        ax2.set_ylabel('Charges Receiver [C]')

        ax3.errorbar(wls[wls > 0.], ratio, yerr=ratio_err, c='indigo', marker='.', ls="")
        ax3.set_ylabel('Ratio Receiver / Sphere [No unit]')
        ax3.set_xlabel('wavelengths [nm]')

        fig.tight_layout()
        plt.show()

    return ratio, ratio_err


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