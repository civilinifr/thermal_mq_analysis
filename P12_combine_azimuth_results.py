"""
Combines the azimuth run results of every event into a single measure of azimuth and computes mean and variance.

"""
# Import packages
import glob
import pandas as pd
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


# Functions
def plot_azi(df, evid):
    """
    Plot the azimuths for each event
    :param df:
    :param evid:
    :return:
    """
    # Convert the theta value in radians
    theta_values = df['theta'].values
    theta_rad = []
    for theta in theta_values:
        theta_rad.append(math.radians(theta))

    # Get the mean and standard deviation of the theta
    theta_mean = np.round(np.mean(theta_values), decimals=1)
    theta_var = np.round(np.var(theta_values), decimals=3)
    avg_misfit = np.round(np.mean(df['misfit_fin'].values), decimals=3)

    # Plot
    fig = plt.figure(figsize=(10, 10), num=2, clear=True)

    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    for evid_ind in np.arange(len(theta_rad)):
        ax.plot([0, theta_rad[evid_ind]], [0, 1])
    ax.set_title(f'Evid : {evid}, mean theta={str(theta_mean)}, variance={str(theta_var)}, '
                 f'misfit={str(avg_misfit)}', fontweight='bold')
    fig.savefig(f'{output_dir}images/{evid}_combined_results.png')

    # Create a new dataframe
    data = [[theta_mean, theta_var, avg_misfit]]
    df_new = pd.DataFrame(data, columns=['theta_mean', 'theta_variance', 'avg_misfit'])
    df_new.to_csv(f'{output_dir}data/{evid}_combined_results.csv', index=False)

    print(f'Finished processing event {evid}!')
    return


def process_azimuth_wrapper(azi_file):

    file_bn = os.path.basename(azi_file)
    evid = file_bn.split('_')[2].split('.')[0]

    if os.path.exists(f'{output_dir}data/{evid}_combined_results.csv'):
        print(f'Evid {evid} already processed. Skipping...')
        return

    df = pd.read_csv(azi_file)

    plot_azi(df, evid)
    return


# Main
# Set the number of cores
num_cores = 10

# Set up the directories
out_dir = 'C:/data/lunar_output/'
input_directory = f'{out_dir}results/locations/'
output_dir = f'{input_directory}azimuth_combined/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(f'{output_dir}images'):
    os.mkdir(f'{output_dir}images')
if not os.path.exists(f'{output_dir}data'):
    os.mkdir(f'{output_dir}data')

filelist = glob.glob(f'{input_directory}azimuth/*/azimuth_results*.csv')

if num_cores == 1:
    # Single-core version
    for infile in filelist:
        process_azimuth_wrapper(infile)
else:
    # Multi-core version
    Parallel(n_jobs=num_cores)(delayed(process_azimuth_wrapper)(infile)
                               for infile in filelist)
