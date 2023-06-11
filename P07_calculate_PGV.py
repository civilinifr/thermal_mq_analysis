"""
Use the sac data created from the P6 code to find the Peak Ground Velocity (PGV) for each Grade A event.

"""
# Import packages
import glob
import pandas as pd
import datetime as dt
import numpy as np
from obspy.core import read
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import os


# Functions
def find_nearest(array, value):
    """
    Find nearest index corresponding to a value in an array
    :param array: [np array] Array that we are searching
    :param value: [float] Value that we're searching for
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_pgv(df, data_folder):
    """
    Function to find the PGV
    :param df: [pd df] Dataframe of the event stats
    :param data_folder: [str] Folder path of the event day of sac data

    :return: A vector of PGV values
    """
    geophones = df['geophone'].values
    evid = df['evid'].values[0]
    PGV_vector = []

    # Cycle through all the geophones
    for geophone_ind in np.arange(len(geophones)):
        arrival_time_str = df['ft_arrival_time'].values[geophone_ind]
        arrival_time = dt.datetime.strptime(arrival_time_str, "%Y-%m-%dT%H:%M:%S.%f")
        arrival_time_hour = arrival_time.hour
        arrival_time_hour_str = "{:02d}".format(arrival_time.hour)
        geophone = df['geophone'].values[geophone_ind][1:]

        # Find the corresponding sac file
        if len(glob.glob(f'{data_folder}*{geophone}_{arrival_time_hour_str}_ID')) == 0:
            print(f'Missing arrival file for G{geophone} {arrival_time_hour}! Skipping...')
            if evid == '770425-00-M1':
                PGV_vector = [0, 0, 0, 0]
                return PGV_vector
            return
        corr_file = glob.glob(f'{data_folder}*{geophone}_{arrival_time_hour_str}_ID')[0]

        # Load the file
        st = read(corr_file)
        tr = st[0]
        trace_time = tr.times()
        trace_data = tr.data

        # Find the absolute time corresponding to the start of the hourly trace
        # Then find the relative arrival time, relative end time of the trace
        start_time = dt.datetime(tr.stats.starttime.year, tr.stats.starttime.month, tr.stats.starttime.day,
                                 tr.stats.starttime.hour)
        arrival_rel_diff = arrival_time - start_time
        arrival_rel_time = arrival_rel_diff.seconds + arrival_rel_diff.microseconds/10**6
        envelope_rel_end = arrival_rel_time + df['envelope_length'].values[geophone_ind]

        # Find the PGV, which is the maximum of the data between the arrival start time and the end envelope
        arrival_ind = find_nearest(trace_time, arrival_rel_time)
        envelope_end_ind = find_nearest(trace_time, envelope_rel_end)
        if envelope_end_ind - arrival_ind <= 1:
            PGV_val = np.nanmax(trace_data[arrival_ind:arrival_ind+int(tr.stats.sampling_rate*60)])
            if len(trace_data[arrival_ind:arrival_ind+int(tr.stats.sampling_rate*60)]) == 1:
                PGV_val = 0
        else:
            PGV_val = np.nanmax(trace_data[arrival_ind:envelope_end_ind])

        PGV_ind = np.where(trace_data == PGV_val)[0]
        PGV_vector.append(PGV_val)

        # Plot an image of the PGV
        plot_minwin = 180
        plot_maxwin = 360
        if arrival_rel_time-plot_minwin < 0:
            plot_minwin_ind = 0
        else:
            plot_minwin_ind = find_nearest(trace_time, arrival_rel_time-plot_minwin)
        if arrival_rel_time+plot_maxwin > trace_time[-1]:
            plot_maxwin_ind = len(trace_time)
        else:
            plot_maxwin_ind = find_nearest(trace_time, arrival_rel_time + plot_maxwin)

        plt.figure(figsize=(8, 4), num=1, clear=True)
        plt.plot(trace_time[plot_minwin_ind:plot_maxwin_ind], trace_data[plot_minwin_ind:plot_maxwin_ind])
        plt.xlim((trace_time[plot_minwin_ind:plot_maxwin_ind][0], trace_time[plot_minwin_ind:plot_maxwin_ind][-1]))
        plt.axvline(x=arrival_rel_time, c='r')
        plt.axvline(x=envelope_rel_end, c='g')
        plt.axhline(y=PGV_val, c='magenta')
        plt.title(f'{evid} {geophones[geophone_ind]}: PGV = {str(np.round(PGV_val, decimals=1))} nm/s')
        plt.savefig(f'{image_output}{evid}_{geophones[geophone_ind]}_PGV.png')
        plt.savefig(f'{image_output}eps_{evid}_{geophones[geophone_ind]}_PGV.eps')

    return PGV_vector


def find_pgv_wrapper(infile):
    """
    Wrapper code for calculating PGV for each event

    :param infile: [str] Path of the stats file
    :return:
    """
    # Load the file
    df = pd.read_csv(infile)
    evid = df['evid'].values[0]

    # If this dataframe already has a PGV column, we have already processed this, so skip it.
    if len(np.where(df.columns.values == 'PGV')[0]) > 0:
        if remove_column == 'yes':
            df = df.drop(columns='PGV')
        else:
            print(f'Already computed PGV for evid {evid}! Skipping...')
            return

    # Get the fine-tuned arrival time for each geophone
    arrival_vector_str = df['ft_arrival_time'].values
    arrival_vector_dt = []
    for arrival_str in arrival_vector_str:
        arrival_vector_dt.append(dt.datetime.strptime(arrival_str, "%Y-%m-%dT%H:%M:%S.%f"))

    # Find the corresponding sac data file
    arrival_date = dt.datetime.strftime(arrival_vector_dt[0], "%Y%m%d")
    data_folder = f'{data_directory}{arrival_date}/'

    # Find the PGV for each geophone
    PGV_vector = find_pgv(df, data_folder)

    # Add the PGV vector to the dataframe
    # Save the new dataframe
    df.insert(11, "PGV", PGV_vector, True)
    df.to_csv(f'{cat_stats_directory}GradeA_evid_{evid}_stats.csv', index=False)

    print(f'Computed PGV for evid {evid}!')

    return


# Main
# Setup folders
data_dir = 'C:/data/lunar_data/'
out_dir = 'C:/data/lunar_output/'
data_directory = f'{data_dir}LSPE_sac_hourly_phys/sac_data/'
cat_stats_directory = f'{out_dir}fine-tuned/cat_stats/'

# Select the number of CPU cores to use
num_cores = 1

image_output = f'{out_dir}PGV_plots/'
if not os.path.exists(image_output):
    os.mkdir(image_output)

# Leftover legacy code, removes any PGV column in the data.
# Keep 'yes' although it's probably not necessary
remove_column = 'yes'

# Get the file list
stats_files = sorted(glob.glob(f'{cat_stats_directory}*.csv'))

# Uncomment below if you just want to check particular evids.
# stats_files = sorted(glob.glob(f'{cat_stats_directory}*_760815-16-M3_stats.csv'))

if num_cores == 1:
    for stats_file in stats_files:
        find_pgv_wrapper(stats_file)
else:
    Parallel(n_jobs=num_cores)(delayed(find_pgv_wrapper)(stats_file)
                               for stats_file in stats_files)
