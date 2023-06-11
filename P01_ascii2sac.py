"""
Converts daily ASCII data to hourly SAC data in decompressed volts.
"""

# Import packages
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import os
from datetime import datetime, timedelta
from obspy.io.sac.sactrace import SACTrace
import time
from joblib import Parallel, delayed


# Setup Functions
def interp_data(in_time, in_data, out_time):
    """
    Interpolates the input data to a new target samplerate
    :param in_time: [Vector] Input time
    :param in_data: [Vector] Input data
    :param out_time: [Vector] New target samplerate
    :return: [Vector] Interpolated data
    """

    # Remove the mean and cosine taper (due to the large amount of data, only do 0.05% taper on each side)
    trace_nomean = in_data - np.mean(in_data)
    n = len(trace_nomean)
    taper_function = cosine_taper(n, p=0.001)
    trace_taper = trace_nomean * taper_function

    # Interpolate the data
    f = interp1d(in_time, trace_taper)
    out_data_interp = f(out_time)

    return out_data_interp


def running_median(seq, win):
    """
    Computes a running median on the input data
    :param seq: [Vector] Input data to find the running median
    :param win: [Integer] Window size in samples
    :return: [Vector] The running median
    """

    medians = []
    window_middle = int(np.ceil(win/2))

    for ind in np.arange(len(seq)):

        if ind <= window_middle:
            medians.append(np.median(abs(seq[0:win])))

        if ind >= len(seq)-window_middle:
            medians.append(np.median(abs(seq[len(seq)-win:len(seq)])))

        if window_middle < ind < len(seq)-window_middle:
            medians.append(np.median(abs(seq[ind-int(np.floor(win/2)):ind+int(np.floor(win/2))])))

    return np.array(medians)


def despike(input_d, med_multiplier=5.):
    """
    Despikes the input data according to Renee's 2005 paper (Median despiker).
    Note: This routine does NOT do a bandpass.

    :param input_d: [Vector] Input data to despike
    :param med_multiplier: [Scalar] The median multiplier threshold (excludes greater values)
    :return: [Vector] Despiked data
    """

    # Compute a running median on the data
    # The window size (in samples) should be about 5 seconds and odd
    # window_size = int(fs * 120)
    window_size = 589
    if window_size % 2 == 0:
        window_size = window_size + 1
    med = running_median(input_d, window_size)

    # Find values greater than 5 times the running median
    indices_to_remove = []
    for ind in np.arange(len(input_d)):
        if input_d[ind] > abs(med[ind] * med_multiplier) or input_d[ind] < -1 * abs(med[ind] * med_multiplier):
            indices_to_remove.append(ind)

    # Change data values for those indices to zero
    input_d[indices_to_remove] = 0

    return input_d


def cut_data(input_interp_time, input_interp_data, sta_name, input_file_start_time,
             input_year, input_jday, out, target_delta=0.0085):
    """
    Cuts the input interpolated data into hourly time segments and despikes them

    :param input_interp_time: [Vector] Input interpolated time
    :param input_interp_data: [Vector] Input interpolated seismic data
    :param sta_name: [String] Station name
    :param input_jday: [Int] Julian day
    :param input_year: [Int] Input year
    :param input_file_start_time: [Datetime] Start time of day (entire file)
    :param out: [String] Folder where we output the SAC file
    :param target_delta: [Scalar] Time difference between samples

    :return: Nothing is returned
    """

    progress_month = '{:02d}'.format(input_file_start_time.month)
    progress_day = '{:02d}'.format(input_file_start_time.day)
    clock_start_time = time.time()

    # print('------------------------------------')
    # print(f'Working on {sta_name} {input_year}-{progress_month}-{progress_day}...')

    # Set the hourly start and end times (in seconds)
    start_times = []
    for start_hour_index in np.arange(24):
        start_times.append(start_hour_index * 3600)

    # Find the index corresponding to the hourly segments
    for start_hour_index in np.arange(24):
        if start_hour_index == 23:
            end_time = input_interp_time[-1]
        else:
            end_time = start_times[start_hour_index + 1]

        indices = np.intersect1d(np.where(input_interp_time >= start_times[start_hour_index]),
                                 np.where(input_interp_time < end_time))

        hour_type = '{:02d}'.format(start_hour_index)
        output_filename = f'{input_year}{progress_month}{progress_day}_17{sta_name}_{hour_type}_ID'

        # If there isn't any data for the time period, skip this hour
        if len(indices) == 0:
            print(f'{output_filename} : No data')
            continue

        # Set up time variables for this hour that we can call to
        hourly_time = input_interp_time[indices]
        hourly_data = input_interp_data[indices]
        hour_start_time = input_file_start_time + timedelta(seconds=hourly_time[0])

        # If data has already been processed for this hour, skip it
        if os.path.exists(f'{out}{output_filename}'):
            print(f'{output_filename} : Already done')
            continue

        # Create a deep copy so we can check result of despiking (Uncomment for debugging)
        # hourly_data_original = copy.deepcopy(hourly_data)

        # Despike the data
        despiked_data = despike(hourly_data)

        # Save to sac
        # Setup the SAC start header
        header = {'kstnm': 'ST17', 'kcmpnm': sta_name, 'nzyear': int(input_year),
                  'nzjday': input_jday, 'nzhour': hour_start_time.hour, 'nzmin': hour_start_time.minute,
                  'nzsec': hour_start_time.second, 'nzmsec': hour_start_time.microsecond, 'delta': target_delta}

        sac = SACTrace(data=despiked_data, **header)
        sac.write(f'{out}{output_filename}')
        print(f'{output_filename} : Completed')

    elapsed_time = time.time() - clock_start_time

    print(
        f'Finished despiking {sta_name} {input_year}-{progress_month}-{progress_day}! '
        f'(Elapsed time: {np.round(elapsed_time / 60.0, decimals=1)} minutes)')

    return


def process_ascii(input_file, target_delta=0.0085):
    """
    Processes the input ASCII data into resampled, despiked sac traces.
    The input ASCII traces are one day files. The output is in hours.

    :param input_file: [String] Input file
    :param target_delta: [Float] Time between each sample for resampled trace in SECONDS
    :return:
    """

    # Read in the data (one day)
    df = pd.read_csv(input_file, delimiter=' ', header=None)
    data_time = df[0].values - np.floor(df[0][0])
    geo1 = df[1].values
    geo2 = df[2].values
    geo3 = df[3].values
    geo4 = df[4].values

    # Create a resampled time vector
    # Time is currently a float between 0 and 1 representing in DAYS. So it must be converted to SECONDS
    data_time = data_time * 24*60*60
    time_interp = np.arange(np.ceil(data_time[0]), np.floor(data_time[-1]), target_delta)
    geo1_interp = interp_data(data_time, geo1, time_interp)
    geo2_interp = interp_data(data_time, geo2, time_interp)
    geo3_interp = interp_data(data_time, geo3, time_interp)
    geo4_interp = interp_data(data_time, geo4, time_interp)

    # Convert the start and end times into proper datestrings
    file_bn = os.path.basename(input_file)
    year = file_bn[0:4]
    month = file_bn[4:6]
    day = file_bn[6:8]
    file_start = datetime(int(year), int(month), int(day), 0, 0, 0)
    start_time = file_start + timedelta(seconds=time_interp[0])

    # Figure out the julian day corresponding to the year/date
    tt = start_time.timetuple()
    jday = tt.tm_yday

    # Create the output folder
    out_folder = f'{output_directory}{year}{month}{day}/'

    # Split the data into hour segments and save the output
    cut_data(time_interp, geo1_interp, 'Geo1', file_start, year, jday, out_folder)
    cut_data(time_interp, geo2_interp, 'Geo2', file_start, year, jday, out_folder)
    cut_data(time_interp, geo3_interp, 'Geo3', file_start, year, jday, out_folder)
    cut_data(time_interp, geo4_interp, 'Geo4', file_start, year, jday, out_folder)

    return


# Main
data_dir = 'C:/data/lunar_data/'
input_folder = f'{data_dir}LSPE_ascii_day/'

# Number of CPU cores to use in the processing
num_cores = 8

# Find a list of files
folder_list = glob.glob(f'{input_folder}197*')
output_directory = f'{data_dir}LSPE_sac_hourly/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

input_files = []
for folder in folder_list:
    dayfiles = glob.glob(f'{folder}/*.txt')
    for dayname in dayfiles:
        input_files.append(dayname)

# Create the folders before parallelizing due to a bug in the parallel libraries
for input_data in input_files:
    day_actual = os.path.basename(input_data).split('_')[0]
    if not os.path.exists(f'{output_directory}{day_actual}'):
        os.mkdir(f'{output_directory}{day_actual}')

if num_cores == 1:
    # Non-parallel version of the code
    for input_data in input_files:
        process_ascii(input_data)
else:
    # Parallel version of the code
    Parallel(n_jobs=num_cores)(delayed(process_ascii)(input_data) for input_data in input_files)
