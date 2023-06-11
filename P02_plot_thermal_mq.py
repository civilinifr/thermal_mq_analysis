"""
Plots the thermal moonquakes detected by the Civilini et al. 2021 catalog and saves their data as PKL files for easier
access during the finetuning step.

"""
# Import packages
import pandas as pd
import numpy as np
import os
import glob
from obspy.core import read
from scipy import signal
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from matplotlib import pyplot as plt
import datetime
from matplotlib import cm
from joblib import Parallel, delayed
import pickle
# import warnings
# import matplotlib.cbook
# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# Setup functions
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


def run_highpass_filter(input_data, input_corner_freq):
    """
    Passes a highpass filter on the data

    :param input_data: [obspy trace] Seismic input trace in Obspy format
    :param input_corner_freq: [int] Corner frequency of highpass filter
    :return:
    """
    # Remove the mean and run a cosine taper
    trace_nomean = input_data.data - np.mean(input_data.data)
    N = len(trace_nomean)
    taper_function = cosine_taper(N, p=0.1)
    trace_taper = trace_nomean * taper_function

    # Highpass filterbound in Hertz
    data_filt = highpass(trace_taper, input_corner_freq, input_data.stats.sampling_rate, corners=4, zerophase=False)
    tr_filt = input_data
    tr_filt.data = data_filt

    return tr_filt


def plot_individual_mq(input_file, evid, pre_event_window_size=180, post_event_window_size=360):
    """
    Plots the time series and spectrogram of a single moonquake file
    :param input_file: [str] Filepath of the hourly data trace
    :param evid: [str] The event ID
    :param pre_event_window_size: [int] Size of the window in seconds before the moonquake arrival
    :param post_event_window_size: [int] Size of the window in seconds after the moonquake arrival
    :return:
    """

    # Find the header information for this station earthquake in the mq catalog
    file_bn = os.path.basename(input_file)
    sta = f"g{file_bn.split('_')[1][3:]}"
    evid_header = cat.loc[(cat['evid'] == evid) & (cat['station'] == sta)].iloc[0]

    # Create variables based on the header
    sta = evid_header['station']
    ucase_sta = f'G{sta[1:]}'
    abs_time = evid_header['abs_time']
    year = abs_time.split('-')[0]
    month = abs_time.split('-')[1]
    day = abs_time.split('-')[2].split('T')[0]
    hour = abs_time.split('T')[1].split(':')[0]
    minute = abs_time.split('T')[1].split(':')[1]
    second = abs_time.split('T')[1].split(':')[2]

    # Get the UTC time from the sacfile (it takes too much space to output to txt in P3_a)
    st = read(input_file)
    tr = st[0]
    utc_time = tr.times("utcdatetime")

    # Find the relative time to the start of the record for the moonquake arrival
    dt_event = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    dt_trace_start = datetime.datetime(int(year), int(month), int(day), int(hour))
    diff = dt_event - dt_trace_start
    rel_det = diff.seconds

    f_original, t_original, Sxx_original = signal.spectrogram(tr.data, tr.stats.sampling_rate)
    # Find the maximum value for the segment of data we're looking at
    Sxx_window_start_ind = find_nearest(t_original, rel_det - Sxx_background_threshold_window)
    Sxx_window_end_ind = find_nearest(t_original, rel_det)
    Sxx_subset = Sxx_original[:, Sxx_window_start_ind:Sxx_window_end_ind]
    Sxx_background_val = np.max(Sxx_subset)
    if Sxx_background_val < Sxx_background_threshold:
        mia_data = 1
    else:
        mia_data = 0

    # Use below if you need to run a highpass filter
    # trace_filtered = tr
    trace_filtered = run_highpass_filter(tr, 10.0)
    f, t, Sxx = signal.spectrogram(trace_filtered.data, trace_filtered.stats.sampling_rate)

    # Have to change the evid header for the absolute time for Windows machines. Windows doesn't support :
    evid_header['abs_time'] = f"{evid_header['abs_time'].split('T')[0]}T" \
                              f"{evid_header['abs_time'].split('T')[1].split(':')[0]}.{evid_header['abs_time'].split('T')[1].split(':')[1]}." \
                              f"{evid_header['abs_time'].split('T')[1].split(':')[2]}"

    # Name the title and figure
    plt_title = f"{evid_header['station']}, {evid_header['evid']}, {evid_header['abs_time']}, " \
                f"Grade {evid_header['grade']}, SNR = {evid_header['snr']} ({evid_header['practical_snr']})"
    plt_filename = f"{evid_header['station']}_{evid_header['evid']}_{evid_header['abs_time']}_G{evid_header['grade']}"

    # # Plot the figure
    tme = (tr.times("matplotlib") - tr.times("matplotlib")[0]) * 86400

    if plt_individual == 'yes':
        fig = plt.figure(figsize=(10, 10), num=1, clear=True)
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(tme, tr.data, "b-")
        ax.set_ylabel('Velocity (volts)', fontweight='bold')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_xlim((rel_det - pre_event_window_size, rel_det + post_event_window_size))
        ax.axvline(x=rel_det, c='r')
        ax.axvspan(rel_det, rel_det + evid_header['envelope_size'], alpha=0.5, color='gray')
        ax.set_title(plt_title, fontweight='bold')

        ax2 = fig.add_subplot(2, 1, 2)
        specmax = 1e-6
        specax = ax2.pcolormesh(t, f, Sxx, cmap=cm.jet, vmax=specmax, shading='auto')
        ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2.set_xlabel('Time (sec)', fontweight='bold')
        ax2.set_xlim((rel_det - pre_event_window_size, rel_det + post_event_window_size))
        ax2.axvline(x=rel_det, c='r')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2, top=0.9)
        cbar_ax = fig.add_axes([0.30, 0.08, 0.4, 0.03])
        cbar = fig.colorbar(specax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Spectral Energy', fontweight='bold')
        fig.savefig(f"{out_image_directory}{evid_header['station']}/{plt_filename}.png")
        # plt.savefig(f"{out_image_directory}{evid_header['station']}/EPS_{plt_filename}.eps")

    return tme, tr.data, t, f, Sxx, rel_det, mia_data


def plot_combined(time_array, data_array, spec_t_array, spec_f_array, rel_det_vector,
                  Sxx_array, evid, data_geophone_list, pre_event_window_size, post_event_window_size):
    """

    :param time_array: [np_array] 2D array of time series time
    :param data_array: [np_array] 2D array of time series data
    :param spec_t_array: [np_array] 2D array of spectrogram time
    :param spec_f_array: [np_array] 2D array of spectrogram frequency
    :param rel_det_vector: [list] Vector of relative detections
    :param Sxx_array: [np_array] 3D array of sepctrogram values, with depth correpsonding to geophone
    :param data_geophone_list: [vector] Describes which index corresponds to which geophone
    :param evid: [str] event ID
    :param window_size: [int] Size of the window in seconds around the moonquake
    :return:
    """
    # We will plot by iterating across each geophone
    # This way our codes will be flexible when it comes to events with grade lower than A (i.e. don't have 4 geophones)
    num_geophones = np.shape(time_array)[1]

    # Find a good ylimit for plotting the traces
    # Cut the traces for what we are plotting below
    max_trace_ylim = []
    for geophone_ind in np.arange(num_geophones):
        xlim_min_ind = find_nearest(time_array[:, geophone_ind], rel_det_vector[geophone_ind] - pre_event_window_size)
        xlim_max_ind = find_nearest(time_array[:, geophone_ind], rel_det_vector[geophone_ind] + post_event_window_size)
        segmented_trace = data_array[xlim_min_ind:xlim_max_ind, geophone_ind]
        max_trace_ylim.append(np.max([np.max(segmented_trace), abs(np.min(segmented_trace))]))
    trace_plot_ymax = np.max(max_trace_ylim)

    # Start the iteration and plot
    fig, axs = plt.subplots(2, num_geophones, figsize=(16, 7), num=2, clear=True)
    for geophone_ind in np.arange(num_geophones):

        # Plot the time series
        axs[0, geophone_ind].plot(time_array[:, geophone_ind], data_array[:, geophone_ind], "b-")
        axs[0, geophone_ind].set_ylabel('Amplitude (counts)', fontweight='bold')
        axs[0, geophone_ind].set_xlabel('Time (s)', fontweight='bold')
        axs[0, geophone_ind].set_xlim((rel_det_vector[geophone_ind] - pre_event_window_size,
                                       rel_det_vector[geophone_ind] + post_event_window_size))
        axs[0, geophone_ind].set_ylim((-1*trace_plot_ymax, trace_plot_ymax))
        axs[0, geophone_ind].axvline(x=rel_det_vector[geophone_ind], c='r')
        axs[0, geophone_ind].set_title(f'{data_geophone_list[geophone_ind]} (t={int(rel_det_vector[geophone_ind])} sec)')

        # Plot the spectrogram
        specmax = 1e-6
        specax = axs[1, geophone_ind].pcolormesh(spec_t_array[:, geophone_ind], spec_f_array[:, geophone_ind],
                                                 Sxx_array[:, :, geophone_ind], cmap=cm.jet, vmax=specmax,
                                                 shading='auto')
        # specax = axs[1, geophone_ind].pcolormesh(spec_t_array[:, geophone_ind], spec_f_array[:, geophone_ind],
        #                                          Sxx_array[:, :, geophone_ind], cmap=cm.jet)
        axs[1, geophone_ind].set_ylabel('Frequency (Hz)', fontweight='bold')
        axs[1, geophone_ind].set_xlabel('Time (sec)', fontweight='bold')
        axs[1, geophone_ind].set_xlim((rel_det_vector[geophone_ind] - pre_event_window_size,
                                       rel_det_vector[geophone_ind] + post_event_window_size))
        axs[1, geophone_ind].axvline(x=rel_det_vector[geophone_ind], c='r')

    # Do tight_layout BEFORE adjusting the subplots to add the colorbar
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.9)
    fig.suptitle(f'{evid}', fontweight='bold')
    cbar_ax = fig.add_axes([0.30, 0.08, 0.4, 0.03])
    cbar = fig.colorbar(specax, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Spectral Energy', fontweight='bold')
    fig.savefig(f"{out_image_directory}combined/GradeA_evid_{evid}.png")
    # fig.savefig(f"{out_image_directory}combined/EPS_GradeA_evid_{evid}.eps")
    # plt.close('all')

    return


def plot_mq_wrapper(input_info, pre_event_window_size=180, post_event_window_size=360):
    """
    Wrapper code for plotting the thermal moonquakes
    :param input_info: [series] Displays the catalog information for the moonquake
    :param pre_event_window_size: [float] Time in seconds before the arrival to take
    :param post_event_window_size: [float] Time in seconds after the arrival to take
    :return:
    """
    # Get information about the moonquake from the header
    # Note, get only the general date/hour for this step, as the precise time will vary for the instruments.
    abs_time = input_info['abs_time']
    year = abs_time.split('-')[0]
    month = abs_time.split('-')[1]
    day = abs_time.split('-')[2].split('T')[0]
    hour = abs_time.split('T')[1].split(':')[0]
    evid = input_info['evid']

    # If the event has already been done, skip
    if os.path.exists(f'{out_image_directory}combined_files/evid_{evid}.pkl'):
        print(f'Event {evid} has already been processed...')
        return

    # Find the piece of data that matches this event (data is hourly)
    data_files = sorted(glob.glob(f'{sacdata}{year}{month}{day}/{year}{month}{day}_17Geo*_{hour}_ID'))

    # Create a data geophone list so we can know which index corresponds to which geophone
    # Note, this is fine for Grade A events where it goes Geo1, 2, 3, 4, but it's good to keep track
    # for later down the line if we choose events with fewer number of geophones.
    num_geophones = len(data_files)
    data_geophone_list = []
    for geophone_ind in np.arange(num_geophones):
        data_file_bn = os.path.basename(data_files[geophone_ind])
        data_geophone_list.append(f"g{data_file_bn.split('_')[1].split('17')[1][1:]}")

    # Plot individual figures of the events
    # Save each data vector for the plotting so we can easily combine without reopening the data
    # We have to wait to intialize the arrays after getting the shape from the first piece of data
    geophone_ind = 0
    mia_vector = []
    for data_file in data_files:
        time, data, spec_t, spec_f, spec_Sxx, rel_det, mia_val = plot_individual_mq(data_file, input_info['evid'])
        mia_vector.append(mia_val)

        # Note: time_array, data_array, spec_f_array, spec_t_array will be 2D ARRAYS
        # rel_det_vector will be a VECTOR
        # Sxx_array is a 3D array with depth being the iteration of the geophone
        if geophone_ind == 0:
            time_array = np.zeros((len(time), len(data_files)))
            data_array = np.zeros((len(data), len(data_files)))
            spec_t_array = np.zeros((len(spec_t), len(data_files)))
            spec_f_array = np.zeros((len(spec_f), len(data_files)))
            rel_det_vector = np.zeros((len(data_files)))
            Sxx_array = np.zeros((np.shape(spec_Sxx)[0], np.shape(spec_Sxx)[1], len(data_files)))

        # Now place each vector or array into their representative array (or 3D array in the case of Sxx)
        time_array[:, geophone_ind] = time
        data_array[:, geophone_ind] = data
        spec_t_array[:, geophone_ind] = spec_t
        spec_f_array[:, geophone_ind] = spec_f
        rel_det_vector[geophone_ind] = rel_det
        Sxx_array[:, :, geophone_ind] = spec_Sxx

        # Increase the geophone index
        geophone_ind = geophone_ind + 1

    # If the background value is very low (i.e. missing data), do not continue the processing
    mia_sum = np.sum(mia_vector)
    if mia_sum > 0:
        print(f'{evid} has missing data! Skipping...')
        return

    # Save the cut time series into a different location
    # The time-window will need to be a bit larger to make sure that we are getting the full extent of the trace
    # Use geophone 1 as the reference geophone (but it shouldnt matter)
    ref_geophone_ind = 0
    first_arrival = np.min(rel_det_vector)
    last_arrival = np.max(rel_det_vector)
    xcut_min_ind = find_nearest(time_array[:, ref_geophone_ind], first_arrival - pre_event_window_size)
    xcut_max_ind = find_nearest(time_array[:, ref_geophone_ind], last_arrival + post_event_window_size)

    # Plot a figure showing all of the traces together
    plot_combined(time_array, data_array, spec_t_array, spec_f_array, rel_det_vector, Sxx_array,
                  evid, data_geophone_list, pre_event_window_size, post_event_window_size)

    # As the time array is displayed in relative time, we need to know the start and end times of the trace
    xcut_min_ind_value = time_array[xcut_min_ind, ref_geophone_ind]
    xcut_max_ind_value = time_array[xcut_max_ind, ref_geophone_ind]
    dt_start = datetime.datetime(int(year), int(month), int(day), int(hour))
    abs_trace_start = dt_start + datetime.timedelta(seconds=xcut_min_ind_value)
    abs_trace_end = dt_start + datetime.timedelta(seconds=xcut_max_ind_value)

    # Cut the data
    time_array_cut = time_array[xcut_min_ind:xcut_max_ind, :]
    data_array_cut = data_array[xcut_min_ind:xcut_max_ind, :]

    # Save this information along with the header of the hour file in the pickle
    with open(f'{out_image_directory}combined_files/evid_{evid}.pkl', 'wb') as f:
        pickle.dump([time_array_cut, data_array_cut, abs_trace_start, abs_trace_end, rel_det_vector,
                     input_info, data_geophone_list], f)
    print(f"Finished saving figures and data for {evid}...")

    return


def find_largest_events(input_cat, num_events):
    """
    Returns the indices of the largest snr events within a catalog file
    :param input_cat: [pd df] Catalog file
    :param num_events: [int] The number of events we want
    :return:
    """
    snr_vector = input_cat['practical_snr'].values
    maxsnr_evids = []
    for evt_index in np.arange(num_events):
        maxval_index = np.where(snr_vector == np.max(snr_vector))[0][0]

        # Find the evt corresponding to this index
        event_desig = input_cat['evid'].values[maxval_index]
        maxsnr_evids.append(event_desig)
        snr_vector[maxval_index] = -9999

    # For all of these evids, remove the ones whose maximum amplitude value is not

    # Now go back to the original catalog and find the index corresponding to the evid
    maxsnr_idx = []
    for maxsnr_evid in maxsnr_evids:
        maxsnr_idx.append(np.where(input_cat['evid'].values == maxsnr_evid)[0][0])

    return maxsnr_idx


# Main
# Running directory is where the codes are located
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
data_dir = 'C:/data/lunar_data/'
sacdata = f'{data_dir}LSPE_sac_hourly/'
output_dir = 'C:/data/lunar_output/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Choose whether or not to plot individual plots for each instrument or just a single aggregate plot
plt_individual = 'yes'

# Set number of CPU cores
num_cores = 5

catalog_file = f'{rundir}fc_lspe_dl_catalog.csv'
out_image_directory = f'{output_dir}original_images/'

if not os.path.exists(out_image_directory):
    os.mkdir(out_image_directory)

# Setup the maximum amplitude threshold. If all 4 geophones are less than this value, do not process
Sxx_background_threshold = 1e-10
Sxx_background_threshold_window = 50

# Setup the output image directories for the geophone
if not os.path.exists(f'{out_image_directory}combined/'):
    os.mkdir(f'{out_image_directory}combined/')
if not os.path.exists(f'{out_image_directory}combined_files/'):
    os.mkdir(f'{out_image_directory}combined_files/')
geophones = ['geo1', 'geo2', 'geo3', 'geo4']
for geophone in geophones:
    if not os.path.exists(f'{out_image_directory}{geophone}'):
        os.mkdir(f'{out_image_directory}{geophone}')

# Select Grade A events from the catalog (found on all 4 geophones). Select B, C, D, for other grades
cat = pd.read_csv(catalog_file)
grade_a = cat.loc[(cat['grade'] == 'A') & (cat['station'] == 'geo1')]
grade_a = grade_a.reset_index()
subset_cat = grade_a

# For each event, plot a figure showing the time series and spectrogram of the event
if num_cores == 1:
    for image_ind in np.arange(len(subset_cat)):
        plot_mq_wrapper(subset_cat.iloc[image_ind])
else:
    Parallel(n_jobs=num_cores)(delayed(plot_mq_wrapper)(subset_cat.iloc[image_ind])
                               for image_ind in np.arange(len(subset_cat)))



