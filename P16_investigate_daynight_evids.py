"""
Compares day and night impulsive and emergent events and creates an additional plot which shows when they occur
"""
# Import packages
import pandas as pd
import glob
import pickle
from matplotlib import pyplot as plt
from scipy import fftpack
import numpy as np
from matplotlib import cm
from scipy import signal
import datetime as dt
import csv
import math
import matplotlib
import colorcet as cc
import os
from obspy.core import read


# Functions
def clip_emerg_vals(array, emergence_clip_thresh):
    """
    Clip values at a particular threshold

    :param array: [vector] Values to clip
    :param emergence_clip_thresh: [float] Clip value
    :return:
    """
    output_array = []
    for val in array:
        if val > emergence_clip_thresh:
            output_array.append(emergence_clip_thresh)
        else:
            output_array.append(val)

    return output_array


def compute_radii(input_azi, bins):
    """
    Divide the azimuth into bins

    :param input_azi: [vector] List of azimuth values
    :param bins: [vector] List of bins
    :return:
    """
    input_azi_radii = []
    for bin_ind in np.arange(len(bins) - 1):

        input_azi_radii.append(len(np.intersect1d(np.where(input_azi >= bins[bin_ind])[0],
                                                  np.where(input_azi < bins[bin_ind + 1])[0])))

    return input_azi_radii


def avg(dates):
    """
    Gets the average of a list of datetime values
    :param dates: [vector] List of datetime values
    :return:
    """
    any_reference_date = dt.datetime(1900, 1, 1)
    return any_reference_date + sum([date - any_reference_date for date in dates], dt.timedelta()) / len(dates)


def get_temperature(input_file):
    """
    Strip the whitespace from the files

    :param input_file:
    :return:
    """
    start_time = dt.datetime(1976, 8, 15)

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        time_s, time_day = [], []
        rad, temp_rock, temp_reg = [], [], []
        abs_time = []
        for row in csv_reader:
            if len(row) > 1:
                whitespace_ind = np.where(np.array(row) == '')[0]
                row = np.delete(row, whitespace_ind)
                time_s.append(int(row[0]))
                time_day.append(float(row[1]))
                rad.append(float(row[2]))
                temp_rock.append(float(row[3]))
                temp_reg.append(float(row[4]))
                abs_time.append(start_time + dt.timedelta(seconds=int(row[0])))
        list_of_tuples = list(zip(abs_time, time_s, time_day, rad,
                                  temp_rock, temp_reg))
        temperature_df = pd.DataFrame(list_of_tuples, columns=['abs_time', 'rel_time', 'rel_time_day',
                                                               'rad', 'temp_rock', 'temp_reg'])

    return temperature_df


def remove_evts_from_cat(df, evids):
    """
    Removes faulty events from the pandas dataframe of the catalog

    :param df: [pd df] Pandas dataframe of the seismic catalog
    :param evids: [vector] List of events to remove
    :return:
    """
    inds_to_remove = []
    for evid in evids:
        indices = np.where(df['evid'].values == evid)[0]
        for ind in indices:
            inds_to_remove.append(ind)

    # Find nan indices in theta_var and remove them from everything
    nan_indices = np.argwhere(np.isnan(df['theta_variance'].values))
    for nan_index in nan_indices:
        inds_to_remove.append(nan_index[0])

    df = df.drop(index=inds_to_remove)

    return df


def plot_timing(evid, desig, df):
    """
    Plots the timing of the evid within the time-dependent emergent plot

    :param evid: [str] Event id
    :param desig: [str] Designation of the event from the user (i.e. emergent/impulsive day/night event)
    :param df: [pd df] Pandas dataframe of the catalog
    :return:
    """

    df = df.reset_index(drop=True)
    temp_df = get_temperature(temperature_file)

    theta_var = df[df['geophone'] == 'geo1']['theta_variance'].values
    time_str_geo1 = df[df['geophone'] == 'geo1']['ft_arrival_time'].values
    time_str_geo2 = df[df['geophone'] == 'geo2']['ft_arrival_time'].values
    time_str_geo3 = df[df['geophone'] == 'geo3']['ft_arrival_time'].values
    time_str_geo4 = df[df['geophone'] == 'geo4']['ft_arrival_time'].values

    misfit = df[df['geophone'] == 'geo1']['avg_misfit'].values
    theta_mean = df[df['geophone'] == 'geo1']['theta_mean'].values

    emerg_geo1 = df[df['geophone'] == 'geo1']['emergence'].values
    emerg_geo2 = df[df['geophone'] == 'geo2']['emergence'].values
    emerg_geo3 = df[df['geophone'] == 'geo3']['emergence'].values
    emerg_geo4 = df[df['geophone'] == 'geo4']['emergence'].values
    emerg_mean = []
    for emerg_ind in np.arange(len(emerg_geo1)):
        emerg_mean.append(np.mean([emerg_geo1[emerg_ind], emerg_geo2[emerg_ind],
                                   emerg_geo3[emerg_ind], emerg_geo4[emerg_ind]]))

    # Do the same thing for the PGV
    pgv_geo1 = df[df['geophone'] == 'geo1']['PGV'].values
    pgv_geo2 = df[df['geophone'] == 'geo2']['PGV'].values
    pgv_geo3 = df[df['geophone'] == 'geo3']['PGV'].values
    pgv_geo4 = df[df['geophone'] == 'geo4']['PGV'].values
    pgv_mean = []
    for pgv_ind in np.arange(len(pgv_geo1)):
        pgv_mean.append(np.mean([pgv_geo1[pgv_ind], pgv_geo2[pgv_ind],
                                 pgv_geo3[pgv_ind], pgv_geo4[pgv_ind]]))

    time_num_geo1, time_num_geo2, time_num_geo3, time_num_geo4 = [], [], [], []
    time_num_avg = []
    for timeind in np.arange(len(time_str_geo1)):
        tg1 = dt.datetime.strptime(time_str_geo1[timeind], "%Y-%m-%dT%H:%M:%S.%f")
        tg2 = dt.datetime.strptime(time_str_geo2[timeind], "%Y-%m-%dT%H:%M:%S.%f")
        tg3 = dt.datetime.strptime(time_str_geo3[timeind], "%Y-%m-%dT%H:%M:%S.%f")
        tg4 = dt.datetime.strptime(time_str_geo4[timeind], "%Y-%m-%dT%H:%M:%S.%f")

        time_num_geo1.append(tg1)
        time_num_geo2.append(tg2)
        time_num_geo3.append(tg3)
        time_num_geo4.append(tg4)

        time_num_avg.append(avg([tg1, tg2, tg3, tg4]))

    # Get the cutoff theta variance
    var_cutoff = 50
    num_events_pass = len(np.where(theta_var <= var_cutoff)[0])
    events_pass_ind = np.where(theta_var <= var_cutoff)[0]
    events_fail_ind = np.where(theta_var > var_cutoff)[0]

    # Compute the parameters of the rose plot
    width = (2 * np.pi) / 18
    bins = np.arange(0, 2 * np.pi, width / 2)
    bins = np.append(bins, 1.999 * np.pi)
    theta = []
    for bin_ind in np.arange(len(bins) - 1):
        theta.append(((bins[bin_ind + 1] - bins[bin_ind]) / 2) + bins[bin_ind])

    theta_mean_pass = np.array(theta_mean)[events_pass_ind]
    theta_pass_values = [math.radians(val) for val in np.array(theta_mean)[events_pass_ind]]
    theta_pass_radii = compute_radii(theta_pass_values, bins)
    emerg_mean_pass = np.array(emerg_mean)[events_pass_ind]
    emerg_geo1_pass = np.array(emerg_geo1)[events_pass_ind]
    emerg_geo2_pass = np.array(emerg_geo2)[events_pass_ind]
    emerg_geo3_pass = np.array(emerg_geo3)[events_pass_ind]
    emerg_geo4_pass = np.array(emerg_geo4)[events_pass_ind]

    pgv_mean_pass = np.array(pgv_mean)[events_pass_ind]
    pgv_geo1_pass = np.array(pgv_geo1)[events_pass_ind]
    pgv_geo2_pass = np.array(pgv_geo2)[events_pass_ind]
    pgv_geo3_pass = np.array(pgv_geo3)[events_pass_ind]
    pgv_geo4_pass = np.array(pgv_geo4)[events_pass_ind]

    emerg_mean_fail = np.array(emerg_mean)[events_fail_ind]
    emerg_geo1_fail = np.array(emerg_geo1)[events_fail_ind]
    emerg_geo2_fail = np.array(emerg_geo2)[events_fail_ind]
    emerg_geo3_fail = np.array(emerg_geo3)[events_fail_ind]
    emerg_geo4_fail = np.array(emerg_geo4)[events_fail_ind]

    pgv_mean_fail = np.array(pgv_mean)[events_fail_ind]
    pgv_geo1_fail = np.array(pgv_geo1)[events_fail_ind]
    pgv_geo2_fail = np.array(pgv_geo2)[events_fail_ind]
    pgv_geo3_fail = np.array(pgv_geo3)[events_fail_ind]
    pgv_geo4_fail = np.array(pgv_geo4)[events_fail_ind]

    time_num_avg_pass = np.array(time_num_avg)[events_pass_ind]
    time_num_geo1_pass = np.array(time_num_geo1)[events_pass_ind]
    time_num_geo2_pass = np.array(time_num_geo2)[events_pass_ind]
    time_num_geo3_pass = np.array(time_num_geo3)[events_pass_ind]
    time_num_geo4_pass = np.array(time_num_geo4)[events_pass_ind]

    time_num_avg_fail = np.array(time_num_avg)[events_fail_ind]
    time_num_geo1_fail = np.array(time_num_geo1)[events_fail_ind]
    time_num_geo2_fail = np.array(time_num_geo2)[events_fail_ind]
    time_num_geo3_fail = np.array(time_num_geo3)[events_fail_ind]
    time_num_geo4_fail = np.array(time_num_geo4)[events_fail_ind]

    # Set a max for emergence color
    emergence_clip_thresh = 40
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=emergence_clip_thresh)
    cmap = plt.cm.jet_r

    # Clip the emergence values
    emerg_mean_pass_clipped = clip_emerg_vals(emerg_mean_pass, emergence_clip_thresh)
    emerg_mean_fail_clipped = clip_emerg_vals(emerg_mean_fail, emergence_clip_thresh)

    emerg_geo1_pass_clipped = clip_emerg_vals(emerg_geo1_pass, emergence_clip_thresh)
    emerg_geo2_pass_clipped = clip_emerg_vals(emerg_geo2_pass, emergence_clip_thresh)
    emerg_geo3_pass_clipped = clip_emerg_vals(emerg_geo3_pass, emergence_clip_thresh)
    emerg_geo4_pass_clipped = clip_emerg_vals(emerg_geo4_pass, emergence_clip_thresh)

    emerg_geo1_fail_clipped = clip_emerg_vals(emerg_geo1_fail, emergence_clip_thresh)
    emerg_geo2_fail_clipped = clip_emerg_vals(emerg_geo2_fail, emergence_clip_thresh)
    emerg_geo3_fail_clipped = clip_emerg_vals(emerg_geo3_fail, emergence_clip_thresh)
    emerg_geo4_fail_clipped = clip_emerg_vals(emerg_geo4_fail, emergence_clip_thresh)

    # Clip the PGV
    pgv_clip_thresh = 400
    pgv_mean_pass_clipped = clip_emerg_vals(pgv_mean_pass, pgv_clip_thresh)
    pgv_mean_fail_clipped = clip_emerg_vals(pgv_mean_fail, pgv_clip_thresh)

    pgv_geo1_pass_clipped = clip_emerg_vals(pgv_geo1_pass, pgv_clip_thresh)
    pgv_geo2_pass_clipped = clip_emerg_vals(pgv_geo2_pass, pgv_clip_thresh)
    pgv_geo3_pass_clipped = clip_emerg_vals(pgv_geo3_pass, pgv_clip_thresh)
    pgv_geo4_pass_clipped = clip_emerg_vals(pgv_geo4_pass, pgv_clip_thresh)

    pgv_geo1_fail_clipped = clip_emerg_vals(pgv_geo1_fail, pgv_clip_thresh)
    pgv_geo2_fail_clipped = clip_emerg_vals(pgv_geo2_fail, pgv_clip_thresh)
    pgv_geo3_fail_clipped = clip_emerg_vals(pgv_geo3_fail, pgv_clip_thresh)
    pgv_geo4_fail_clipped = clip_emerg_vals(pgv_geo4_fail, pgv_clip_thresh)

    # Create a new colormap for this other plot
    color_norm2 = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(theta_mean_pass))
    cmap2 = cc.cm.colorwheel

    # Find the timing of the particular evid we're looking at
    evid_index = np.where(df[df['geophone'] == 'geo1']['evid'].values == evid)[0][0]
    evid_time_num_avg = time_num_avg[evid_index]
    evid_emerg_mean = emerg_mean[evid_index]
    evid_theta_mean = theta_mean[evid_index]

    fig4 = plt.figure(figsize=(18, 5), num=1, clear=True)
    ax0 = plt.subplot2grid((1, 5), (0, 0), colspan=4, rowspan=1)
    temp_rock_plt = ax0.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')

    color0 = 'tab:red'
    ax0.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    # ax0.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper right', fancybox=True,
    #            ncol=2)
    ax0.tick_params(axis='y', labelcolor=color0)
    ax0.set_xlim((temp_df['abs_time'].values[0], temp_df['abs_time'].values[-1]))
    # ax0.axvline(abs_day, c='k')
    ax0b = ax0.twinx()  # instantiate a second axes that shares the same x-axis
    color0b = 'tab:blue'
    ax0b.set_ylabel('Average Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_avg_pass, emerg_mean_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_avg_fail, emerg_mean_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)
    ax0b.axvline(evid_time_num_avg, color='black')
    ax0b.scatter(evid_time_num_avg, evid_emerg_mean,
                 color=cmap2(color_norm2(evid_theta_mean)), edgecolor='red', linewidth=1.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig4.tight_layout()
    fig4.subplots_adjust(top=0.9)
    fig4.suptitle(f'Evid {evid}: {desig}')
    fig4.savefig(f'{outfolder}{desig}_evid_{evid}_loc.png')
    fig4.savefig(f'{outfolder}{desig}_evid_{evid}_loc.eps')

    return


def pull_individual_mq(input_file, evid, abs_time, pre_event_window_size=180, post_event_window_size=360):
    """
    Plots the time series and spectrogram of a single moonquake file
    :param input_file: [str] Filepath of the hourly data trace
    :param evid: [str] The event ID
    :param window_size: [int] Size of the window in seconds around the moonquake

    :return:
    """

    # Find the header information for this station earthquake in the mq catalog
    file_bn = os.path.basename(input_file)
    sta = f"g{file_bn.split('_')[1][3:]}"
    # evid_header = cat.loc[(cat['evid'] == evid) & (cat['station'] == sta)].iloc[0]

    # Create variables based on the header
    # sta = evid_header['station']
    ucase_sta = f'G{sta[1:]}'
    # abs_time = evid_header['abs_time']
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
    dt_event = dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    dt_trace_start = dt.datetime(int(year), int(month), int(day), int(hour))
    diff = dt_event - dt_trace_start
    rel_det = diff.seconds

    f, t, Sxx = signal.spectrogram(tr.data, tr.stats.sampling_rate)

    # # Plot the figure
    tme = (tr.times("matplotlib") - tr.times("matplotlib")[0]) * 86400

    return tme, tr.data, t, f, Sxx, rel_det


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


def plot_evid(evid, desig, df, pre_event_window_size=180, post_event_window_size=360):
    """
    Main wrapper code for plotting the event time series, spectrogram, and FFT for all geophones

    :param evid: [str] Event ID
    :param desig: [str] Designation of the event from the user (i.e. emergent/impulsive day/night event)
    :param df: [pd df] Pandas dataframe of the catalog
    :return:
    """

    plot_timing(evid, desig, df)

    # Load the pickle file, although we're only going to be using the input info varible
    trace_file = glob.glob(f'{pkl_folder}evid_{evid}.pkl')[0]
    with open(trace_file, 'rb') as f:
        time_array_cut_tmp, data_array_cut_tmp, abs_trace_start, abs_trace_end, \
        rel_det_vector, input_info, data_geophone_list = pickle.load(f)

    # Get the absolute time
    abs_time = input_info['abs_time']
    year = abs_time.split('-')[0]
    month = abs_time.split('-')[1]
    day = abs_time.split('-')[2].split('T')[0]
    hour = abs_time.split('T')[1].split(':')[0]
    evid = input_info['evid']

    data_files = sorted(glob.glob(f'{sacdata}{year}{month}{day}/{year}{month}{day}_17Geo*_{hour}_ID'))

    geophone_ind = 0
    for data_file in data_files:
        time, data, spec_t, spec_f, spec_Sxx, rel_det = pull_individual_mq(data_file, input_info['evid'], abs_time)

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

    ref_geophone_ind = 0
    first_arrival = np.min(rel_det_vector)
    last_arrival = np.max(rel_det_vector)

    # Cut the time series
    xcut_min_ind = find_nearest(time_array[:, ref_geophone_ind], first_arrival - pre_event_window_size)
    xcut_max_ind = find_nearest(time_array[:, ref_geophone_ind], last_arrival + post_event_window_size)

    # Cut the data
    time_array_cut = time_array[xcut_min_ind:xcut_max_ind, :]
    data_array_cut = data_array[xcut_min_ind:xcut_max_ind, :]

    # Cut the Spectrogram
    xcut_min_ind_spec = find_nearest(spec_t_array[:, ref_geophone_ind], first_arrival - pre_event_window_size)
    xcut_max_ind_spec = find_nearest(spec_t_array[:, ref_geophone_ind], last_arrival + post_event_window_size)
    spec_t_array = spec_t_array[xcut_min_ind_spec:xcut_max_ind_spec, :]
    # spec_f_array = spec_f_array[xcut_min_ind_spec:xcut_max_ind_spec, :]
    Sxx_array = Sxx_array[:, xcut_min_ind_spec:xcut_max_ind_spec, :]

    # Take an FFT of the data
    # Find the time difference using the time vector
    sr_target = 117.7667
    delta_target = 1. / sr_target
    fft_f = fftpack.fftfreq(len(time_array_cut[:, 0]), d=delta_target)
    fft_full_array = np.zeros((len(fft_f), 4))

    for geophone_ind in np.arange(4):
        data_vector = data_array_cut[:, geophone_ind]
        fft_full_array[:, geophone_ind] = fftpack.fft(data_vector)

    # Plot a combined figure
    fig = plt.figure(figsize=(20, 10), num=2, clear=True)
    # Plot the time series
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(time_array_cut[:, 0], data_array_cut[:, 0])
    ax1.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))
    ax1.set_title('Geo1', fontweight='bold')

    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(time_array_cut[:, 0], data_array_cut[:, 1])
    ax2.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))
    ax2.set_title('Geo2', fontweight='bold')

    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(time_array_cut[:, 0], data_array_cut[:, 2])
    ax3.set_title('Geo3', fontweight='bold')
    ax3.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))

    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(time_array_cut[:, 0], data_array_cut[:, 3])
    ax4.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))
    ax4.set_title('Geo4', fontweight='bold')

    # Plot the spectrograms
    specmax = 1e-6

    ax5 = plt.subplot(3, 4, 5)
    ax5.pcolormesh(spec_t_array[:, 0], spec_f_array[:, 0], Sxx_array[:, :, 0], vmax=specmax, cmap=cm.jet, shading='auto')
    # ax5.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))

    ax6 = plt.subplot(3, 4, 6)
    ax6.pcolormesh(spec_t_array[:, 1], spec_f_array[:, 1], Sxx_array[:, :, 1], vmax=specmax, cmap=cm.jet, shading='auto')
    # ax6.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))

    ax7 = plt.subplot(3, 4, 7)
    ax7.pcolormesh(spec_t_array[:, 2], spec_f_array[:, 2], Sxx_array[:, :, 2], vmax=specmax, cmap=cm.jet, shading='auto')
    # ax7.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))

    ax8 = plt.subplot(3, 4, 8)
    ax8.pcolormesh(spec_t_array[:, 3], spec_f_array[:, 3], Sxx_array[:, :, 3], vmax=specmax, cmap=cm.jet, shading='auto')
    # ax8.set_xlim((time_array_cut[:, 0][0], time_array_cut[:, 0][-1]))

    # Plot the FFTs
    ax9 = plt.subplot(3, 4, 9)
    ax9.loglog(fft_f[0:len(fft_f)//2], abs(fft_full_array[0:len(fft_f)//2, 0]))
    ax9.set_xlim((0.9, sr_target/2))
    ax9.set_ylim((10**(-6), 10**1))

    ax10 = plt.subplot(3, 4, 10)
    ax10.loglog(fft_f[0:len(fft_f) // 2], abs(fft_full_array[0:len(fft_f) // 2, 1]))
    ax10.set_xlim((0.9, sr_target / 2))
    ax10.set_ylim((10 ** (-6), 10 ** 1))

    ax11 = plt.subplot(3, 4, 11)
    ax11.loglog(fft_f[0:len(fft_f) // 2], abs(fft_full_array[0:len(fft_f) // 2, 2]))
    ax11.set_xlim((0.9, sr_target / 2))
    ax11.set_ylim((10 ** (-6), 10 ** 1))

    ax12 = plt.subplot(3, 4, 12)
    ax12.loglog(fft_f[0:len(fft_f) // 2], abs(fft_full_array[0:len(fft_f) // 2, 3]))
    ax12.set_xlim((0.9, sr_target / 2))
    ax12.set_ylim((10 ** (-6), 10 ** 1))

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f'{desig} : {evid}', fontweight='bold')
    fig.savefig(f'{outfolder}{desig}_evid_{evid}.png')
    fig.savefig(f'{outfolder}{desig}_evid_{evid}.eps')
    print(f'Saved {desig}_evid_{evid}...')

    return


# Main
datadir = 'C:/data/lunar_data/'
sacdata = f'{datadir}LSPE_sac_hourly/'
outdir = 'C:/data/lunar_output/'
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
spectr_folder = f'{outdir}fine-tuned/ft_spectrograms/'
catalog_file = f'{outdir}catalogs/GradeA_thermal_mq_catalog_final.csv'
pkl_folder = f'{outdir}original_images/combined_files/'
temperature_file = f'{rundir}longterm_thermal_data.txt'

# Create output directory if it doesn't exist already
outfolder = f'{outdir}results/thermal_results_daynight/'
if not os.path.exists(outfolder):
    os.mkdir(outfolder)

# Read in the catalog and remove problematic events
cat_df = pd.read_csv(catalog_file)
evids_to_exclude = ['770425-00-M1', '761111-21-M2', '770416-10-M1', '770325-13-M6', '770114-15-M1',
                    '761105-08-M1', '770314-22-M1', '761021-07-M1', '760901-14-M2']
cat_df = remove_evts_from_cat(cat_df, evids_to_exclude)


# Select the night/day emergent and impulsive events to look at
in_evids = ['760920-07-M1', '760921-20-M1', '760904-13-M2', '760904-01-M1']
in_designation = ['night_emergent', 'night_impulsive', 'day_emergent', 'day_impulsive']

for cycle_ind in np.arange(len(in_designation)):
    plot_evid(in_evids[cycle_ind], in_designation[cycle_ind], cat_df)
