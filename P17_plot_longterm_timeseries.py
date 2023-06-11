"""
Check continuous spectrogram segements for changes during night-day transitions.

Warning: If you plan to output images in EPS format, know that they are large files!

"""
# Import packages
import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import glob
import os
from obspy.core import read
from scipy import signal
from matplotlib import cm
import math
from joblib import Parallel, delayed
import colorcet as cc
import matplotlib


# Functions
def avg(dates):
    """
    Gets the average of a list of datetime values
    :param dates: [vector] List of datetime values
    :return:
    """
    any_reference_date = dt.datetime(1900, 1, 1)
    return any_reference_date + sum([date - any_reference_date for date in dates], dt.timedelta()) / len(dates)


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


def get_temperature(input_file):
    """
    Strip the whitespace from the files

    :param input_file: [str] Path to input file
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


def find_files(dir, start, end):
    """
    Finds the hourly files

    :param dir: [str] path to the input directory where the files are held.
    :param start: [str] Beginning day for plotting in YYYY-MM-DD format.
    :param end: [str] Ending day for plotting in YYYY-MM-DD format.
    :return:
    """

    folderlist = sorted(glob.glob(f'{dir}*'))
    start_mod = f'{start.split("-")[0]}{start.split("-")[1]}{start.split("-")[2]}'
    start_mod_ind = np.where(np.array(folderlist) == f'{dir[0:-1]}\\{start_mod}')[0][0]
    end_mod = f'{end.split("-")[0]}{end.split("-")[1]}{end.split("-")[2]}'
    end_mod_ind = np.where(np.array(folderlist) == f'{dir[0:-1]}\\{end_mod}')[0][0]

    folderlist_condensed = folderlist[start_mod_ind:end_mod_ind]

    # For each folder, find the files associated with this time period
    filelist = []
    for folder in folderlist_condensed:
        sub_filelist = glob.glob(f'{folder}/*_17G{instrument[1:]}_*')
        for subfile in sub_filelist:
            filelist.append(subfile)

    return filelist


def plot_hourly_trace(infile, num_hours):
    """
    Plots the hourly trace, its spectrogram, and its position with temperature and gradient temperature

    :param infile: [str] Path to the hourly file
    :param num_hours: [int] The number of hours we are processing. Used to compute the percentage completion.
    :return:
    """
    # Get the timing from the filename
    infile_bn = os.path.basename(infile)
    date = infile_bn.split('_')[0]
    date_mod = f'{date[0:4]}-{date[4:6]}-{date[6:8]}'
    hour = infile_bn.split('_')[-2]
    filestring = f'{date_mod}_hour{hour}'

    if os.path.exists(f'{outdir}{filestring}.png'):
        print(f'{filestring} already processed')
        return

    # Turn the start times and end times in datenums
    start_datetime = dt.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(hour))
    end_datetime = start_datetime + dt.timedelta(hours=1)

    # Check if we have any mia_GradeA, gradeB, gradeC, and gradeD events in this
    mia_gradeA_match = np.intersect1d(np.where(np.array(mia_gradeA_arrivals) > start_datetime),
                                      np.where(np.array(mia_gradeA_arrivals) < end_datetime))
    gradeB_match = np.intersect1d(np.where(np.array(gradeB_arrivals) > start_datetime),
                                  np.where(np.array(gradeB_arrivals) < end_datetime))
    gradeC_match = np.intersect1d(np.where(np.array(gradeC_arrivals) > start_datetime),
                                  np.where(np.array(gradeC_arrivals) < end_datetime))
    gradeD_match = np.intersect1d(np.where(np.array(gradeD_arrivals) > start_datetime),
                                  np.where(np.array(gradeD_arrivals) < end_datetime))

    # Get the PGV, emergence, and arrival times from the catalog
    df = cat
    theta_var = df[df['geophone'] == 'geo1']['theta_variance'].values
    time_str_geo = df[df['geophone'] == instrument]['ft_arrival_time'].values

    misfit = df[df['geophone'] == 'geo1']['avg_misfit'].values
    theta_mean = df[df['geophone'] == 'geo1']['theta_mean'].values

    emerg_geo = df[df['geophone'] == instrument]['emergence'].values

    # Do the same thing for the PGV
    pgv_geo = df[df['geophone'] == instrument]['PGV'].values

    time_num_geo = []
    for timeind in np.arange(len(time_str_geo)):
        tg = dt.datetime.strptime(time_str_geo[timeind], "%Y-%m-%dT%H:%M:%S.%f")
        time_num_geo.append(tg)

    var_cutoff = 50
    num_events_pass = len(np.where(theta_var <= var_cutoff)[0])
    events_pass_ind = np.where(theta_var <= var_cutoff)[0]
    events_fail_ind = np.where(theta_var > var_cutoff)[0]


    theta_mean_pass = np.array(theta_mean)[events_pass_ind]
    theta_pass_values = [math.radians(val) for val in np.array(theta_mean)[events_pass_ind]]
    emerg_geo_pass = np.array(emerg_geo)[events_pass_ind]

    pgv_geo_pass = np.array(pgv_geo)[events_pass_ind]

    emerg_geo_fail = np.array(emerg_geo)[events_fail_ind]

    pgv_geo_fail = np.array(pgv_geo)[events_fail_ind]

    time_num_geo_pass = np.array(time_num_geo)[events_pass_ind]

    time_num_geo_fail = np.array(time_num_geo)[events_fail_ind]

    # Clip the emergence
    emergence_clip_thresh = 40

    emerg_geo_pass_clipped = clip_emerg_vals(emerg_geo_pass, emergence_clip_thresh)

    emerg_geo_fail_clipped = clip_emerg_vals(emerg_geo_fail, emergence_clip_thresh)

    # Clip the PGV
    pgv_clip_thresh = 400

    pgv_geo_pass_clipped = clip_emerg_vals(pgv_geo_pass, pgv_clip_thresh)

    pgv_geo_fail_clipped = clip_emerg_vals(pgv_geo_fail, pgv_clip_thresh)

    # Create a new colormap for this other plot
    color_norm2 = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(theta_mean_pass))
    cmap2 = cc.cm.colorwheel

    # Get the data and spectrogram
    st = read(infile)
    tr = st[0]
    tme = (tr.times("matplotlib") - tr.times("matplotlib")[0]) * 86400
    f, t, Sxx = signal.spectrogram(tr.data, tr.stats.sampling_rate)

    # Plot the figure
    fig = plt.figure(figsize=(20, 10), num=2, clear=True)
    ax0 = plt.subplot2grid((4, 5), (0, 0), colspan=4, rowspan=1)
    ax0.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    ax0.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    ax0.axvline(start_datetime, color='black')
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
    ax0b.scatter(time_num_geo_pass, emerg_geo_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo_fail, emerg_geo_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((4, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')

    # Plot the small segment of our current temperature
    ax2 = plt.subplot2grid((4, 5), (1, 0), colspan=5, rowspan=1)
    ax2.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    ax2.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax2.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    # ax2.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper right', fancybox=True,
    #            ncol=2)
    ax2.tick_params(axis='y', labelcolor=color0)
    ax2.set_xlim((start_datetime, end_datetime))
    ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color2b = 'tab:blue'
    ax2b.set_ylabel(f'Emergence {instrument}', color=color2b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax2b.scatter(time_num_geo_pass, emerg_geo_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax2b.scatter(time_num_geo_fail, emerg_geo_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)
    if len(mia_gradeA_match) > 0:
        for mia_gradeA_match_evid in mia_gradeA_match:
            ax2.axvline(mia_gradeA_arrivals[mia_gradeA_match_evid], c='red')

    if len(gradeB_match) > 0:
        for gradeB_match_evid in gradeB_match:
            ax2.axvline(gradeB_arrivals[gradeB_match_evid], c='purple')

    if len(gradeC_match) > 0:
        for gradeC_match_evid in gradeC_match:
            ax2.axvline(gradeC_arrivals[gradeC_match_evid], c='blue')

    if len(gradeD_match) > 0:
        for gradeD_match_evid in gradeD_match:
            ax2.axvline(gradeD_arrivals[gradeD_match_evid], c='gray')

    ax3 = plt.subplot2grid((4, 5), (2, 0), colspan=5, rowspan=1)
    ax3.plot(tme, tr.data, "b-")
    ax3.set_xlim((tme[0], tme[-1]))
    ax3.set_xlabel('Time')

    ax4 = plt.subplot2grid((4, 5), (3, 0), colspan=5, rowspan=1)
    if datatype == 'volts':
        ax3.set_ylabel('Amplitude (volts)', fontweight='bold')
        specmax = 1e-6
    elif datatype == 'phys':
        ax3.set_ylabel('Velocity (nm/s)', fontweight='bold')
        specmax = 60
    ax4.pcolormesh(t, f, Sxx, cmap=cm.jet, vmax=specmax, shading='auto')
    # ax4.pcolormesh(t, f, Sxx, cmap=cm.jet, shading='auto')
    ax4.set_xlim((tme[0], tme[-1]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f'{instrument}: {filestring}', fontweight='bold')
    fig.savefig(f'{outdir}{filestring}.png')
    fig.savefig(f'{outdir}EPS_{filestring}.eps')
    print(f'Finished saving {filestring}...')

    files_completed = glob.glob(f'{outdir}*.png')
    progress = str(np.round((len(files_completed) / num_hours) * 100, decimals=1))
    print(f'{progress}%...')

    return


def check_transition_wrapper(folder, start, end):
    """
    Wrapper file for plotting the data.

    :param folder: [str] path to the input directory where the files are held.
    :param start: [str] Beginning day for plotting in YYYY-MM-DD format.
    :param end: [str] Ending day for plotting in YYYY-MM-DD format.
    :return:
    """

    filelist = find_files(folder, start, end)
    filelist = [filelist[10]]
    if num_cores == 1:
        # Non-parallel version
        for infile in filelist:
            plot_hourly_trace(infile, len(filelist))
    else:
        # Parallel version
        Parallel(n_jobs=num_cores)(delayed(plot_hourly_trace)(infile, len(filelist))
                                   for infile in filelist)

    return


# Main
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
datadir = 'C:/data/lunar_data/'
out_dir = 'C:/data/lunar_output/'
# pkl_dir = f'{out_dir}results/locations/temporal_emergence/data/'

# Time-series datatype. Either physical units (nm/s) or output volts (volts). This will change labels as well as
# the specmax value
datatype = 'volts'
if datatype == 'volts':
    filedir = f'{datadir}LSPE_sac_hourly/'
elif datatype == 'phys':
    filedir = f'{datadir}LSPE_sac_hourly_phys/sac_data/'

# Select the number of CPU cores to use
num_cores = 1

# Get the time period for analysis
time_period_start = '1976-08-30'
time_period_end = '1976-08-31'

# Select the geophone to look at
instrument = 'geo1'

# Write output directories
outdir = f'{out_dir}results/temporal_spectrogram_comparison/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
outdir = f'{outdir}{time_period_start}_{time_period_end}/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Import the temperature and the good, located catalog
temperature_file = f'{rundir}longterm_thermal_data.txt'
starting_catalog = f'{out_dir}catalogs/GradeA_thermal_mq_catalog_final.csv'
temp_df = get_temperature(temperature_file)
cat = pd.read_csv(starting_catalog)

# Import the old detections catalog
ori_cat = f'{rundir}fc_lspe_dl_catalog.csv'
ori_cat_df = pd.read_csv(ori_cat)

# Find any Grade A events missed by our processing
gradeA_evids = np.unique(ori_cat_df[ori_cat_df['grade'] == 'A']['evid'].values)
mia_gradeA_evids = set(gradeA_evids).difference(np.unique(cat['evid'].values))
mia_gradeA_arrivals = []
for mia_gradeA_evid in mia_gradeA_evids:
    evid_ind = np.where(ori_cat_df['evid'] == mia_gradeA_evid)[0][0]
    mia_gradeA_arrivals.append(dt.datetime.strptime(ori_cat_df['abs_time'].values[evid_ind], '%Y-%m-%dT%H:%M:%S'))

# Get arrival times for the grade B, C, and D catalogs
gradeB_evids = np.unique(ori_cat_df[ori_cat_df['grade'] == 'B']['evid'].values)
gradeC_evids = np.unique(ori_cat_df[ori_cat_df['grade'] == 'C']['evid'].values)
gradeD_evids = np.unique(ori_cat_df[ori_cat_df['grade'] == 'D']['evid'].values)

gradeB_arrivals, gradeC_arrivals, gradeD_arrivals = [], [], []
for gradeB_evid in gradeB_evids:
    evid_ind = np.where(ori_cat_df['evid'] == gradeB_evid)[0][0]
    gradeB_arrivals.append(dt.datetime.strptime(ori_cat_df['abs_time'].values[evid_ind], '%Y-%m-%dT%H:%M:%S'))

for gradeC_evid in gradeC_evids:
    evid_ind = np.where(ori_cat_df['evid'] == gradeC_evid)[0][0]
    gradeC_arrivals.append(dt.datetime.strptime(ori_cat_df['abs_time'].values[evid_ind], '%Y-%m-%dT%H:%M:%S'))

for gradeD_evid in gradeD_evids:
    evid_ind = np.where(ori_cat_df['evid'] == gradeD_evid)[0][0]
    gradeD_arrivals.append(dt.datetime.strptime(ori_cat_df['abs_time'].values[evid_ind], '%Y-%m-%dT%H:%M:%S'))

# Find all the files that we will incorporate and run the plotting code
check_transition_wrapper(filedir, time_period_start, time_period_end)

