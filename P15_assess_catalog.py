"""
Do a statistical analysis of the moonquake catalog.

Computes both rose plots of the entire dataset as well as temporal plots
"""
# Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
import os
import datetime as dt
import csv
import colorcet as cc


# Functions
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


def norm_array(inarray):
    """
    Normalize a vector to values between 0 and 1

    :param inarray: [vector] Values to normalize
    :return:
    """

    array_normalized = []
    for val in inarray:
        array_normalized.append((val-np.min(inarray))/(np.max(inarray)-np.min(inarray)))

    return array_normalized


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


def avg(dates):
    """
    Gets the average of a list of datetime values
    :param dates: [vector] List of datetime values
    :return:
    """

    any_reference_date = dt.datetime(1900, 1, 1)
    return any_reference_date + sum([date - any_reference_date for date in dates], dt.timedelta()) / len(dates)

def assess_stats_noLM(df):
    """
        Computes time-series and rose plots of the Grade A seismic events without looking at the events that occur
        from the direction of the LM.

        :param df: [pd df] Pandas dataframe of the seismic catalog
        :return:
        """

    # NOTE: Something weird is happening to the next to last evid (770424-22-M2) for Geo 4.
    # For some reason, geo4 has a theta mean = 0, which kinda screws things up. The theta value is copied from
    # geo 3 for the time being.
    df = df.reset_index(drop=True)
    min_remove_theta = 80
    max_remove_theta = 120

    azi_remove_inds = np.intersect1d(np.where(df['theta_mean'].values > min_remove_theta)[0],
                                     np.where(df['theta_mean'].values < max_remove_theta)[0])
    df = df.drop(index=azi_remove_inds)
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

    fig = plt.figure(figsize=(8, 10))
    ax0 = plt.subplot(3, 1, 1)
    ax0.scatter(misfit, theta_var, edgecolor='black')
    # ax0.plot(indep, dep, color='red')
    ax0.set_ylabel('Azimuth Variance (degrees)', fontweight='bold')
    ax0.set_xlabel('misfit', fontweight='bold')
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    # ax0.set_title('Misfit with Variance', fontweight='bold')
    ax1 = plt.subplot(3, 1, 2)
    ax1.scatter(time_num_avg, theta_var, edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel('Evid Index', fontweight='bold')
    ax1.set_ylabel('Azimuth Variance (degrees)', fontweight='bold')
    # ax1.set_title('Azimuth Variance with Time', fontweight='bold')
    ax2 = plt.subplot(3, 1, 3)
    ax2.hist(theta_var, bins=20, range=(0, 100), edgecolor='black')
    ax2.set_xlabel('Azimuth Variance (degrees)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_folder_noLM}misfit_variance_noLM.png')
    fig.savefig(f'{output_folder_noLM}misfit_variance_noLM.eps')

    # Get the cutoff theta variance
    var_cutoff = 50
    num_events_pass = len(np.where(theta_var <= var_cutoff)[0])
    events_pass_ind = np.where(theta_var <= var_cutoff)[0]
    events_fail_ind = np.where(theta_var > var_cutoff)[0]

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

    theta_mean_pass = np.array(theta_mean)[events_pass_ind]
    theta_pass_values = [math.radians(val) for val in np.array(theta_mean)[events_pass_ind]]

    # Compute the parameters of the rose plot
    width = (2 * np.pi) / 18
    bins = np.arange(0, 2 * np.pi, width / 2)
    bins = np.append(bins, 1.999 * np.pi)
    theta = []
    theta_bin_emerg_vals = []
    theta_bin_emerg_vals_geo1, theta_bin_emerg_vals_geo2 = [], []
    theta_bin_emerg_vals_geo3, theta_bin_emerg_vals_geo4 = [], []
    theta_bin_pgv_vals = []
    theta_bin_pgv_vals_geo1, theta_bin_pgv_vals_geo2 = [], []
    theta_bin_pgv_vals_geo3, theta_bin_pgv_vals_geo4 = [], []
    for bin_ind in np.arange(len(bins) - 1):
        theta_val = ((bins[bin_ind + 1] - bins[bin_ind]) / 2) + bins[bin_ind]
        theta.append(theta_val)

        # Find the indices corresponding to the events within this bin
        theta_bin_indices = np.intersect1d(np.where(theta_pass_values < theta_val + width)[0],
                                           np.where(theta_pass_values >= theta_val)[0])

        # Now find the average emergence for this bin
        theta_bin_emerg_vals.append(np.mean(emerg_mean_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo1.append(np.mean(emerg_geo1_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo2.append(np.mean(emerg_geo2_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo3.append(np.mean(emerg_geo3_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo4.append(np.mean(emerg_geo4_pass[theta_bin_indices]))
        theta_bin_pgv_vals.append(np.mean(pgv_mean_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo1.append(np.mean(pgv_geo1_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo2.append(np.mean(pgv_geo2_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo3.append(np.mean(pgv_geo3_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo4.append(np.mean(pgv_geo4_pass[theta_bin_indices]))

    theta_pass_radii = compute_radii(theta_pass_values, bins)

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
    emergence_clip_thresh_plot = 40
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=emergence_clip_thresh_plot)
    cmap = plt.cm.jet_r

    # Set max for PGV color
    pgv_clip_thresh = 400
    pgv_clip_thresh_plot = 200
    color_norm3 = matplotlib.colors.Normalize(vmin=0, vmax=pgv_clip_thresh_plot)
    cmap3 = plt.cm.coolwarm

    fig2 = plt.figure(figsize=(14, 10))
    ax0 = plt.subplot(2, 2, 3)
    ax0.hist(emerg_mean_pass, bins=20, range=(0, 100), edgecolor='black')
    ax1 = plt.subplot(2, 2, 1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals)),
            edgecolor='black', alpha=0.7)
    ax1.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')

    ax1.set_title(f'Azimuth with variance < {var_cutoff} degrees Emergence',
                  fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    clb = fig2.colorbar(sm)
    clb.set_label('Average Emergence (s)', fontweight='bold')

    ax3 = plt.subplot(2, 2, 4)
    ax3.hist(pgv_mean_pass, bins=20, range=(0, 400), edgecolor='black')

    ax3 = plt.subplot(2, 2, 2, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals)),
            edgecolor='black', alpha=0.7)
    ax3.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax3.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    sm3 = plt.cm.ScalarMappable(cmap=cmap3, norm=color_norm3)
    clb3 = fig2.colorbar(sm3)
    clb3.set_label('Average PGV (s)', fontweight='bold')

    ax3.set_title(f'Azimuth with variance < {var_cutoff} degrees PGV',
                  fontweight='bold')

    fig2.tight_layout()
    fig2.savefig(f'{output_folder_noLM}avg_emer_pgv_azimuth_noLM.png')
    fig2.savefig(f'{output_folder_noLM}avg_emer_pgv_azimuth_noLM.eps')

    fig3 = plt.figure(figsize=(12, 12))
    ax0 = plt.subplot(2, 2, 1, polar=True)
    ax0.set_theta_direction(-1)
    ax0.set_theta_zero_location("N")
    ax0.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo1)),
            edgecolor='black', alpha=0.7)
    ax0.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax0.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax0.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo1 ({len(emerg_geo1_pass)} events)',
                  fontweight='bold')

    ax1 = plt.subplot(2, 2, 2, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo2)),
            edgecolor='black', alpha=0.7)
    ax1.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax1.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo2 ({len(emerg_geo2_pass)} events)',
                  fontweight='bold')

    ax2 = plt.subplot(2, 2, 3, polar=True)
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location("N")
    ax2.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo3)),
            edgecolor='black', alpha=0.7)
    ax2.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax2.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax2.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo3 ({len(emerg_geo3_pass)} events)',
                  fontweight='bold')

    ax3 = plt.subplot(2, 2, 4, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo4)),
            edgecolor='black', alpha=0.7)
    ax3.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax3.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax3.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo4 ({len(emerg_geo4_pass)} events)',
                  fontweight='bold')

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    # clb = fig3.colorbar(sm, orientation='horizontal')
    # clb.set_label('Average Emergence (s)', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(f'{output_folder_noLM}avg_emergence_with_azimuth_bysta_noLM.png')
    fig3.savefig(f'{output_folder_noLM}avg_emergence_with_azimuth_bysta_noLM.eps')

    fig33 = plt.figure(figsize=(12, 12))
    ax0 = plt.subplot(2, 2, 1, polar=True)
    ax0.set_theta_direction(-1)
    ax0.set_theta_zero_location("N")
    ax0.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo1)),
            edgecolor='black', alpha=0.7)
    ax0.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax0.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax0.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo1 ({len(emerg_geo1_pass)} events)',
                  fontweight='bold')

    ax1 = plt.subplot(2, 2, 2, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo2)),
            edgecolor='black', alpha=0.7)
    ax1.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax1.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo2 ({len(emerg_geo2_pass)} events)',
                  fontweight='bold')

    ax2 = plt.subplot(2, 2, 3, polar=True)
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location("N")
    ax2.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo3)),
            edgecolor='black', alpha=0.7)
    ax2.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax2.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax2.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo3 ({len(emerg_geo3_pass)} events)',
                  fontweight='bold')

    ax3 = plt.subplot(2, 2, 4, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo4)),
            edgecolor='black', alpha=0.7)
    ax3.plot([0, math.radians(min_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax3.plot([0, math.radians(max_remove_theta)], [0, np.max(theta_pass_radii)], color='black')
    ax3.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo4 ({len(emerg_geo4_pass)} events)',
                  fontweight='bold')

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    # clb = fig3.colorbar(sm, orientation='horizontal')
    # clb.set_label('Average Emergence (s)', fontweight='bold')
    fig33.tight_layout()
    fig33.savefig(f'{output_folder_noLM}avg_pgv_with_azimuth_bysta_noLM.png')
    fig33.savefig(f'{output_folder_noLM}avg_pgv_with_azimuth_bysta_noLM.eps')

    # ------------------------------------------------
    # Measure temporal changes
    temp_df = get_temperature(temperature_file)
    start_time = dt.datetime.strptime(str(np.min(temp_df['abs_time'].values))[0:-3], "%Y-%m-%dT%H:%M:%S.%f")
    end_time = dt.datetime.strptime(str(np.max(temp_df['abs_time'].values))[0:-3], "%Y-%m-%dT%H:%M:%S.%f")

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

    fig4 = plt.figure(figsize=(18, 5))
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

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig4.tight_layout()
    fig4.subplots_adjust(top=0.9)
    fig4.suptitle(f'Emergence Geo Mean: {var_cutoff} deg. var cutoff')
    fig4.savefig(f'{output_folder_noLM}emergence_mean_{var_cutoff}var_noLM.png')
    fig4.savefig(f'{output_folder_noLM}emergence_mean_{var_cutoff}var_noLM.eps')

    # Geo1
    fig51 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo1_pass, emerg_geo1_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo1_fail, emerg_geo1_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig51.tight_layout()
    fig51.subplots_adjust(top=0.9)
    fig51.suptitle(f'Emergence Geo1: {var_cutoff} deg. var cutoff')
    fig51.savefig(f'{output_folder_noLM}emergence_geo1_{var_cutoff}var_noLM.png')
    fig51.savefig(f'{output_folder_noLM}emergence_geo1_{var_cutoff}var_noLM.eps')

    # Geo2
    fig52 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo2_pass, emerg_geo2_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo2_fail, emerg_geo2_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig52.tight_layout()
    fig52.subplots_adjust(top=0.9)
    fig52.suptitle(f'Emergence Geo2: {var_cutoff} deg. var cutoff')
    fig52.savefig(f'{output_folder_noLM}emergence_geo2_{var_cutoff}var_noLM.png')
    fig52.savefig(f'{output_folder_noLM}emergence_geo2_{var_cutoff}var_noLM.eps')

    # Geo1
    fig53 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo3_pass, emerg_geo3_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo3_fail, emerg_geo3_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig53.tight_layout()
    fig53.subplots_adjust(top=0.9)
    fig53.suptitle(f'Emergence Geo3: {var_cutoff} deg. var cutoff')
    fig53.savefig(f'{output_folder_noLM}emergence_geo3_{var_cutoff}var_noLM.png')
    fig53.savefig(f'{output_folder_noLM}emergence_geo3_{var_cutoff}var_noLM.eps')

    # Geo4
    fig54 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo4_pass, emerg_geo4_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo4_fail, emerg_geo4_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig54.tight_layout()
    fig54.subplots_adjust(top=0.9)
    fig54.suptitle(f'Emergence Geo4: {var_cutoff} deg. var cutoff')
    fig54.savefig(f'{output_folder_noLM}emergence_geo4_{var_cutoff}var_noLM.png')
    fig54.savefig(f'{output_folder_noLM}emergence_geo4_{var_cutoff}var_noLM.eps')

    fig6 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Average PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_avg_pass, pgv_mean_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_avg_fail, pgv_mean_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig6.tight_layout()
    fig6.subplots_adjust(top=0.9)
    fig6.suptitle(f'PGV Geo Mean: {var_cutoff} deg. var cutoff')
    fig6.savefig(f'{output_folder_noLM}PGV_mean_{var_cutoff}var_noLM.png')
    fig6.savefig(f'{output_folder_noLM}PGV_mean_{var_cutoff}var_noLM.eps')

    # PGV geo1
    fig71 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo1_pass, pgv_geo1_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo1_fail, pgv_geo1_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig71.tight_layout()
    fig71.subplots_adjust(top=0.9)
    fig71.suptitle(f'PGV Geo 1: {var_cutoff} deg. var cutoff')
    fig71.savefig(f'{output_folder_noLM}PGV_geo1_{var_cutoff}var_noLM.png')
    fig71.savefig(f'{output_folder_noLM}PGV_geo1_{var_cutoff}var_noLM.eps')

    # Geo 2
    fig72 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo2_pass, pgv_geo2_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo2_fail, pgv_geo2_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig72.tight_layout()
    fig72.subplots_adjust(top=0.9)
    fig72.suptitle(f'PGV Geo 2: {var_cutoff} deg. var cutoff')
    fig72.savefig(f'{output_folder_noLM}PGV_geo2_{var_cutoff}var_noLM.png')
    fig72.savefig(f'{output_folder_noLM}PGV_geo2_{var_cutoff}var_noLM.eps')

    # Geo 3
    fig73 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo3_pass, pgv_geo3_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo3_fail, pgv_geo3_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig73.tight_layout()
    fig73.subplots_adjust(top=0.9)
    fig73.suptitle(f'PGV Geo 3: {var_cutoff} deg. var cutoff')
    fig73.savefig(f'{output_folder_noLM}PGV_geo3_{var_cutoff}var_noLM.png')
    fig73.savefig(f'{output_folder_noLM}PGV_geo3_{var_cutoff}var_noLM.eps')

    # Geo 4
    fig74 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo4_pass, pgv_geo4_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo4_fail, pgv_geo4_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.plot([0, math.radians(min_remove_theta)], [0, 1], color='black')
    ax1.plot([0, math.radians(max_remove_theta)], [0, 1], color='black')
    ax1.set_title('Incident Angle')
    fig74.tight_layout()
    fig74.subplots_adjust(top=0.9)
    fig74.suptitle(f'PGV Geo 4: {var_cutoff} deg. var cutoff')
    fig74.savefig(f'{output_folder_noLM}PGV_geo4_{var_cutoff}var_noLM.png')
    fig74.savefig(f'{output_folder_noLM}PGV_geo4_{var_cutoff}var_noLM.eps')
    return


def assess_stats(df):
    """
    Computes time-series and rose plots of the Grade A seismic events

    :param df: [pd df] Pandas dataframe of the seismic catalog
    :return:
    """

    df = df.reset_index(drop=True)

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

    fig = plt.figure(figsize=(8, 10))
    ax0 = plt.subplot(3, 1, 1)
    ax0.scatter(misfit, theta_var, edgecolor='black')
    # ax0.plot(indep, dep, color='red')
    ax0.set_ylabel('Azimuth Variance (degrees)', fontweight='bold')
    ax0.set_xlabel('misfit', fontweight='bold')
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    # ax0.set_title('Misfit with Variance', fontweight='bold')
    ax1 = plt.subplot(3, 1, 2)
    ax1.scatter(time_num_avg, theta_var, edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel('Evid Index', fontweight='bold')
    ax1.set_ylabel('Azimuth Variance (degrees)', fontweight='bold')
    # ax1.set_title('Azimuth Variance with Time', fontweight='bold')
    ax2 = plt.subplot(3, 1, 3)
    ax2.hist(theta_var, bins=20, range=(0, 100), edgecolor='black')
    ax2.set_xlabel('Azimuth Variance (degrees)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{output_folder}misfit_variance.png')
    fig.savefig(f'{output_folder}misfit_variance.eps')

    # Get the cutoff theta variance
    var_cutoff = 50
    num_events_pass = len(np.where(theta_var <= var_cutoff)[0])
    events_pass_ind = np.where(theta_var <= var_cutoff)[0]
    events_fail_ind = np.where(theta_var > var_cutoff)[0]

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

    theta_mean_pass = np.array(theta_mean)[events_pass_ind]
    theta_pass_values = [math.radians(val) for val in np.array(theta_mean)[events_pass_ind]]

    # Compute the parameters of the rose plot
    width = (2 * np.pi) / 18
    bins = np.arange(0, 2 * np.pi, width / 2)
    bins = np.append(bins, 1.999 * np.pi)
    theta = []
    theta_bin_emerg_vals = []
    theta_bin_emerg_vals_geo1, theta_bin_emerg_vals_geo2 = [], []
    theta_bin_emerg_vals_geo3, theta_bin_emerg_vals_geo4 = [], []
    theta_bin_pgv_vals = []
    theta_bin_pgv_vals_geo1, theta_bin_pgv_vals_geo2 = [], []
    theta_bin_pgv_vals_geo3, theta_bin_pgv_vals_geo4 = [], []
    for bin_ind in np.arange(len(bins) - 1):
        theta_val = ((bins[bin_ind + 1] - bins[bin_ind]) / 2) + bins[bin_ind]
        theta.append(theta_val)

        # Find the indices corresponding to the events within this bin
        theta_bin_indices = np.intersect1d(np.where(theta_pass_values < theta_val+width)[0],
                                           np.where(theta_pass_values >= theta_val)[0])

        # Now find the average emergence for this bin
        theta_bin_emerg_vals.append(np.mean(emerg_mean_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo1.append(np.mean(emerg_geo1_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo2.append(np.mean(emerg_geo2_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo3.append(np.mean(emerg_geo3_pass[theta_bin_indices]))
        theta_bin_emerg_vals_geo4.append(np.mean(emerg_geo4_pass[theta_bin_indices]))
        theta_bin_pgv_vals.append(np.mean(pgv_mean_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo1.append(np.mean(pgv_geo1_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo2.append(np.mean(pgv_geo2_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo3.append(np.mean(pgv_geo3_pass[theta_bin_indices]))
        theta_bin_pgv_vals_geo4.append(np.mean(pgv_geo4_pass[theta_bin_indices]))

    theta_pass_radii = compute_radii(theta_pass_values, bins)

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
    emergence_clip_thresh_plot = 40
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=emergence_clip_thresh_plot)
    cmap = plt.cm.jet_r

    # Set max for PGV color
    pgv_clip_thresh = 400
    pgv_clip_thresh_plot = 200
    color_norm3 = matplotlib.colors.Normalize(vmin=0, vmax=pgv_clip_thresh_plot)
    cmap3 = plt.cm.coolwarm

    # fig2 = plt.figure(figsize=(8, 12))
    # ax0 = plt.subplot(2, 1, 1)
    # ax0.hist(emerg_mean_pass, bins=20, range=(0, 100), edgecolor='black')
    # ax1 = plt.subplot(2, 1, 2, polar=True)
    # ax1.set_theta_direction(-1)
    # ax1.set_theta_zero_location("N")
    # # ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(emerg_mean_pass)),
    # #         edgecolor='black', alpha=0.7)
    # ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals)),
    #         edgecolor='black', alpha=0.7)
    #
    # ax0.set_title(f'Azimuth with variance < {var_cutoff} degrees ({num_events_pass} events)',
    #               fontweight='bold')
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    # clb = fig2.colorbar(sm)
    # clb.set_label('Average Emergence (s)', fontweight='bold')
    # fig2.tight_layout()
    # fig2.savefig(f'{output_folder}avg_emergence_with_azimuth.png')
    # fig2.savefig(f'{output_folder}avg_emergence_with_azimuth.eps')

    fig2 = plt.figure(figsize=(14, 10))
    ax0 = plt.subplot(2, 2, 3)
    ax0.hist(emerg_mean_pass, bins=20, range=(0, 100), edgecolor='black')
    ax1 = plt.subplot(2, 2, 1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals)),
            edgecolor='black', alpha=0.7)

    ax1.set_title(f'Azimuth with variance < {var_cutoff} degrees Emergence',
                  fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    clb = fig2.colorbar(sm)
    clb.set_label('Average Emergence (s)', fontweight='bold')

    ax3 = plt.subplot(2, 2, 4)
    ax3.hist(pgv_mean_pass, bins=20, range=(0, 400), edgecolor='black')

    ax3 = plt.subplot(2, 2, 2, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals)),
            edgecolor='black', alpha=0.7)
    sm3 = plt.cm.ScalarMappable(cmap=cmap3, norm=color_norm3)
    clb3 = fig2.colorbar(sm3)
    clb3.set_label('Average PGV (s)', fontweight='bold')

    ax3.set_title(f'Azimuth with variance < {var_cutoff} degrees PGV',
                  fontweight='bold')

    fig2.tight_layout()
    fig2.savefig(f'{output_folder}avg_emer_pgv_azimuth.png')
    fig2.savefig(f'{output_folder}avg_emer_pgv_azimuth.eps')

    fig3 = plt.figure(figsize=(12, 12))
    ax0 = plt.subplot(2, 2, 1, polar=True)
    ax0.set_theta_direction(-1)
    ax0.set_theta_zero_location("N")
    ax0.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo1)),
            edgecolor='black', alpha=0.7)
    ax0.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo1 ({len(emerg_geo1_pass)} events)',
                  fontweight='bold')

    ax1 = plt.subplot(2, 2, 2, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo2)),
            edgecolor='black', alpha=0.7)
    ax1.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo2 ({len(emerg_geo2_pass)} events)',
                  fontweight='bold')

    ax2 = plt.subplot(2, 2, 3, polar=True)
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location("N")
    ax2.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo3)),
            edgecolor='black', alpha=0.7)
    ax2.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo3 ({len(emerg_geo3_pass)} events)',
                  fontweight='bold')

    ax3 = plt.subplot(2, 2, 4, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap(color_norm(theta_bin_emerg_vals_geo4)),
            edgecolor='black', alpha=0.7)
    ax3.set_title(f'Emergence Azimuth Var < {var_cutoff} deg.: Geo4 ({len(emerg_geo4_pass)} events)',
                  fontweight='bold')

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    # clb = fig3.colorbar(sm, orientation='horizontal')
    # clb.set_label('Average Emergence (s)', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(f'{output_folder}avg_emergence_with_azimuth_bysta.png')
    fig3.savefig(f'{output_folder}avg_emergence_with_azimuth_bysta.eps')

    fig33 = plt.figure(figsize=(12, 12))
    ax0 = plt.subplot(2, 2, 1, polar=True)
    ax0.set_theta_direction(-1)
    ax0.set_theta_zero_location("N")
    ax0.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo1)),
            edgecolor='black', alpha=0.7)
    ax0.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo1 ({len(emerg_geo1_pass)} events)',
                  fontweight='bold')

    ax1 = plt.subplot(2, 2, 2, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo2)),
            edgecolor='black', alpha=0.7)
    ax1.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo2 ({len(emerg_geo2_pass)} events)',
                  fontweight='bold')

    ax2 = plt.subplot(2, 2, 3, polar=True)
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location("N")
    ax2.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo3)),
            edgecolor='black', alpha=0.7)
    ax2.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo3 ({len(emerg_geo3_pass)} events)',
                  fontweight='bold')

    ax3 = plt.subplot(2, 2, 4, polar=True)
    ax3.set_theta_direction(-1)
    ax3.set_theta_zero_location("N")
    ax3.bar(theta, theta_pass_radii, width=width, color=cmap3(color_norm3(theta_bin_pgv_vals_geo4)),
            edgecolor='black', alpha=0.7)
    ax3.set_title(f'PGV Azimuth Var < {var_cutoff} deg.: Geo4 ({len(emerg_geo4_pass)} events)',
                  fontweight='bold')

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    # clb = fig3.colorbar(sm, orientation='horizontal')
    # clb.set_label('Average Emergence (s)', fontweight='bold')
    fig33.tight_layout()
    fig33.savefig(f'{output_folder}avg_pgv_with_azimuth_bysta.png')
    fig33.savefig(f'{output_folder}avg_pgv_with_azimuth_bysta.eps')

    # ------------------------------------------------
    # Measure temporal changes
    temp_df = get_temperature(temperature_file)
    start_time = dt.datetime.strptime(str(np.min(temp_df['abs_time'].values))[0:-3], "%Y-%m-%dT%H:%M:%S.%f")
    end_time = dt.datetime.strptime(str(np.max(temp_df['abs_time'].values))[0:-3], "%Y-%m-%dT%H:%M:%S.%f")

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

    fig4 = plt.figure(figsize=(18, 5))
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
    fig4.suptitle(f'Emergence Geo Mean: {var_cutoff} deg. var cutoff')
    fig4.savefig(f'{output_folder}emergence_mean_{var_cutoff}var.png')
    fig4.savefig(f'{output_folder}emergence_mean_{var_cutoff}var.eps')

    # Geo1
    fig51 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo1_pass, emerg_geo1_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo1_fail, emerg_geo1_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig51.tight_layout()
    fig51.subplots_adjust(top=0.9)
    fig51.suptitle(f'Emergence Geo1: {var_cutoff} deg. var cutoff')
    fig51.savefig(f'{output_folder}emergence_geo1_{var_cutoff}var.png')
    fig51.savefig(f'{output_folder}emergence_geo1_{var_cutoff}var.eps')

    # Geo2
    fig52 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo2_pass, emerg_geo2_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo2_fail, emerg_geo2_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig52.tight_layout()
    fig52.subplots_adjust(top=0.9)
    fig52.suptitle(f'Emergence Geo2: {var_cutoff} deg. var cutoff')
    fig52.savefig(f'{output_folder}emergence_geo2_{var_cutoff}var.png')
    fig52.savefig(f'{output_folder}emergence_geo2_{var_cutoff}var.eps')

    # Geo1
    fig53 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo3_pass, emerg_geo3_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo3_fail, emerg_geo3_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig53.tight_layout()
    fig53.subplots_adjust(top=0.9)
    fig53.suptitle(f'Emergence Geo3: {var_cutoff} deg. var cutoff')
    fig53.savefig(f'{output_folder}emergence_geo3_{var_cutoff}var.png')
    fig53.savefig(f'{output_folder}emergence_geo3_{var_cutoff}var.eps')

    # Geo4
    fig54 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Emergence', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo4_pass, emerg_geo4_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo4_fail, emerg_geo4_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig54.tight_layout()
    fig54.subplots_adjust(top=0.9)
    fig54.suptitle(f'Emergence Geo4: {var_cutoff} deg. var cutoff')
    fig54.savefig(f'{output_folder}emergence_geo4_{var_cutoff}var.png')
    fig54.savefig(f'{output_folder}emergence_geo4_{var_cutoff}var.eps')

    fig6 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('Average PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_avg_pass, pgv_mean_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_avg_fail, pgv_mean_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig6.tight_layout()
    fig6.subplots_adjust(top=0.9)
    fig6.suptitle(f'PGV Geo Mean: {var_cutoff} deg. var cutoff')
    fig6.savefig(f'{output_folder}PGV_mean_{var_cutoff}var.png')
    fig6.savefig(f'{output_folder}PGV_mean_{var_cutoff}var.eps')

    # PGV geo1
    fig71 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo1_pass, pgv_geo1_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo1_fail, pgv_geo1_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig71.tight_layout()
    fig71.subplots_adjust(top=0.9)
    fig71.suptitle(f'PGV Geo 1: {var_cutoff} deg. var cutoff')
    fig71.savefig(f'{output_folder}PGV_geo1_{var_cutoff}var.png')
    fig71.savefig(f'{output_folder}PGV_geo1_{var_cutoff}var.eps')

    # Geo 2
    fig72 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo2_pass, pgv_geo2_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo2_fail, pgv_geo2_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig72.tight_layout()
    fig72.subplots_adjust(top=0.9)
    fig72.suptitle(f'PGV Geo 2: {var_cutoff} deg. var cutoff')
    fig72.savefig(f'{output_folder}PGV_geo2_{var_cutoff}var.png')
    fig72.savefig(f'{output_folder}PGV_geo2_{var_cutoff}var.eps')

    # Geo 3
    fig73 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo3_pass, pgv_geo3_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo3_fail, pgv_geo3_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig73.tight_layout()
    fig73.subplots_adjust(top=0.9)
    fig73.suptitle(f'PGV Geo 3: {var_cutoff} deg. var cutoff')
    fig73.savefig(f'{output_folder}PGV_geo3_{var_cutoff}var.png')
    fig73.savefig(f'{output_folder}PGV_geo3_{var_cutoff}var.eps')

    # Geo 4
    fig74 = plt.figure(figsize=(18, 5))
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
    ax0b.set_ylabel('PGV', color=color0b)
    # ax0b.bar(bin_midval_num, bin_total_evts, width=bin_width / (60 * 60 * 24))
    ax0b.scatter(time_num_geo4_pass, pgv_geo4_pass_clipped,
                 color=cmap2(color_norm2(theta_mean_pass)), edgecolor='black', linewidth=0.5)
    ax0b.scatter(time_num_geo4_fail, pgv_geo4_fail_clipped,
                 color='gray', edgecolor='black', linewidth=0.5)

    ax1 = plt.subplot2grid((1, 5), (0, 4), colspan=1, rowspan=1, polar=True)
    ax1.set_theta_direction(-1)
    ax1.set_ylim((0, 1))
    ax1.axes.yaxis.set_visible(False)
    ax1.set_theta_zero_location("N")

    for theta_plot in np.arange(0, 360):
        ax1.plot([0, math.radians(theta_plot)], [0, 1], color=cmap2(color_norm2(theta_plot)))
    ax1.set_title('Incident Angle')
    fig74.tight_layout()
    fig74.subplots_adjust(top=0.9)
    fig74.suptitle(f'PGV Geo 4: {var_cutoff} deg. var cutoff')
    fig74.savefig(f'{output_folder}PGV_geo4_{var_cutoff}var.png')
    fig74.savefig(f'{output_folder}PGV_geo4_{var_cutoff}var.eps')
    return


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


# Main
outdir = 'C:/data/lunar_output/'
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
catalog_file = f'{outdir}catalogs/GradeA_thermal_mq_catalog_final.csv'
temperature_file = f'{rundir}longterm_thermal_data.txt'

# Create output directories
output_folder = f'{outdir}results/thermal_location_results/'
output_folder_noLM = f'{outdir}results/thermal_location_results_noLM/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(output_folder_noLM):
    os.mkdir(output_folder_noLM)

# Read the catalog file
cat_df = pd.read_csv(catalog_file)

# Remove problematic events from the catalog
evids_to_exclude = ['770425-00-M1', '761111-21-M2', '770416-10-M1', '770325-13-M6', '770114-15-M1',
                    '761105-08-M1', '770314-22-M1', '761021-07-M1', '760901-14-M2']
cat_df = remove_evts_from_cat(cat_df, evids_to_exclude)

assess_stats(cat_df)

assess_stats_noLM(cat_df)


