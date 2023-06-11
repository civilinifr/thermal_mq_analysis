"""
Assesses the moonquake start times picked in P18A
"""
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import csv
import os


def find_nearest_index(array, value):
    """
    Find nearest index corresponding to a value in an array
    :param array: [np array] Array that we are searching
    :param value: [float] Value that we're searching for
    :return:
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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


def get_param_values(arrival_times_num_total, runsum_total, arrival_times_num_onlylm, runsum_onlylm,
                     arrival_times_num_nolm, runsum_nolm, burst1_start_num, burst1_end_num, burst2_start_num,
                     burst2_end_num):
    """
    Get parameter values for all three datasets

    :param arrival_times_num_total: [vector] Arrival times for all data
    :param runsum_total: [vector] Running sum for all data
    :param arrival_times_num_onlylm: [vector] Arrival times for only LM events
    :param runsum_onlylm: [vector] Running sum for only LM events
    :param arrival_times_num_nolm: [vector] Arrival times for only non-LM events
    :param runsum_nolm: [vector] Running sum for only non-LM events
    :param burst1_start_num: [datetime] Start of burst 1
    :param burst1_end_num: [datetime] End of burst 1
    :param burst2_start_num: [datetime] Start of burst 2
    :param burst2_end_num: [datetime] End of burst 2
    :return:
    """

    burst_diff_hrs = ((burst2_start_num - burst1_end_num).seconds +
                      (burst2_start_num - burst1_end_num).microseconds / 10 ** 6) / (60 * 60)
    burst_diff_hrs = np.round(burst_diff_hrs, decimals=1)

    cycle_labels = ['duration_b1(hrs)', 'duration_b2(hrs)', 'num_mq_b1', 'num_mq_b2', 'rate_b1(num/hr)',
                    'rate_b2(num/hr)']

    # ---------
    # Total
    burst1_start_total_ind = find_nearest_index(arrival_times_num_total, burst1_start_num)
    burst1_end_total_ind = find_nearest_index(arrival_times_num_total, burst1_end_num)
    burst2_start_total_ind = find_nearest_index(arrival_times_num_total, burst2_start_num)
    burst2_end_total_ind = find_nearest_index(arrival_times_num_total, burst2_end_num)
    burst1_start_total_time = arrival_times_num_total[burst1_start_total_ind]
    burst1_end_total_time = arrival_times_num_total[burst1_end_total_ind]
    burst2_start_total_time = arrival_times_num_total[burst2_start_total_ind]
    burst2_end_total_time = arrival_times_num_total[burst2_end_total_ind]

    # Duration in hours
    burst1_total_duration_hrs = ((burst1_end_total_time - burst1_start_total_time).seconds +
                                 ((burst1_end_total_time - burst1_start_total_time).microseconds / 10 ** 6)) / (60 * 60)
    burst2_total_duration_hrs = ((burst2_end_total_time - burst2_start_total_time).seconds +
                                 ((burst2_end_total_time - burst2_start_total_time).microseconds / 10 ** 6)) / (60 * 60)
    burst1_total_duration_hrs = np.round(burst1_total_duration_hrs, decimals=1)
    burst2_total_duration_hrs = np.round(burst2_total_duration_hrs, decimals=1)

    # Number of mq before and after
    burst1_start_total_num = runsum_total[burst1_start_total_ind]
    burst1_end_total_num = runsum_total[burst1_end_total_ind]
    burst2_start_total_num = runsum_total[burst2_start_total_ind]
    burst2_end_total_num = runsum_total[burst2_end_total_ind]
    burst1_total_mqnum = burst1_end_total_num - burst1_start_total_num
    burst2_total_mqnum = burst2_end_total_num - burst2_start_total_num

    # Get the MQ rate for each burst
    burst1_total_rate_phr = np.round(burst1_total_mqnum / burst1_total_duration_hrs, decimals=1)
    burst2_total_rate_phr = np.round(burst2_total_mqnum / burst2_total_duration_hrs, decimals=1)

    cycle_vals_total = [burst1_total_duration_hrs, burst2_total_duration_hrs, burst1_total_mqnum, burst2_total_mqnum,
                        burst1_total_rate_phr, burst2_total_rate_phr]

    cycle_vals_total_nums = [burst1_start_total_num, burst1_end_total_num, burst2_start_total_num, burst2_end_total_num]

    # ---------
    # onlylm
    burst1_start_onlylm_ind = find_nearest_index(arrival_times_num_onlylm, burst1_start_num)
    burst1_end_onlylm_ind = find_nearest_index(arrival_times_num_onlylm, burst1_end_num)
    burst2_start_onlylm_ind = find_nearest_index(arrival_times_num_onlylm, burst2_start_num)
    burst2_end_onlylm_ind = find_nearest_index(arrival_times_num_onlylm, burst2_end_num)
    burst1_start_onlylm_time = arrival_times_num_onlylm[burst1_start_onlylm_ind]
    burst1_end_onlylm_time = arrival_times_num_onlylm[burst1_end_onlylm_ind]
    burst2_start_onlylm_time = arrival_times_num_onlylm[burst2_start_onlylm_ind]
    burst2_end_onlylm_time = arrival_times_num_onlylm[burst2_end_onlylm_ind]

    # Duration in hours
    burst1_onlylm_duration_hrs = ((burst1_end_onlylm_time - burst1_start_onlylm_time).seconds +
                                  ((burst1_end_onlylm_time - burst1_start_onlylm_time).microseconds / 10 ** 6)) / (
                                         60 * 60)
    burst2_onlylm_duration_hrs = ((burst2_end_onlylm_time - burst2_start_onlylm_time).seconds +
                                  ((burst2_end_onlylm_time - burst2_start_onlylm_time).microseconds / 10 ** 6)) / (
                                         60 * 60)
    burst1_onlylm_duration_hrs = np.round(burst1_onlylm_duration_hrs, decimals=1)
    burst2_onlylm_duration_hrs = np.round(burst2_onlylm_duration_hrs, decimals=1)

    # Number of mq before and after
    burst1_start_onlylm_num = runsum_onlylm[burst1_start_onlylm_ind]
    burst1_end_onlylm_num = runsum_onlylm[burst1_end_onlylm_ind]
    burst2_start_onlylm_num = runsum_onlylm[burst2_start_onlylm_ind]
    burst2_end_onlylm_num = runsum_onlylm[burst2_end_onlylm_ind]
    burst1_onlylm_mqnum = burst1_end_onlylm_num - burst1_start_onlylm_num
    burst2_onlylm_mqnum = burst2_end_onlylm_num - burst2_start_onlylm_num

    # Get the MQ rate for each burst
    burst1_onlylm_rate_phr = np.round(burst1_onlylm_mqnum / burst1_onlylm_duration_hrs, decimals=1)
    burst2_onlylm_rate_phr = np.round(burst2_onlylm_mqnum / burst2_onlylm_duration_hrs, decimals=1)

    cycle_vals_onlylm = [burst1_onlylm_duration_hrs, burst2_onlylm_duration_hrs, burst1_onlylm_mqnum,
                         burst2_onlylm_mqnum,
                         burst1_onlylm_rate_phr, burst2_onlylm_rate_phr]

    cycle_vals_onlylm_nums = [burst1_start_onlylm_num, burst1_end_onlylm_num,
                              burst2_start_onlylm_num, burst2_end_onlylm_num]

    # ---------
    # nolm
    burst1_start_nolm_ind = find_nearest_index(arrival_times_num_nolm, burst1_start_num)
    burst1_end_nolm_ind = find_nearest_index(arrival_times_num_nolm, burst1_end_num)
    burst2_start_nolm_ind = find_nearest_index(arrival_times_num_nolm, burst2_start_num)
    burst2_end_nolm_ind = find_nearest_index(arrival_times_num_nolm, burst2_end_num)
    burst1_start_nolm_time = arrival_times_num_nolm[burst1_start_nolm_ind]
    burst1_end_nolm_time = arrival_times_num_nolm[burst1_end_nolm_ind]
    burst2_start_nolm_time = arrival_times_num_nolm[burst2_start_nolm_ind]
    burst2_end_nolm_time = arrival_times_num_nolm[burst2_end_nolm_ind]

    # Duration in hours
    burst1_nolm_duration_hrs = ((burst1_end_nolm_time - burst1_start_nolm_time).seconds +
                                ((burst1_end_nolm_time - burst1_start_nolm_time).microseconds / 10 ** 6)) / (60 * 60)
    burst2_nolm_duration_hrs = ((burst2_end_nolm_time - burst2_start_nolm_time).seconds +
                                ((burst2_end_nolm_time - burst2_start_nolm_time).microseconds / 10 ** 6)) / (60 * 60)
    burst1_nolm_duration_hrs = np.round(burst1_nolm_duration_hrs, decimals=1)
    burst2_nolm_duration_hrs = np.round(burst2_nolm_duration_hrs, decimals=1)

    # Number of mq before and after
    burst1_start_nolm_num = runsum_nolm[burst1_start_nolm_ind]
    burst1_end_nolm_num = runsum_nolm[burst1_end_nolm_ind]
    burst2_start_nolm_num = runsum_nolm[burst2_start_nolm_ind]
    burst2_end_nolm_num = runsum_nolm[burst2_end_nolm_ind]
    burst1_nolm_mqnum = burst1_end_nolm_num - burst1_start_nolm_num
    burst2_nolm_mqnum = burst2_end_nolm_num - burst2_start_nolm_num

    # Get the MQ rate for each burst
    burst1_nolm_rate_phr = np.round(burst1_nolm_mqnum / burst1_nolm_duration_hrs, decimals=1)
    burst2_nolm_rate_phr = np.round(burst2_nolm_mqnum / burst2_nolm_duration_hrs, decimals=1)

    cycle_vals_nolm = [burst1_nolm_duration_hrs, burst2_nolm_duration_hrs, burst1_nolm_mqnum, burst2_nolm_mqnum,
                       burst1_nolm_rate_phr, burst2_nolm_rate_phr]

    cycle_vals_nolm_nums = [burst1_start_nolm_num, burst1_end_nolm_num,
                            burst2_start_nolm_num, burst2_end_nolm_num]

    return burst_diff_hrs, cycle_labels, cycle_vals_total, cycle_vals_onlylm, cycle_vals_nolm, cycle_vals_total_nums, \
           cycle_vals_onlylm_nums, cycle_vals_nolm_nums


def main():
    """
    Main wrapper function

    :return:
    """
    rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
    outdir = 'C:/data/lunar_output/'
    catalog_file = f'{outdir}catalogs/GradeA_thermal_mq_catalog_final.csv'
    temperature_file = f'{rundir}longterm_thermal_data.txt'
    output_directory = f'{outdir}results/sunrise_burst_analysis/'
    nightfall_file = f'{rundir}nightfall_burst.txt'
    sunrise_file = f'{rundir}sunrise_burst.txt'

    if not os.path.exists(f'{output_directory}detailed_analysis/'):
        os.mkdir(f'{output_directory}detailed_analysis/')
    if not os.path.exists(f'{output_directory}detailed_analysis/plots/'):
        os.mkdir(f'{output_directory}detailed_analysis/plots/')
    if not os.path.exists(f'{output_directory}detailed_analysis/data/'):
        os.mkdir(f'{output_directory}detailed_analysis/data/')

    cat_df = pd.read_csv(catalog_file)

    temp_df = get_temperature(temperature_file)

    nightfall_df = pd.read_csv(nightfall_file, header=None)
    nightfall_num = []
    for nightfall_val in nightfall_df[0].values:
        nightfall_num.append(dt.datetime.strptime(nightfall_val, "%Y-%m-%dT%H:%M:%S.%f"))

    sunrise_df = pd.read_csv(sunrise_file, header=None)
    sunrise_num = []
    for sunrise_val in sunrise_df[0].values:
        sunrise_num.append(dt.datetime.strptime(sunrise_val, "%Y-%m-%dT%H:%M:%S.%f"))

    stations = ['geo1', 'geo2', 'geo3', 'geo4']
    for sta in stations:
        cat_sta = cat_df[cat_df['geophone'] == sta]
        cat_sta = cat_sta.sort_values(by=['ft_arrival_time'])
        df_onlylm = cat_sta[(cat_sta['theta_mean'] > 80) & (cat_sta['theta_mean'] < 120)]
        df_nolm = cat_sta[(cat_sta['theta_mean'] < 80) | (cat_sta['theta_mean'] > 120)]

        runsum_total = np.arange(len(cat_sta)) + 1
        runsum_onlylm = np.arange(len(df_onlylm)) + 1
        runsum_nolm = np.arange(len(df_nolm)) + 1

        # Convert ft arrival times into datetime values
        arrival_times_num_total = []
        for arrival_val in cat_sta['ft_arrival_time'].values:
            arrival_times_num_total.append(dt.datetime.strptime(arrival_val, "%Y-%m-%dT%H:%M:%S.%f"))

        arrival_times_num_onlylm = []
        for arrival_val in df_onlylm['ft_arrival_time'].values:
            arrival_times_num_onlylm.append(dt.datetime.strptime(arrival_val, "%Y-%m-%dT%H:%M:%S.%f"))

        arrival_times_num_nolm = []
        for arrival_val in df_nolm['ft_arrival_time'].values:
            arrival_times_num_nolm.append(dt.datetime.strptime(arrival_val, "%Y-%m-%dT%H:%M:%S.%f"))

        # Choose the end of the sunrise burst in seconds
        sunrise_end_hours = 48
        sunrise_end_time = int(sunrise_end_hours * 60 * 60)

        # Convert the temperature time to datetime
        temp_time_str = temp_df['abs_time'].values.astype('str')
        temp_time_num = []
        for time_str in temp_time_str:
            temp_time_num.append(dt.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f000"))

        # Plot all the sunrise values
        sunlight_cycle = 1
        # The time difference between the bursts should be the same across all stations
        burst_diff_hrs_vector = []
        cycle_vals_total_vector, cycle_vals_onlylm_vector, cycle_vals_nolm_vector = [], [], []
        for sunrise_val in sunrise_num:
            # Find the file associated with this particular sunrise cycle
            cycle_df = pd.read_csv(f'{output_directory}cycle{sunlight_cycle}_burst_times.txt', header=None)

            burst1_start_num = dt.datetime(1970, 1, 1) + dt.timedelta(days=cycle_df[0].values[0])
            burst1_end_num = dt.datetime(1970, 1, 1) + dt.timedelta(days=cycle_df[0].values[1])
            burst2_start_num = dt.datetime(1970, 1, 1) + dt.timedelta(days=cycle_df[0].values[2])
            burst2_end_num = dt.datetime(1970, 1, 1) + dt.timedelta(days=cycle_df[0].values[3])

            # Get parameter values for all three datasets
            burst_diff_hrs, cycle_labels, cycle_vals_total, cycle_vals_onlylm, cycle_vals_nolm, cycle_vals_total_nums, \
            cycle_vals_onlylm_nums, cycle_vals_nolm_nums = \
                get_param_values(arrival_times_num_total, runsum_total, arrival_times_num_onlylm, runsum_onlylm,
                                 arrival_times_num_nolm, runsum_nolm, burst1_start_num, burst1_end_num,
                                 burst2_start_num, burst2_end_num)

            # Add the values to a vector for each cycle
            burst_diff_hrs_vector.append(burst_diff_hrs)
            cycle_vals_total_vector.append(cycle_vals_total)
            cycle_vals_onlylm_vector.append(cycle_vals_onlylm)
            cycle_vals_nolm_vector.append(cycle_vals_nolm)

            sunrise_end = sunrise_val + dt.timedelta(seconds=sunrise_end_time)

            fig = plt.figure(figsize=(18, 12), num=2, clear=True)

            # Total MQ
            sunrise_val_ind = find_nearest_index(arrival_times_num_total, sunrise_val)
            sunrise_end_ind = find_nearest_index(arrival_times_num_total, sunrise_end) + 1
            sunrise_max = np.max(runsum_total[sunrise_val_ind:sunrise_end_ind])
            sunrise_min = np.min(runsum_total[sunrise_val_ind:sunrise_end_ind])

            ax0 = plt.subplot2grid(shape=(3, 6), loc=(0, 0), rowspan=1, colspan=4)
            ax0.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax0.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax0.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax0.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True,
                       ncol=2)
            ax0.tick_params(axis='y', labelcolor=color0)
            ax0.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax0b = ax0.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax0b.plot(arrival_times_num_total, runsum_total)
            ax0b.set_ylabel(f'Total Cumulative Sum', fontweight='bold', color=color0b)
            ax0.axvspan(xmin=sunrise_val, xmax=sunrise_end, color='blue', alpha=0.3)

            ax1 = plt.subplot2grid(shape=(3, 6), loc=(0, 4), rowspan=1, colspan=2)
            ax1.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax1.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax1.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax1.tick_params(axis='y', labelcolor=color0)
            ax1.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax1b = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax1b.plot(arrival_times_num_total, runsum_total)
            ax1b.scatter([burst1_start_num, burst1_end_num],
                         cycle_vals_total_nums[0:2], marker='o', c='magenta', edgecolor='k')
            ax1b.scatter([burst2_start_num, burst2_end_num],
                         cycle_vals_total_nums[2:], marker='o', c='gray', edgecolor='k')
            ax1b.set_ylabel(f'Total Cumulative Sum', fontweight='bold', color=color0b)
            ax1.set_xlim((sunrise_val, sunrise_end))
            ax1b.set_xlim((sunrise_val, sunrise_end))
            ax1b.set_ylim((sunrise_min, sunrise_max))
            ax1.set_xticklabels(ax1.get_xticks(), rotation=45)

            # Only LM
            sunrise_val_ind = find_nearest_index(arrival_times_num_onlylm, sunrise_val)
            sunrise_end_ind = find_nearest_index(arrival_times_num_onlylm, sunrise_end) + 1
            sunrise_max = np.max(runsum_onlylm[sunrise_val_ind:sunrise_end_ind])
            sunrise_min = np.min(runsum_onlylm[sunrise_val_ind:sunrise_end_ind])

            ax2 = plt.subplot2grid(shape=(3, 6), loc=(1, 0), rowspan=1, colspan=4)
            ax2.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax2.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax2.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax2.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True,
                       ncol=2)
            ax2.tick_params(axis='y', labelcolor=color0)
            ax2.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax2b.plot(arrival_times_num_onlylm, runsum_onlylm)
            ax2b.set_ylabel(f'Only LM Cumulative Sum', fontweight='bold', color=color0b)
            ax2.axvspan(xmin=sunrise_val, xmax=sunrise_end, color='blue', alpha=0.3)

            ax3 = plt.subplot2grid(shape=(3, 6), loc=(1, 4), rowspan=1, colspan=2)
            ax3.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax3.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax3.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax3.tick_params(axis='y', labelcolor=color0)
            ax3.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax3b = ax3.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax3b.plot(arrival_times_num_onlylm, runsum_onlylm)
            ax3b.scatter([burst1_start_num, burst1_end_num],
                         cycle_vals_onlylm_nums[0:2], marker='o', c='magenta', edgecolor='k')
            ax3b.scatter([burst2_start_num, burst2_end_num],
                         cycle_vals_onlylm_nums[2:], marker='o', c='gray', edgecolor='k')
            ax3b.set_ylabel(f'Only LM Cumulative Sum', fontweight='bold', color=color0b)
            ax3.set_xlim((sunrise_val, sunrise_end))
            ax3b.set_xlim((sunrise_val, sunrise_end))
            ax3b.set_ylim((sunrise_min, sunrise_max))
            ax3.set_xticklabels(ax3.get_xticks(), rotation=45)

            # No LM
            sunrise_val_ind = find_nearest_index(arrival_times_num_nolm, sunrise_val)
            sunrise_end_ind = find_nearest_index(arrival_times_num_nolm, sunrise_end) + 1
            sunrise_max = np.max(runsum_nolm[sunrise_val_ind:sunrise_end_ind])
            sunrise_min = np.min(runsum_nolm[sunrise_val_ind:sunrise_end_ind])

            ax4 = plt.subplot2grid(shape=(3, 6), loc=(2, 0), rowspan=1, colspan=4)
            ax4.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax4.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax4.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax4.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True,
                       ncol=2)
            ax4.tick_params(axis='y', labelcolor=color0)
            ax4.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax4b = ax4.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax4b.plot(arrival_times_num_nolm, runsum_nolm)
            ax4b.set_ylabel(f'No LM Cumulative Sum', fontweight='bold', color=color0b)
            ax4.axvspan(xmin=sunrise_val, xmax=sunrise_end, color='blue', alpha=0.3)

            ax5 = plt.subplot2grid(shape=(3, 6), loc=(2, 4), rowspan=1, colspan=2)
            ax5.plot(temp_time_num, temp_df['temp_rock'].values, c='r')
            ax5.plot(temp_time_num, temp_df['temp_reg'].values, c='orange')
            color0 = 'tab:red'
            ax5.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
            ax5.tick_params(axis='y', labelcolor=color0)
            ax5.set_xlim((temp_time_num[0], temp_time_num[-1]))
            ax5b = ax5.twinx()  # instantiate a second axes that shares the same x-axis
            color0b = 'tab:blue'
            ax5b.plot(arrival_times_num_nolm, runsum_nolm)
            ax5b.scatter([burst1_start_num, burst1_end_num],
                         cycle_vals_nolm_nums[0:2], marker='o', c='magenta', edgecolor='k')
            ax5b.scatter([burst2_start_num, burst2_end_num],
                         cycle_vals_nolm_nums[2:], marker='o', c='gray', edgecolor='k')
            ax5b.set_ylabel(f'No LM Cumulative Sum', fontweight='bold', color=color0b)
            ax5.set_xlim((sunrise_val, sunrise_end))
            ax5b.set_xlim((sunrise_val, sunrise_end))
            ax5b.set_ylim((sunrise_min, sunrise_max))
            ax5.set_xticklabels(ax5.get_xticks(), rotation=45)

            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            fig.suptitle(f'Sunrise cyle {sunlight_cycle} for station {sta}: {sunrise_val}-{sunrise_end}',
                         fontweight='bold')
            fig.savefig(f'{output_directory}detailed_analysis/plots/{sta}_cycle{sunlight_cycle}_onlydata1')
            fig.savefig(f'{output_directory}detailed_analysis/plots/EPS_{sta}_cycle{sunlight_cycle}_onlydata.eps')

            print(f'Saved plots for sunlight cycle {sunlight_cycle} for {sta}...')

            sunlight_cycle = sunlight_cycle + 1

        # Combine the vectors into a pandas dataframe and save them
        row_labels = ['Sunrise_1', 'Sunrise_2', 'Sunrise_3', 'Sunrise_4', 'Sunrise_5', 'Sunrise_6',
                      'Sunrise_7', 'Sunrise_8', 'Sunrise_9']

        df_burstdiff = pd.DataFrame(burst_diff_hrs_vector, columns=['burst_diff(hrs)'], index=row_labels)
        df_total = pd.DataFrame(cycle_vals_total_vector, columns=cycle_labels, index=row_labels)
        df_onlylm = pd.DataFrame(cycle_vals_onlylm_vector, columns=cycle_labels, index=row_labels)
        df_nolm = pd.DataFrame(cycle_vals_nolm_vector, columns=cycle_labels, index=row_labels)
        df_burstdiff.to_csv(f'{output_directory}detailed_analysis/data/{sta}_burstdiff.csv')
        df_total.to_csv(f'{output_directory}detailed_analysis/data/{sta}_total_values.csv')
        df_onlylm.to_csv(f'{output_directory}detailed_analysis/data/{sta}_total_onlylm.csv')
        df_nolm.to_csv(f'{output_directory}detailed_analysis/data/{sta}_total_nolm.csv')

    return


main()
