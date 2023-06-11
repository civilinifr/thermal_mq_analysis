"""
Determine the wavespeed between each station for each event based on traveltime obtained from the finetuned arrival
times
"""
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import os


def get_wavespeed(df, evid):
    """
    Function to get the wavespeed

    :param df: [pd df] Pandas dataframe of the catalog file
    :param evid: [str] Event ID
    :return:
    """

    df_sub = df[df['evid'] == evid]

    geo1 = np.multiply((0.0455, 0.0341), 1000)
    geo2 = np.multiply((-0.0534, 0.0192), 1000)
    geo3 = (0, 0)
    geo4 = np.multiply((0.0119, -0.0557), 1000)
    geo_locs = [geo1, geo2, geo3, geo4]

    # Find the earliest arrival time and the latest arrival time, then get the wavespeed between those two
    # Find the max and min arrival times
    arrival_times = []
    arrival_times_str = df_sub['ft_arrival_time'].values
    for arrival_time_str in arrival_times_str:
        arrival_times.append(dt.datetime.strptime(arrival_time_str, "%Y-%m-%dT%H:%M:%S.%f"))

    min_ind = np.where(np.array(arrival_times) == np.min(arrival_times))[0][0]
    max_ind = np.where(np.array(arrival_times) == np.max(arrival_times))[0][0]

    # Find the distance between the two geophone corresponding to the min and max values
    geo_min = geo_locs[min_ind]
    geo_max = geo_locs[max_ind]

    geo_min_x = geo_min[0]
    geo_min_y = geo_min[1]
    geo_max_x = geo_max[0]
    geo_max_y = geo_max[1]

    # Get the distance between the two of them and the time to find the wavespeed
    dist = np.sqrt((geo_max_x - geo_min_x) ** 2 + (geo_max_y - geo_min_y) ** 2)
    diff = np.max(arrival_times) - np.min(arrival_times)
    time = diff.seconds + diff.microseconds/(10**6)
    vel = dist/time

    return vel

def main():
    """
    Main wrapper function

    :return:
    """
    out_dir = 'C:/data/lunar_output/'
    if not os.path.exists(f'{out_dir}results/'):
        os.mkdir(f'{out_dir}results/')
    output_folder = f'{out_dir}results/wavespeed/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load and read the catalog file
    catfile = f'{out_dir}catalogs/GradeA_thermal_mq_catalog.csv'
    df = pd.read_csv(catfile)

    # Get the total number of evids
    vel_values = []
    total_evids = np.unique(df['evid'].values)
    for evid in total_evids:
        evid_ind = np.where(df['evid'].values == evid)[0][0]
        vel = get_wavespeed(df, evid)
        vel_values.append(vel)
        progress = ((evid_ind/4) / (len(df['evid'].values)/4))*100
        print(f'{str(np.round(progress, decimals=1))}%...')
    total_evids_num = np.arange(len(total_evids))

    # # Remove all values greater than 200 m/s, nans, and in our previous list
    bad_vel_ind = []
    evids_to_exclude = ['770425-00-M1', '761111-21-M2', '770416-10-M1', '770325-13-M6', '770114-15-M1',
                        '761105-08-M1', '770314-22-M1', '761021-07-M1', '760901-14-M2']
    for exclude_evid in evids_to_exclude:
        bad_vel_ind.append(np.where(total_evids == exclude_evid)[0][0])

    nan_value = np.argwhere(np.isnan(vel_values))[0][0]
    bad_vel_ind.append(nan_value)

    exceeding_values = np.where(np.array(vel_values) > 200.0)[0]
    for exceeding_value in exceeding_values:
        bad_vel_ind.append(exceeding_value)
    bad_vel_ind = sorted(np.unique(bad_vel_ind))

    good_values = np.delete(vel_values, bad_vel_ind)
    good_evids = np.delete(total_evids, bad_vel_ind)
    good_evids_num = np.delete(total_evids_num, bad_vel_ind)
    bad_values = np.array(vel_values)[bad_vel_ind]
    bad_evids = np.array(total_evids)[bad_vel_ind]
    bad_evids_num = np.array(total_evids_num)[bad_vel_ind]

    meanval = np.mean(good_values)
    stdev = np.std(good_values)

    fig = plt.figure(figsize=(10, 8))
    ax0 = plt.subplot(2, 1, 1)
    ax0.scatter(good_evids_num, good_values, edgecolor='black')
    ax0.scatter(bad_evids_num, bad_values, edgecolor='black', c='red')
    ax0.set_xlim((0, len(vel_values)))
    ax0.axhline(200, linestyle='dashed', c='black')
    ax0.set_yscale('log')
    ax0.set_xlabel('Evid', fontweight='bold')
    ax0.set_ylabel('Velocity (m/s)', fontweight='bold')
    ax1 = plt.subplot(2, 1, 2)
    ax1.hist(good_values, bins=10, edgecolor='black')
    ax1.set_xlabel('Velocity (m/s)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax0.set_title(f'Avg: {str(np.round(meanval, decimals=2))}m/s   '
                  f'stdev: {str(np.round(stdev, decimals=2))}')
    fig.tight_layout()

    fig.savefig(f'{output_folder}thermal_mq_wavespeed.png')
    fig.savefig(f'{output_folder}thermal_mq_wavespeed.eps')

    return


main()
