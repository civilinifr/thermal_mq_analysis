"""
Visual way to pick the start and end time of the two bursts occurring at sunrise for each cycle
"""

import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import csv
import os


# Functions
class Cursor:
    """
    A cross hair cursor for the plot.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


def onclick(event):
    """
    Function to pick numbers when mouse is clicked

    :param event: Mouse click
    :return:
    """
    global ix, iy, evid
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy})')

    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))

    for coords_num in np.arange(len(coords)):
        ax3b.axvline(x=coords[coords_num][0], c='black', linestyle='dashed')

    if len(coords) == 4:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
    return


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


# Main
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
outdir = 'C:/data/lunar_output/'
catalog_file = f'{outdir}catalogs/GradeA_thermal_mq_catalog_final.csv'
temperature_file = f'{rundir}longterm_thermal_data.txt'
output_directory = f'{outdir}results/sunrise_burst_analysis/'
nightfall_file = f'{rundir}nightfall_burst.txt'
sunrise_file = f'{rundir}sunrise_burst.txt'

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
if not os.path.exists(f'{output_directory}detailed_analysis/'):
    os.mkdir(f'{output_directory}detailed_analysis/')

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

# stations = ['geo1', 'geo2', 'geo3', 'geo4']
stations = ['geo1']
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
    sunrise_end_time = int(sunrise_end_hours*60*60)

    # Convert the temperature time to datetime
    temp_time_str = temp_df['abs_time'].values.astype('str')
    temp_time_num = []
    for time_str in temp_time_str:
        temp_time_num.append(dt.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f000"))

    # Plot all the sunrise values
    sunlight_cycle = 1
    for sunrise_val in sunrise_num:
        coords = []

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
        ax5b.set_ylabel(f'No LM Cumulative Sum', fontweight='bold', color=color0b)
        ax5.set_xlim((sunrise_val, sunrise_end))
        ax5b.set_xlim((sunrise_val, sunrise_end))
        ax5b.set_ylim((sunrise_min, sunrise_max))
        ax5.set_xticklabels(ax5.get_xticks(), rotation=45)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.suptitle(f'Sunrise cyle {sunlight_cycle} for station {sta}: {sunrise_val}-{sunrise_end}', fontweight='bold')
        fig.savefig(f'{output_directory}detailed_analysis/{sta}_cycle{sunlight_cycle}_onlydata.png')
        fig.savefig(f'{output_directory}detailed_analysis/EPS_{sta}_cycle{sunlight_cycle}_onlydata.eps')

        # If the times already exist, skip them
        if os.path.exists(f'{output_directory}cycle{sunlight_cycle}_burst_times.txt'):
            print(f'cycle{sunlight_cycle} burst_times already picked! Skipping...')
            sunlight_cycle = sunlight_cycle + 1
            continue

        if sta == 'geo1':
            # Call click func
            cid = fig.canvas.mpl_connect('button_press_event', onclick)

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            cursor = Cursor(ax3b)
            fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

            plt.show()

            # Get the time coordinates for each of the two sunrise bursts
            burst1_start_click = coords[0][0]
            burst1_end_click = coords[1][0]
            burst2_start_click = coords[2][0]
            burst2_end_click = coords[3][0]

            # Save the values
            f = open(f'{output_directory}cycle{sunlight_cycle}_burst_times.txt', "w")
            f.write(f'{burst1_start_click}\n')
            f.write(f'{burst1_end_click}\n')
            f.write(f'{burst2_start_click}\n')
            f.write(f'{burst2_end_click}\n')
            f.close()

        # Advance through the sunlight cycle
        sunlight_cycle = sunlight_cycle + 1


