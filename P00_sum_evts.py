"""
Sums the events in the original Civilini 2021 catalog and displays them as a daily bar plot
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
import csv
import numpy as np


def get_temperature(input_file):
    """
    Strip the whitespace from the files

    :param input_file: [str] Path to temperature file
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


def find_total_events(input_df, plt_directory):
    """
    Finds the total number of events and places them in a pie chart

    :param input_df: [pd df] Pandas dataframe of the catalog
    :param plt_directory: [str] Output directory
    :return:
    """

    match4 = len(input_df[input_df.grade == 'A'])/4
    match3 = len(input_df[input_df.grade == 'B'])/3
    match2 = len(input_df[input_df.grade == 'C'])/2
    match1 = len(input_df[input_df.grade == 'D'])

    labels = 'Grade A', 'Grade B', 'Grade C', 'Grade D'
    sizes = [match4, match3, match2, match1]
    colors = ['gold', 'silver', '#CD7F32', '#AF5BBA']

    # Plot
    fig = plt.figure(figsize=(6, 6))
    patches = plt.pie(sizes, labels=labels, colors=colors,
                      autopct='%1.1f%%', startangle=140)
    plt.legend([f'Grade A: {match4}', f'Grade B: {match3}', f'Grade C: {match2}', f'Grade D: {match1}'],
               loc="lower right")
    plt.title(f'Total Moonquakes: {match4 + match3 + match2 + match1} Events', fontweight='bold')

    plt.axis('equal')
    fig.savefig(f'{plt_directory}total_event_distribution.png', dpi=200)
    fig.savefig(f'{plt_directory}total_event_distribution.eps', dpi=200)
    # plt.show()
    plt.close()
    print('Saved total event distribution...')

    return


def calc_daily(input_cat):
    """
    Computes the number of daily events from the input catalog

    :param input_cat: [pd df] Pandas dataframe of input catalog
    :return:
    """
    # Find the upper and lower limits of the date/time vector
    # We are going to use datetime so we can do things properly
    start_string = '1976-08-15'
    start_dt = dt.datetime.strptime(start_string, '%Y-%m-%d')
    end_string = '1977-04-25'
    end_dt = dt.datetime.strptime(end_string, '%Y-%m-%d')
    vector_datetime = []
    for day in np.arange((end_dt - start_dt).days + 1):
        vector_datetime.append(start_dt + dt.timedelta(days=int(day)))

    # Now find the total number of events when we isolate each day
    cumulative_events = []
    for day_datenum in vector_datetime:
        cat_year = day_datenum.year
        cat_month = day_datenum.month
        cat_day = day_datenum.day

        cat_df_subset = input_cat.loc[(input_cat['year'] == cat_year) & (input_cat['month'] == cat_month) &
                                      (input_cat['day'] == cat_day)]

        cumulative_events.append(len(cat_df_subset))

    return vector_datetime, cumulative_events


def sum_daily_evts(input_df, plt_directory, temp_df):
    """
    Plots the sum of daily events

    :param input_df: [pd df] Pandas dataframe of the catalog
    :param plt_directory: [str] Output directory
    :param temp_df: [pd df] Pandas dataframe of the temperature
    :return:
    """

    # Get subsets df for each plot we are interested in making
    df_geo1_match4 = input_df.loc[(input_df['grade'] == 'A') & (input_df['station'] == 'geo1')]
    df_geo2_match4 = input_df.loc[(input_df['grade'] == 'A') & (input_df['station'] == 'geo2')]
    df_geo3_match4 = input_df.loc[(input_df['grade'] == 'A') & (input_df['station'] == 'geo3')]
    df_geo4_match4 = input_df.loc[(input_df['grade'] == 'A') & (input_df['station'] == 'geo4')]

    df_geo1_match3 = input_df.loc[(input_df['grade'] == 'B') & (input_df['station'] == 'geo1')]
    df_geo2_match3 = input_df.loc[(input_df['grade'] == 'B') & (input_df['station'] == 'geo2')]
    df_geo3_match3 = input_df.loc[(input_df['grade'] == 'B') & (input_df['station'] == 'geo3')]
    df_geo4_match3 = input_df.loc[(input_df['grade'] == 'B') & (input_df['station'] == 'geo4')]

    df_geo1_match2 = input_df.loc[(input_df['grade'] == 'C') & (input_df['station'] == 'geo1')]
    df_geo2_match2 = input_df.loc[(input_df['grade'] == 'C') & (input_df['station'] == 'geo2')]
    df_geo3_match2 = input_df.loc[(input_df['grade'] == 'C') & (input_df['station'] == 'geo3')]
    df_geo4_match2 = input_df.loc[(input_df['grade'] == 'C') & (input_df['station'] == 'geo4')]

    df_geo1_match1 = input_df.loc[(input_df['grade'] == 'D') & (input_df['station'] == 'geo1')]
    df_geo2_match1 = input_df.loc[(input_df['grade'] == 'D') & (input_df['station'] == 'geo2')]
    df_geo3_match1 = input_df.loc[(input_df['grade'] == 'D') & (input_df['station'] == 'geo3')]
    df_geo4_match1 = input_df.loc[(input_df['grade'] == 'D') & (input_df['station'] == 'geo4')]

    geo1_m4_days, geo1_m4_numevents = calc_daily(df_geo1_match4)
    geo2_m4_days, geo2_m4_numevents = calc_daily(df_geo2_match4)
    geo3_m4_days, geo3_m4_numevents = calc_daily(df_geo3_match4)
    geo4_m4_days, geo4_m4_numevents = calc_daily(df_geo4_match4)

    geo1_m3_days, geo1_m3_numevents = calc_daily(df_geo1_match3)
    geo2_m3_days, geo2_m3_numevents = calc_daily(df_geo2_match3)
    geo3_m3_days, geo3_m3_numevents = calc_daily(df_geo3_match3)
    geo4_m3_days, geo4_m3_numevents = calc_daily(df_geo4_match3)

    geo1_m2_days, geo1_m2_numevents = calc_daily(df_geo1_match2)
    geo2_m2_days, geo2_m2_numevents = calc_daily(df_geo2_match2)
    geo3_m2_days, geo3_m2_numevents = calc_daily(df_geo3_match2)
    geo4_m2_days, geo4_m2_numevents = calc_daily(df_geo4_match2)

    geo1_m1_days, geo1_m1_numevents = calc_daily(df_geo1_match1)
    geo2_m1_days, geo2_m1_numevents = calc_daily(df_geo2_match1)
    geo3_m1_days, geo3_m1_numevents = calc_daily(df_geo3_match1)
    geo4_m1_days, geo4_m1_numevents = calc_daily(df_geo4_match1)

    # Plot the result
    color_ranking = ['gold', 'silver', '#CD7F32', '#AF5BBA']
    # ---
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    axs[0].bar(geo1_m4_days, geo1_m4_numevents, color=f'{color_ranking[0]}',
               label=f'Grade A ({sum(geo1_m4_numevents)})')
    axs[0].bar(geo1_m3_days, geo1_m3_numevents, bottom=geo1_m4_numevents, color=f'{color_ranking[1]}',
               label=f'Grade B ({sum(geo1_m3_numevents)})')
    axs[0].bar(geo1_m2_days, geo1_m2_numevents, color=f'{color_ranking[2]}',
               bottom=[sum(x) for x in zip(geo1_m4_numevents, geo1_m3_numevents)],
               label=f'Grade C ({sum(geo1_m2_numevents)})')
    axs[0].bar(geo1_m1_days, geo1_m1_numevents, color=f'{color_ranking[3]}',
               bottom=[sum(x) for x in zip(geo1_m4_numevents, geo1_m3_numevents, geo1_m2_numevents)],
               label=f'Grade D ({sum(geo1_m1_numevents)})')

    axs[0].xaxis_date()
    axs[0].set_ylabel('Daily Events', fontweight='bold')
    axs[0].set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    ax0b = axs[0].twinx()
    temp_rock_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax0b.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    ax0b.tick_params(axis='y', labelcolor=color0)
    ax0b.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True, ncol=2)

    axs[1].bar(geo2_m4_days, geo2_m4_numevents, color=f'{color_ranking[0]}',
               label=f'Grade A ({sum(geo2_m4_numevents)})')
    axs[1].bar(geo2_m3_days, geo2_m3_numevents, bottom=geo2_m4_numevents, color=f'{color_ranking[1]}',
               label=f'Grade B ({sum(geo2_m3_numevents)})')
    axs[1].bar(geo2_m2_days, geo2_m2_numevents, color=f'{color_ranking[2]}',
               bottom=[sum(x) for x in zip(geo2_m4_numevents, geo2_m3_numevents)],
               label=f'Grade C ({sum(geo2_m2_numevents)})')
    axs[1].bar(geo2_m1_days, geo2_m1_numevents, color=f'{color_ranking[3]}',
               bottom=[sum(x) for x in zip(geo2_m4_numevents, geo2_m3_numevents, geo2_m2_numevents)],
               label=f'Grade D ({sum(geo2_m1_numevents)})')
    axs[1].xaxis_date()
    axs[1].set_ylabel('Daily Events', fontweight='bold')
    axs[1].set_title(f'Geophone 2', fontweight='bold', y=1.20)
    axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[2].bar(geo3_m4_days, geo3_m4_numevents, color=f'{color_ranking[0]}',
               label=f'Grade A ({sum(geo3_m4_numevents)})')
    axs[2].bar(geo3_m3_days, geo3_m3_numevents, bottom=geo3_m4_numevents, color=f'{color_ranking[1]}',
               label=f'Grade B ({sum(geo3_m3_numevents)})')
    axs[2].bar(geo3_m2_days, geo3_m2_numevents, color=f'{color_ranking[2]}',
               bottom=[sum(x) for x in zip(geo3_m4_numevents, geo3_m3_numevents)],
               label=f'Grade C ({sum(geo3_m2_numevents)})')
    axs[2].bar(geo3_m1_days, geo3_m1_numevents, color=f'{color_ranking[3]}',
               bottom=[sum(x) for x in zip(geo3_m4_numevents, geo3_m3_numevents, geo3_m2_numevents)],
               label=f'Grade D ({sum(geo3_m1_numevents)})')
    axs[2].xaxis_date()
    axs[2].set_ylabel('Daily Events', fontweight='bold')
    axs[2].set_title(f'Geophone 3', fontweight='bold', y=1.20)
    axs[2].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[3].bar(geo4_m4_days, geo4_m4_numevents, color=f'{color_ranking[0]}',
               label=f'Grade A ({sum(geo4_m4_numevents)})')
    axs[3].bar(geo4_m3_days, geo4_m3_numevents, bottom=geo4_m4_numevents, color=f'{color_ranking[1]}',
               label=f'Grade B ({sum(geo4_m3_numevents)})')
    axs[3].bar(geo4_m2_days, geo4_m2_numevents, color=f'{color_ranking[2]}',
               bottom=[sum(x) for x in zip(geo4_m4_numevents, geo4_m3_numevents)],
               label=f'Grade C ({sum(geo4_m2_numevents)})')
    axs[3].bar(geo4_m1_days, geo4_m1_numevents, color=f'{color_ranking[3]}',
               bottom=[sum(x) for x in zip(geo4_m4_numevents, geo4_m3_numevents, geo4_m2_numevents)],
               label=f'Grade D ({sum(geo4_m1_numevents)})')
    axs[3].xaxis_date()
    axs[3].set_ylabel('Daily Events', fontweight='bold')
    axs[3].set_title(f'Geophone 4', fontweight='bold', y=1.20)
    axs[3].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.tight_layout()

    fig.savefig(f'{plt_directory}bar_evt_daily.png', dpi=200)
    fig.savefig(f'{plt_directory}bar_evt_daily.eps', dpi=200)

    plt.close(fig)

    # ---
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.bar(geo1_m4_days, geo1_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
            label=f'Grade A ({sum(geo1_m4_numevents)})')
    axs.bar(geo1_m3_days, geo1_m3_numevents, bottom=geo1_m4_numevents, color=f'{color_ranking[1]}',
            edgecolor="black", linewidth=0.15,
            label=f'Grade B ({sum(geo1_m3_numevents)})')
    axs.bar(geo1_m2_days, geo1_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo1_m4_numevents, geo1_m3_numevents)],
            label=f'Grade C ({sum(geo1_m2_numevents)})')
    axs.bar(geo1_m1_days, geo1_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo1_m4_numevents, geo1_m3_numevents, geo1_m2_numevents)],
            label=f'Grade D ({sum(geo1_m1_numevents)})')

    ax0b = axs.twinx()
    temp_rock_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax0b.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    ax0b.tick_params(axis='y', labelcolor=color0)
    ax0b.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True, ncol=2)

    axs.xaxis_date()
    axs.set_ylabel('Daily Events', fontweight='bold')
    axs.set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    fig.savefig(f'{plt_directory}bar_evt_daily_geo1.png', dpi=300)
    fig.savefig(f'{plt_directory}bar_evt_daily_geo1.eps', dpi=300)

    plt.close(fig)

    # ---
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.bar(geo2_m4_days, geo2_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
            label=f'Grade A ({sum(geo2_m4_numevents)})')
    axs.bar(geo2_m3_days, geo2_m3_numevents, bottom=geo2_m4_numevents, color=f'{color_ranking[1]}',
            edgecolor="black", linewidth=0.15,
            label=f'Grade B ({sum(geo2_m3_numevents)})')
    axs.bar(geo2_m2_days, geo2_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo2_m4_numevents, geo2_m3_numevents)],
            label=f'Grade C ({sum(geo2_m2_numevents)})')
    axs.bar(geo2_m1_days, geo2_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo2_m4_numevents, geo2_m3_numevents, geo2_m2_numevents)],
            label=f'Grade D ({sum(geo2_m1_numevents)})')

    ax0b = axs.twinx()
    temp_rock_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax0b.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    ax0b.tick_params(axis='y', labelcolor=color0)
    ax0b.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True, ncol=2)


    axs.xaxis_date()
    axs.set_ylabel('Daily Events', fontweight='bold')
    axs.set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    fig.savefig(f'{plt_directory}bar_evt_daily_geo2.png', dpi=300)
    fig.savefig(f'{plt_directory}bar_evt_daily_geo2.eps', dpi=300)

    plt.close(fig)

    # ---
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.bar(geo3_m4_days, geo3_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
            label=f'Grade A ({sum(geo3_m4_numevents)})')
    axs.bar(geo3_m3_days, geo3_m3_numevents, bottom=geo3_m4_numevents, color=f'{color_ranking[1]}',
            edgecolor="black", linewidth=0.15,
            label=f'Grade B ({sum(geo3_m3_numevents)})')
    axs.bar(geo3_m2_days, geo3_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo3_m4_numevents, geo3_m3_numevents)],
            label=f'Grade C ({sum(geo3_m2_numevents)})')
    axs.bar(geo3_m1_days, geo3_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo3_m4_numevents, geo3_m3_numevents, geo3_m2_numevents)],
            label=f'Grade D ({sum(geo3_m1_numevents)})')

    ax0b = axs.twinx()
    temp_rock_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax0b.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    ax0b.tick_params(axis='y', labelcolor=color0)
    ax0b.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True, ncol=2)


    axs.xaxis_date()
    axs.set_ylabel('Daily Events', fontweight='bold')
    axs.set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    fig.savefig(f'{plt_directory}bar_evt_daily_geo3.png', dpi=300)
    fig.savefig(f'{plt_directory}bar_evt_daily_geo3.eps', dpi=300)

    plt.close(fig)

    # ---
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.bar(geo4_m4_days, geo4_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
            label=f'Grade A ({sum(geo4_m4_numevents)})')
    axs.bar(geo4_m3_days, geo4_m3_numevents, bottom=geo4_m4_numevents, color=f'{color_ranking[1]}',
            edgecolor="black", linewidth=0.15,
            label=f'Grade B ({sum(geo4_m3_numevents)})')
    axs.bar(geo4_m2_days, geo4_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo4_m4_numevents, geo4_m3_numevents)],
            label=f'Grade C ({sum(geo4_m2_numevents)})')
    axs.bar(geo4_m1_days, geo4_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
            bottom=[sum(x) for x in zip(geo4_m4_numevents, geo4_m3_numevents, geo4_m2_numevents)],
            label=f'Grade D ({sum(geo4_m1_numevents)})')

    ax0b = axs.twinx()
    temp_rock_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_rock'].values, c='r')
    temp_reg_plt = ax0b.plot(temp_df['abs_time'].values, temp_df['temp_reg'].values, c='orange')
    color0 = 'tab:red'
    ax0b.set_ylabel('Temp. (K)', fontweight='bold', color=color0)
    ax0b.tick_params(axis='y', labelcolor=color0)
    ax0b.legend(('Rock Temp.', 'Regolith Temp.'), loc='upper left', fancybox=True, ncol=2)


    axs.xaxis_date()
    axs.set_ylabel('Daily Events', fontweight='bold')
    axs.set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    fig.savefig(f'{plt_directory}bar_evt_daily_geo4.png', dpi=300)
    fig.savefig(f'{plt_directory}bar_evt_daily_geo4.eps', dpi=300)

    plt.close(fig)

    # ----

    fig2, axs = plt.subplots(4, 1, figsize=(8, 12))
    axs[0].bar(geo1_m4_days, geo1_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
               label=f'Grade A ({sum(geo1_m4_numevents)})')

    axs[0].xaxis_date()
    axs[0].set_ylabel('Daily Events', fontweight='bold')
    axs[0].set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[1].bar(geo2_m4_days, geo2_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
               label=f'Grade A ({sum(geo2_m4_numevents)})')

    axs[1].xaxis_date()
    axs[1].set_ylabel('Daily Events', fontweight='bold')
    axs[1].set_title(f'Geophone 2', fontweight='bold', y=1.20)
    axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[2].bar(geo3_m4_days, geo3_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
               label=f'Grade A ({sum(geo3_m4_numevents)})')

    axs[2].xaxis_date()
    axs[2].set_ylabel('Daily Events', fontweight='bold')
    axs[2].set_title(f'Geophone 3', fontweight='bold', y=1.20)
    axs[2].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[3].bar(geo4_m4_days, geo4_m4_numevents, color=f'{color_ranking[0]}', edgecolor="black", linewidth=0.15,
               label=f'Grade A ({sum(geo4_m4_numevents)})')

    axs[3].xaxis_date()
    axs[3].set_ylabel('Daily Events', fontweight='bold')
    axs[3].set_title(f'Geophone 4', fontweight='bold', y=1.20)
    axs[3].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.tight_layout()

    fig2.savefig(f'{plt_directory}bar_evt_daily_match4.png', dpi=200)
    fig2.savefig(f'{plt_directory}bar_evt_daily_match4.eps', dpi=200)

    plt.close(fig2)

    # ----

    fig3, axs = plt.subplots(4, 1, figsize=(8, 12))
    axs[0].bar(geo1_m3_days, geo1_m3_numevents, color=f'{color_ranking[1]}', edgecolor="black", linewidth=0.15,
               label=f'Grade B ({sum(geo1_m3_numevents)})')

    axs[0].xaxis_date()
    axs[0].set_ylabel('Daily Events', fontweight='bold')
    axs[0].set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[1].bar(geo2_m3_days, geo2_m3_numevents, color=f'{color_ranking[1]}', edgecolor="black", linewidth=0.15,
               label=f'Grade B ({sum(geo2_m3_numevents)})')

    axs[1].xaxis_date()
    axs[1].set_ylabel('Daily Events', fontweight='bold')
    axs[1].set_title(f'Geophone 2', fontweight='bold', y=1.20)
    axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[2].bar(geo3_m3_days, geo3_m3_numevents, color=f'{color_ranking[1]}', edgecolor="black", linewidth=0.15,
               label=f'Grade B ({sum(geo3_m3_numevents)})')

    axs[2].xaxis_date()
    axs[2].set_ylabel('Daily Events', fontweight='bold')
    axs[2].set_title(f'Geophone 3', fontweight='bold', y=1.20)
    axs[2].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[3].bar(geo4_m3_days, geo4_m3_numevents, color=f'{color_ranking[1]}', edgecolor="black", linewidth=0.15,
               label=f'Grade B ({sum(geo4_m3_numevents)})')

    axs[3].xaxis_date()
    axs[3].set_ylabel('Daily Events', fontweight='bold')
    axs[3].set_title(f'Geophone 4', fontweight='bold', y=1.20)
    axs[3].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.tight_layout()

    fig3.savefig(f'{plt_directory}bar_evt_daily_match3.png', dpi=200)
    fig3.savefig(f'{plt_directory}bar_evt_daily_match3.eps', dpi=200)

    plt.close(fig3)

    # ----

    fig4, axs = plt.subplots(4, 1, figsize=(8, 12))
    axs[0].bar(geo1_m2_days, geo1_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
               label=f'Grade C ({sum(geo1_m2_numevents)})')

    axs[0].xaxis_date()
    axs[0].set_ylabel('Daily Events', fontweight='bold')
    axs[0].set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[1].bar(geo2_m2_days, geo2_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
               label=f'Grade C ({sum(geo2_m2_numevents)})')

    axs[1].xaxis_date()
    axs[1].set_ylabel('Daily Events', fontweight='bold')
    axs[1].set_title(f'Geophone 2', fontweight='bold', y=1.20)
    axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[2].bar(geo3_m2_days, geo3_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
               label=f'Grade C ({sum(geo3_m2_numevents)})')

    axs[2].xaxis_date()
    axs[2].set_ylabel('Daily Events', fontweight='bold')
    axs[2].set_title(f'Geophone 3', fontweight='bold', y=1.20)
    axs[2].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[3].bar(geo4_m2_days, geo4_m2_numevents, color=f'{color_ranking[2]}', edgecolor="black", linewidth=0.15,
               label=f'Grade C ({sum(geo4_m2_numevents)})')

    axs[3].xaxis_date()
    axs[3].set_ylabel('Daily Events', fontweight='bold')
    axs[3].set_title(f'Geophone 4', fontweight='bold', y=1.20)
    axs[3].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.tight_layout()

    fig4.savefig(f'{plt_directory}bar_evt_daily_match2.png', dpi=200)
    fig4.savefig(f'{plt_directory}bar_evt_daily_match2.eps', dpi=200)

    plt.close(fig4)

    # ----

    fig5, axs = plt.subplots(4, 1, figsize=(8, 12))
    axs[0].bar(geo1_m1_days, geo1_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
               label=f'Grade D ({sum(geo1_m1_numevents)})')

    axs[0].xaxis_date()
    axs[0].set_ylabel('Daily Events', fontweight='bold')
    axs[0].set_title(f'Geophone 1', fontweight='bold', y=1.20)
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[1].bar(geo2_m1_days, geo2_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
               label=f'Grade D ({sum(geo2_m1_numevents)})')

    axs[1].xaxis_date()
    axs[1].set_ylabel('Daily Events', fontweight='bold')
    axs[1].set_title(f'Geophone 2', fontweight='bold', y=1.20)
    axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[2].bar(geo3_m1_days, geo3_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
               label=f'Grade D ({sum(geo3_m1_numevents)})')

    axs[2].xaxis_date()
    axs[2].set_ylabel('Daily Events', fontweight='bold')
    axs[2].set_title(f'Geophone 3', fontweight='bold', y=1.20)
    axs[2].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    axs[3].bar(geo4_m1_days, geo4_m1_numevents, color=f'{color_ranking[3]}', edgecolor="black", linewidth=0.15,
               label=f'Grade D ({sum(geo4_m1_numevents)})')

    axs[3].xaxis_date()
    axs[3].set_ylabel('Daily Events', fontweight='bold')
    axs[3].set_title(f'Geophone 4', fontweight='bold', y=1.20)
    axs[3].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.tight_layout()

    fig5.savefig(f'{plt_directory}bar_evt_daily_match1.png', dpi=200)
    fig5.savefig(f'{plt_directory}bar_evt_daily_match1.eps', dpi=200)

    print('Saved daily event distribution...')
    plt.close(fig5)

    return


def add_time_data(input_df):
    """
    Add time data to the input dataframe. Part of legacy code.

    :param input_df: [pd df] Pandas dataframe of input catalog
    :return:
    """

    datetime_values = input_df['abs_time'].values
    year, month, day = [], [], []
    hour, minute, sec = [], [], []

    for datetime_val in datetime_values:
        datetime_obj = dt.datetime.strptime(datetime_val, '%Y-%m-%dT%H:%M:%S')
        year.append(datetime_obj.year)
        month.append(datetime_obj.month)
        day.append(datetime_obj.day)
        hour.append(datetime_obj.hour)
        minute.append(datetime_obj.minute)
        sec.append(datetime_obj.second)

    input_df['year'] = year
    input_df['month'] = month
    input_df['day'] = day
    input_df['hour'] = hour
    input_df['minute'] = minute
    input_df['second'] = sec

    return input_df


def main():
    """
    Main wrapper function
    :return:
    """
    rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
    outdir = 'C:/data/lunar_output/'
    plt_directory = f'{outdir}sum_events/'
    input_catalog = f'{rundir}fc_lspe_dl_catalog.csv'
    temperature_file = f'{rundir}longterm_thermal_data.txt'

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(plt_directory):
        os.mkdir(plt_directory)

    # Read the catalog
    df_cat = pd.read_csv(input_catalog)

    # Add time data to the catalog (part of legacy code, ignored in other processing steps)
    df_cat = add_time_data(df_cat)

    # Do a pie chart of the events
    find_total_events(df_cat, plt_directory)

    # Get the values of temperature
    temp_df = get_temperature(temperature_file)

    # Plot the daily distribution with temperature
    sum_daily_evts(df_cat, plt_directory, temp_df)

    return


main()
