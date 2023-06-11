"""
Locates thermal moonquakes using stochastic gradient descent for a real event using the parametrization determined in
P10.

"""

# Import packages
import pickle
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import math
import datetime
from joblib import Parallel, delayed


def tt_misfit(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the total travel time misfit across all geophone stations

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """
    # Initialize the vector
    sta_misfit = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):

        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_misfit[sta_ind] = (ts + (np.sqrt(((xs-xi)**2)+((ys-yi)**2))/v) - ti)**2
    misfit = np.sum(sta_misfit)

    return misfit


def tt_misfit_geo3(ts, xs, ys, avg_velocity, geo_locs, t3):
    """
    Computes the total travel time misfit across only geophone 3

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param t3: [float] Relative arrival time for geophone 3
    :return:
    """
    # Compute the misfit
    xi = geo_locs[2][0]
    yi = geo_locs[2][1]
    ti = t3
    misfit = (ts + (np.sqrt(((xs-xi)**2)+((ys-yi)**2))/avg_velocity) - ti)**2

    return misfit


def tt_misfit_polar(ts, rs, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the total travel time misfit relative to station Geophone 3 using polar coordinates

    :param ts: [float] Source time (param)
    :param rs: [float] Source distance location (in meters)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Compute the misfit
    misfit = (ts + (rs/avg_velocity)-t3)**2

    return misfit


def comp_dt_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dt

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dt_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit'
        sta_dt_grad[sta_ind] = (2*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))

    dt_grad = np.sum(sta_dt_grad)

    return dt_grad


def comp_dx_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dx

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dx_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dx_grad[sta_ind] = (2*(xs-xi)*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))/(v*(np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dx_grad = np.sum(sta_dx_grad)

    return dx_grad


def comp_dy_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dy

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dy_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dy_grad[sta_ind] = (2 * (ys - yi) * (ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti)) / (
                    v * (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dy_grad = np.sum(sta_dy_grad)

    return dy_grad


def compute_dt2_grad_polar(ts, rs, avg_velocity, new_rel_time):

    """
    Computes the gradient of ts using polar coordinates.
    Note: we are only caring about geophone 3 in this instance.

    :param ts: [float] Source time
    :param rs: [float] Distance to source
    :param avg_velocity: [float] Average the velocity
    :param new_rel_time: [vector] Arrival times at all geophones
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Solve for the gradient according to ts from the analytical expression
    dt2_grad = 2*(ts + (rs/avg_velocity) - t3)

    return dt2_grad


def compute_dr_grad_polar(ts, rs, avg_velocity, new_rel_time):

    """
    Computes the gradient of rs using polar coordinates.
    Note: we are only caring about geophone 3 in this instance.

    :param ts: [float] Source time
    :param rs: [float] Distance to source
    :param avg_velocity: [float] Average the velocity
    :param new_rel_time: [vector] Arrival times at all geophones
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Solve for the gradient according to ts from the analytical expression
    dr_grad = (2*(avg_velocity*(ts - t3) + rs))/(avg_velocity**2)

    return dr_grad


def model(x_vector, y_vector, new_rel_time, avg_velocity, geo_locs, evid, absolute_arrival_times, xs, ys):
    """
    Stochastic gradient descent
    :param x_vector: [vector] Distance x (from Geo3) parameter space in meters
    :param y_vector: [vector] Distance y (from Geo3) parameter space in meters
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :param avg_velocity: [float] Average expected wave velocity
    :param old_rel_time: [vector] Vector of fine-tuned arrivals non-relative (for plotting purposes only)
    :return:
    """
    # Set the number of iterations and the learning rate
    # True number of iterations is one less than displayed
    num_iterations = 500001
    lr_t = 0.05
    lr_x = 200.0
    lr_y = lr_x

    # Set a misfit improvement cutoff value. If the mean improvement of the past number of iterations is below this,
    # we are probably ok with stopping.
    iteration_num_cutoff = 10000
    iteration_value_cutoff = 0.1

    # We will want ot save the misfit, but the plot step of 100 is too large. 1000 is fine.
    # misfit_save_bounds = np.arange(0, num_iterations, 1000)

    # Initialize the parameters randomly
    # ts = np.random.choice(t_vector)
    ts = 0
    # xs = np.random.choice(x_vector)
    # ys = np.random.choice(y_vector)

    misfit_vector = []
    iteration_vector = []
    ts_vector = []
    xs_vector = []
    ys_vector = []
    theta_vector = []

    # For easier plotting, pull out the x and y values of the geophone locations
    geo_loc_x = []
    geo_loc_y = []
    for geo_loc in geo_locs:
        geo_loc_x.append(geo_loc[0])
        geo_loc_y.append(geo_loc[1])

    # Take the final x and y location from the first step and calculate a theta between that and geo3
    x3 = geo_loc_x[2]
    y3 = geo_loc_y[2]

    # f = open(f"{output_directory}{evid}_{np.round(ts)}_{np.round(xs)}_{np.round(ys)}.txt", "a")
    for iteration in np.arange(num_iterations):

        # Do a forward propagation of the traces
        misfit = tt_misfit(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)

        # Compute theta
        atan_val = math.atan((xs - x3) / (ys - y3))
        atan_val_deg = math.degrees(atan_val)
        if xs > 0 and ys > 0:
            theta_deg = atan_val_deg
        if xs > 0 and ys < 0:
            theta_deg = atan_val_deg + 180
        if xs < 0 and ys < 0:
            theta_deg = atan_val_deg + 180
        if xs < 0 and ys > 0:
            theta_deg = atan_val_deg + 360
        theta_vector.append(theta_deg)

        if iteration > iteration_num_cutoff + 1:
            if abs(theta_vector[iteration - iteration_num_cutoff] - theta_deg) < iteration_value_cutoff:
                break

        iteration_vector.append(iteration)
        misfit_vector.append(misfit)
        ts_vector.append(ts)
        xs_vector.append(xs)
        ys_vector.append(ys)

        # Compute the gradient using the analytical derivative
        # To make things clear, we will create a new function for each parameter
        dt = comp_dt_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)
        dx = comp_dx_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)
        dy = comp_dy_grad(ts, xs, ys, avg_velocity, geo_locs, new_rel_time)

        # Update parameters using the learning rate
        ts = ts - (lr_t * dt)
        xs = xs - (lr_x * dx)
        ys = ys - (lr_y * dy)


        # if iteration % 10000 == 0:
        #     print(f'Misfit1 for iteration {iteration}: {misfit}')
            # f.write(f'{iteration} {misfit}\n')

    # f.close()

    # Some of the misfit vectors are too large to plot all the results. Instead, we will set up plot bounds.
    # We shall plot 1000 points of the vector
    plot_bounds = np.arange(0, len(iteration_vector), int(len(iteration_vector)/1000))
    plot_bounds = np.append(plot_bounds, np.arange(500))
    plot_bounds = sorted(np.unique(plot_bounds))

    # Create variables for the first source values
    xs_start = xs_vector[0]
    ys_start = ys_vector[0]
    ts_start = ts_vector[0]

    xs1 = xs_vector[-1]
    ys1 = ys_vector[-1]

    atan_val = math.atan((xs1 - x3) / (ys1 - y3))
    atan_val_deg = math.degrees(atan_val)
    if xs1 > 0 and ys1 > 0:
        theta_deg = atan_val_deg
    if xs1 > 0 and ys1 < 0:
        theta_deg = atan_val_deg + 180
    if xs1 < 0 and ys1 < 0:
        theta_deg = atan_val_deg + 180
    if xs1 < 0 and ys1 > 0:
        theta_deg = atan_val_deg + 360

    # Get the time series traces for plotting
    trace_file = glob.glob(f'{trace_folder}evid_{evid}.pkl')[0]
    with open(trace_file, 'rb') as f:
        time_array_cut, data_array_cut, abs_trace_start, abs_trace_end, \
        rel_det_vector, input_info, data_geophone_list = pickle.load(f)

    # Get the updated ft spectrograms for plotting (from P1)
    spectr = glob.glob(f'{spectr_folder}evid_spectr_{evid}.pkl')[0]
    with open(spectr, 'rb') as f:
        spec_t_array, spec_f_array, Sxx_array = pickle.load(f)

    # For plotting purposes, we need to get the fine-tuned absolute arrival times in relative format
    relative_arrival_time_plot = []
    for abs_arrival_time in absolute_arrival_times:
        time_diff = abs_arrival_time - abs_trace_start
        relative_arrival_time_plot.append(time_array_cut[0, 0] + time_diff.seconds + time_diff.microseconds/(10 ** 6))

    # Do a plot of the misfit after our analysis
    fig1 = plt.figure(figsize=(15, 8), num=2, clear=True)

    # Plot the varying location in space of the location
    ax0 = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=3)
    ax0.scatter(geo_loc_x, geo_loc_y, marker='^')
    ax0.set_xlim((np.min(x_vector), np.max(x_vector)))
    ax0.set_ylim((np.min(y_vector), np.max(y_vector)))
    ax0.scatter(np.array(xs_vector)[plot_bounds], np.array(ys_vector)[plot_bounds],
                c=np.array(iteration_vector)[plot_bounds], marker='o', s=35, cmap=cm.coolwarm)
    ax0.scatter(xs_start, ys_start, marker='o', s=35, facecolors=None, edgecolors='black')
    ax0.plot([x3, xs_vector[-1]], [y3, ys_vector[-1]], c='k')
    ax0.set_xlabel('X Distance', fontweight='bold')
    ax0.set_ylabel('Y Distance', fontweight='bold')

    # Plot the misfit
    ax1 = plt.subplot2grid((4, 6), (3, 0), colspan=2, rowspan=1)
    ax1.scatter(np.array(iteration_vector)[plot_bounds], np.array(theta_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Theta', fontweight='bold')
    # ax1.set_ylim((np.min(misfit_vector[plot_bounds]), np.max(misfit_vector[plot_bounds])))

    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
    ax2.plot(time_array_cut[:, 0], data_array_cut[:, 0])
    ax2.set_xlim((np.min(time_array_cut[:, 0]), np.max(time_array_cut[:, 0])))
    ax2.axvline(x=relative_arrival_time_plot[0], c='r')
    ax2.set_ylabel('Geo 1')

    ax3 = plt.subplot2grid((4, 6), (1, 2), colspan=2, rowspan=1)
    ax3.plot(time_array_cut[:, 1], data_array_cut[:, 1])
    ax3.set_xlim((np.min(time_array_cut[:, 0]), np.max(time_array_cut[:, 0])))
    ax3.axvline(x=relative_arrival_time_plot[1], c='r')
    ax3.set_ylabel('Geo 2')

    ax4 = plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=1)
    ax4.plot(time_array_cut[:, 2], data_array_cut[:, 2])
    ax4.set_xlim((np.min(time_array_cut[:, 0]), np.max(time_array_cut[:, 0])))
    ax4.axvline(x=relative_arrival_time_plot[2], c='r')
    ax4.set_ylabel('Geo 3')

    ax5 = plt.subplot2grid((4, 6), (3, 2), colspan=2, rowspan=1)
    ax5.plot(time_array_cut[:, 3], data_array_cut[:, 3])
    ax5.set_xlim((np.min(time_array_cut[:, 0]), np.max(time_array_cut[:, 0])))
    ax5.axvline(x=relative_arrival_time_plot[3], c='r')
    ax5.set_ylabel('Geo 4')

    specmax = 1e-6
    ax6 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=1)
    ax6.pcolormesh(spec_t_array[:, 0], spec_f_array[:, 0], Sxx_array[:, :, 0], cmap=cm.jet, shading='auto')
    # ax6.pcolormesh(spec_t_array[:, 0], spec_f_array[:, 0], Sxx_array[:, :, 0], cmap=cm.jet, vmax=specmax,
    #                shading='auto')
    ax6.axvline(x=relative_arrival_time_plot[0], c='r')

    ax7 = plt.subplot2grid((4, 6), (1, 4), colspan=2, rowspan=1)
    ax7.pcolormesh(spec_t_array[:, 1], spec_f_array[:, 1], Sxx_array[:, :, 1], cmap=cm.jet, shading='auto')
    # ax7.pcolormesh(spec_t_array[:, 1], spec_f_array[:, 1], Sxx_array[:, :, 1], cmap=cm.jet, vmax=specmax,
    #                shading='auto')
    ax7.axvline(x=relative_arrival_time_plot[1], c='r')

    ax8 = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=1)
    ax8.pcolormesh(spec_t_array[:, 2], spec_f_array[:, 2], Sxx_array[:, :, 2], cmap=cm.jet, shading='auto')
    # ax8.pcolormesh(spec_t_array[:, 2], spec_f_array[:, 2], Sxx_array[:, :, 2], cmap=cm.jet, vmax=specmax,
    #                shading='auto')
    ax8.axvline(x=relative_arrival_time_plot[2], c='r')

    ax9 = plt.subplot2grid((4, 6), (3, 4), colspan=2, rowspan=1)
    ax9.pcolormesh(spec_t_array[:, 3], spec_f_array[:, 3], Sxx_array[:, :, 3], cmap=cm.jet, shading='auto')
    # ax9.pcolormesh(spec_t_array[:, 3], spec_f_array[:, 3], Sxx_array[:, :, 3], cmap=cm.jet, vmax=specmax,
    #                shading='auto')
    ax9.axvline(x=relative_arrival_time_plot[3], c='r')

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.9)
    fig1.suptitle(f'Azimuth for {evid}, start = ({xs_start}, {ys_start}, {ts_start}) '
                  f': Theta = {np.round(theta_deg, 1)} degrees', fontweight='bold')
    fig1.savefig(f'{output_directory}{evid}/image_{evid}_{np.round(ts_start)}_{np.round(xs_start)}_'
                 f'{np.round(ys_start)}.png')
    # fig1.savefig(f'{output_directory}{evid}/image_{evid}_{np.round(ts_start)}_{np.round(xs_start)}_'
    #              f'{np.round(ys_start)}.eps')

    # plt.show()
    # plt.close()

    # Save the values of misfit
    # combined_data = list(zip(np.arange(num_iterations)[misfit_save_bounds], misfit_vector[misfit_save_bounds]))
    # df = pd.DataFrame(combined_data, columns=['iteration_number', 'misfit'])
    # df.to_csv(f'{output_directory}{evid}/data_{evid}_{np.round(ts)}_{np.round(xs)}_{np.round(ys)}_misfit.csv',
    #           index=False)
    final_misfit = misfit_vector[-1]
    final_iteration = iteration_vector[-1]

    return theta_deg, xs_start, ys_start, ts_start, xs_vector[-1], ys_vector[-1], ts_vector[-1], final_misfit, final_iteration


def preprocess(input_file):
    """
    Obtains the relative velocities

    :param input_file:
    :return:
    """
    # Read in the relative values
    evid_arrival_df = pd.read_csv(input_file)

    absolute_arrival_times_str = evid_arrival_df['ft_arrival_time'].values
    absolute_arrival_times = []
    for abs_time in absolute_arrival_times_str:
        absolute_arrival_times.append(datetime.datetime.strptime(abs_time, "%Y-%m-%dT%H:%M:%S.%f"))
    earliest_arrival = np.min(absolute_arrival_times)

    # Get arrival values relative to the minimum arrival
    rel_values = []
    for abs_arrival_time in absolute_arrival_times:
        time_diff = abs_arrival_time-earliest_arrival
        rel_values.append(time_diff.seconds + time_diff.microseconds/(10 ** 6))

    return rel_values, absolute_arrival_times


def location_wrapper(input_file):
    """
    Wrapper code for finding the location of the input file

    :param input_file: [str] The filepath of the input file
    :return:
    """

    # Pull out the evid from the filename
    file_bn = os.path.basename(input_file)
    evid = file_bn.split('_')[2].split('.')[0]

    # If evids in the bad evids, don't process them
    # evids_to_exclude = ['770425-00-M1', '761111-21-M2', '770416-10-M1', '770325-13-M6', '770114-15-M1',
    #                     '761105-08-M1', '770314-22-M1', '761021-07-M1', '760901-14-M2']
    # if len(np.where(np.array(evids_to_exclude) == evid)[0]) > 0:
    #     print(f'Event {evid} should not be processed!')
    #     return

    # Check if the file has already been processed. If so, skip it
    if os.path.exists(f'{output_directory}{evid}/azimuth_results_{evid}.csv'):

        # Check that a total of 10 runs were done for this
        azi_testdf = pd.read_csv(f'{output_directory}{evid}/azimuth_results_{evid}.csv')
        if len(azi_testdf) < 10:
            os.remove(f'{output_directory}{evid}/azimuth_results_{evid}.csv')
            print(f'Reprocessing evid {evid} due to insufficient events!')
        else:
            print(f'Event {evid} has already been processed...')
            return

    # Set average velocity for this area
    avg_velocity = 34.0

    # Set locations of each geophone relative to geophone 3, the center of the array (in meters)
    geo1 = np.multiply((0.0455, 0.0341), 1000)
    geo2 = np.multiply((-0.0534, 0.0192), 1000)
    geo3 = (0, 0)
    geo4 = np.multiply((0.0119, -0.0557), 1000)
    geo_locs = [geo1, geo2, geo3, geo4]

    # Obtain the relative time differences between each sensor
    new_rel_time, absolute_arrival_times = preprocess(input_file)

    # Setup the starting parameter space
    x_vector = np.arange(-2000, 2001)
    y_vector = np.arange(-2000, 2001)

    theta_vector = []
    xs_start_vector, ys_start_vector, ts_start_vector = [], [], []
    xs_fin_vector, ys_fin_vector, ts_fin_vector = [], [], []
    misfit_fin_vector = []
    final_iteration_vector = []

    np.random.seed(seed=1)
    xs_vector = np.random.choice(x_vector, num_start_locations)
    ys_vector = np.random.choice(y_vector, num_start_locations)

    for start_iter in np.arange(num_start_locations):
        theta, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin, misfit_fin, final_iteration = \
            model(x_vector, y_vector, new_rel_time, avg_velocity, geo_locs, evid, absolute_arrival_times,
                  xs_vector[start_iter], ys_vector[start_iter])
        theta_vector.append(theta)
        xs_start_vector.append(xs_start)
        ys_start_vector.append(ys_start)
        ts_start_vector.append(ts_start)
        xs_fin_vector.append(xs_fin)
        ys_fin_vector.append(ys_fin)
        ts_fin_vector.append(ts_fin)
        misfit_fin_vector.append(misfit_fin)
        final_iteration_vector.append(final_iteration)

    # Combine into a pandas dataframe and save
    combined_data = list(zip(theta_vector, xs_start_vector, ys_start_vector, ts_start_vector,
                             xs_fin_vector, ys_fin_vector, ts_fin_vector, misfit_fin_vector, final_iteration_vector))
    df = pd.DataFrame(combined_data, columns=['theta', 'xs_start', 'ys_start', 'ts_start',
                                              'xs_fin', 'ys_fin', 'ts_fin', 'misfit_fin', 'final_iteration'])
    df.to_csv(f'{output_directory}{evid}/azimuth_results_{evid}.csv', index=False)
    print(f'Finished processing event {evid}...')

    return


def create_file_structure(input_filelist):
    """
    Creates the file structure for the output

    :param input_filelist:
    :return:
    """
    for filepath in filelist:
        # Pull out the evid from the filename
        file_bn = os.path.basename(filepath)
        evid = file_bn.split('_')[2].split('.')[0]

        if not os.path.exists(f'{output_directory}{evid}'):
            os.mkdir(f'{output_directory}{evid}')

    return


# Main
indir = 'C:/data/lunar_output/'
resultsdir = f'{indir}results/'
trace_folder = f'{indir}original_images/combined_files/'
spectr_folder = f'{indir}fine-tuned/ft_spectrograms/'
file_pkl_directory = f'{indir}original_images/combined_files/'
input_directory = f'{indir}fine-tuned/cat_stats/'

# Create output locations
if not os.path.exists(f'{resultsdir}locations'):
    os.mkdir(f'{resultsdir}locations')

output_directory = f'{resultsdir}locations/azimuth/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Select the number of cores to run this data on
num_cores = 10

# Get the number of random start locations
num_start_locations = 10

filelist = glob.glob(f'{input_directory}*_stats.csv')

# Setup a number of randomized starting locations (i.e. number of iterations)
# Create folders for each event to not trip up the parallelization
create_file_structure(filelist)

if num_cores == 1:
    for infile in filelist:
        location_wrapper(infile)
else:
    Parallel(n_jobs=num_cores)(delayed(location_wrapper)(infile)
                               for infile in filelist)
