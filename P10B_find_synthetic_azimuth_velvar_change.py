"""
After obtaining a preferred set of parameters in P10, this code tests the accuracy of those parameters in the case that
the real wavespeed is different from the hypothesized wavespeed.

In other words, it answers the question of how sensitive the results are to the presumed wavespeed.

"""

# Import packages
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import math
from joblib import Parallel, delayed


def preprocess(data, geo_locs, avg_velocity=45):
    """
    Preprocesses the input panda array of data. Produces:
    - Distances between each geophone
    - Computes measure of average velocity to use for later processing
    - Creates a new dataframe using relative times for each station
    :param avg_velocity:
    :param data: [pd df] Pandas dataframe of the arrival times for each geophone
    :param geo_locs: [list] Geophone relative locations in m
    :return:
    """
    # For right now we are not going to compute a velocity
    # We use the group velocity measurement from Larose et al. [2005] (45 m/s)
    rel_time = data['new_relative_time'].values
    new_rel_time = data['new_relative_time'].values - np.min(rel_time)

    # We need to fix the algorithm to be more accurate, as it's really just a matter of seconds.
    # For the time being, we will use a relative time used in Nick Schmerr's code
    new_rel_time = [0.1, 1.9, 0.8, 1.1]

    return new_rel_time, avg_velocity


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


def model(x_vector, y_vector, new_rel_time, avg_velocity, geo_locs, ts_syn, xs_syn, ys_syn,
          input_parameters, xs, ys):
    """
    Runs stochastic gradient descent

    :param x_vector: [vector] Distance x (from Geo3) parameter space in meters
    :param y_vector: [vector] Distance y (from Geo3) parameter space in meters
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :param avg_velocity: [float] Moonquake velocity
    :param geo_locs: [array] XY locations of the geophones
    :param ts_syn: [float] Start time of the synthetic source
    :param xs_syn: [float] Horizontal location (x) of the synthetic source
    :param ys_syn: [float] Vertical location (y) of the synthetic source
    :param input_parameters:
    :param xs: [float] Random horizontal (x) start location of Stochastic Gradient Descent
    :param ys: [float] Random vertical (y) start location of Stochastic Gradient Descent
    :return:
    """
    # Set the number of iterations and the learning rate
    # True number of iterations is one less than displayed
    num_iterations = 500001
    time_step = float(input_parameters.split('_')[1])
    space_step = float(input_parameters.split('_')[0])
    lr_t = time_step
    lr_x = space_step
    lr_y = lr_x

    # Set a misfit improvement cutoff value. If the mean improvement of the past number of iterations is below this,
    # we are probably ok with stopping.
    iteration_num_cutoff = 10000
    iteration_value_cutoff = 0.1

    # Initialize the parameters randomly
    # Source time starts from zero.
    ts = 0

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

    # Some of the misfit vectors are too large to plot all the results. Instead, we will just plot 100 points.
    plot_bounds = np.arange(0, len(iteration_vector), int(len(iteration_vector) / 1000))
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

    # Find the theta of the source
    atan_val_syn = math.atan((xs_syn - x3) / (ys_syn - y3))
    atan_val_deg_syn = math.degrees(atan_val_syn)
    if xs_syn > 0 and ys_syn > 0:
        theta_deg_syn = atan_val_deg_syn
    if xs_syn > 0 and ys_syn < 0:
        theta_deg_syn = atan_val_deg_syn + 180
    if xs_syn < 0 and ys_syn < 0:
        theta_deg_syn = atan_val_deg_syn + 180
    if xs_syn < 0 and ys_syn > 0:
        theta_deg_syn = atan_val_deg_syn + 360

    # Calculate the degrees difference between the actual and calculated difference
    theta_diff = np.abs(theta_deg - theta_deg_syn)

    # Do a plot of the misfit after our analysis
    fig1 = plt.figure(figsize=(12, 8), num=2, clear=True)

    # Plot the varying location in space of the location
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    ax0.scatter(geo_loc_x, geo_loc_y, marker='^')
    ax0.set_xlim((np.min(x_vector), np.max(x_vector)))
    ax0.set_ylim((np.min(y_vector), np.max(y_vector)))
    ax0.scatter(np.array(xs_vector)[plot_bounds], np.array(ys_vector)[plot_bounds],
                c=np.array(np.arange(num_iterations))[plot_bounds], marker='o',
                s=35, cmap=cm.coolwarm)
    ax0.scatter(xs_start, ys_start, marker='o', s=35, facecolors=None, edgecolors='black')
    # plt.scatter(xs_vector[plot_bounds], ys_vector[plot_bounds], c='b', marker='o', s=35)
    # plt.scatter(xs2_vector[plot_bounds], ys2_vector[plot_bounds], c='r', marker='o', s=35)
    ax0.plot([x3, xs_syn], [y3, ys_syn], c='k')
    ax0.plot([x3, xs_vector[-1]], [y3, ys_vector[-1]], c='k')
    ax0.scatter(xs_syn, ys_syn, marker='*', c='k', s=40)
    ax0.set_xlabel('X Distance', fontweight='bold')
    ax0.set_ylabel('Y Distance', fontweight='bold')

    ax1 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
    ax1.scatter(np.array(iteration_vector)[plot_bounds], np.array(misfit_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Misfit', fontweight='bold')
    ax1.set_yscale('log')

    ax2 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
    ax2.scatter(np.array(iteration_vector)[plot_bounds], np.array(theta_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax2.axhline(theta_deg_syn, c='k', linestyle='dashed')
    ax2.set_xlabel('Iteration', fontweight='bold')
    ax2.set_ylabel('Theta', fontweight='bold')

    ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
    ax3.scatter(np.array(iteration_vector)[plot_bounds], np.array(ts_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax3.axhline(ts_syn, c='k', linestyle='dashed')
    ax3.set_xlabel('Iteration', fontweight='bold')
    ax3.set_ylabel('Origin Time (s)', fontweight='bold')



    fig1.tight_layout()
    fig1.subplots_adjust(top=0.9)
    fig1.suptitle(f'Params: Source: ({xs_syn}, {ys_syn}, {str(np.round(ts_syn, decimals=1))}), Start: ({xs_start}, {ys_start}) '
                  f': Theta difference = {np.round(theta_diff, 1)} degrees')
    fig1.savefig(f'{output_directory}{input_parameters}/source_{xs_syn}_{ys_syn}_start_{xs_start}_{ys_start}.png')
    # fig1.savefig(f'{output_directory}{input_parameters}/source_{xs_syn}_{ys_syn}_start_{xs_start}_{ys_start}.eps')

    return theta_diff, xs_start, ys_start, ts_start, xs_vector[-1], ys_vector[-1], ts_vector[-1]


def get_timings(geo_locs, avg_velocity, xs_syn, ys_syn, ts_syn):
    """
    Gets the arrival timings at the geophones from the random source

    :param geo_locs: [array] XY locations of the geophones
    :param avg_velocity: [float] Moonquake velocity
    :param xs_syn: [float] Horizontal location (x) of the synthetic source
    :param ys_syn: [float] Vertical location (y) of the synthetic source
    :param ts_syn: [float] Start time of the synthetic source
    :return:
    """

    # Propagate the source to the array to find the arrival time
    new_rel_time = []
    for geo_ind in np.arange(len(geo_locs)):

        # Find the distance between the source and the array
        stax = geo_locs[geo_ind][0]
        stay = geo_locs[geo_ind][1]
        dist = np.sqrt((xs_syn-stax)**2 + (ys_syn-stay)**2)
        new_rel_time.append((dist/avg_velocity) + ts_syn)

    return new_rel_time


def synthetic_wrapper(input_parameters, xs_syn, ys_syn, ts_syn, xs, ys):
    """
    Wrapper code for the location algorithm using synthetic data.

    :param input_parameters: [str] Designation of the parameter run
    :param xs_syn: [float] Horizontal location (x) of the synthetic source
    :param ys_syn: [float] Vertical location (y) of the synthetic source
    :param ts_syn: [float] Start time of the synthetic source
    :param xs: [float] Random horizontal (x) start location of Stochastic Gradient Descent
    :param ys: [float] Random vertical (y) start location of Stochastic Gradient Descent
    :return:
    """
    # Set locations of each geophone relative to geophone 3, the center of the array (in meters)
    geo1 = np.multiply((0.0455, 0.0341), 1000)
    geo2 = np.multiply((-0.0534, 0.0192), 1000)
    geo3 = (0, 0)
    geo4 = np.multiply((0.0119, -0.0557), 1000)
    geo_locs = [geo1, geo2, geo3, geo4]

    # Read in the data into a dataframe
    avg_vel_synthetic = velval
    avg_velocity = 34.0
    new_rel_time_perfect = get_timings(geo_locs, avg_vel_synthetic, xs_syn, ys_syn, ts_syn)

    new_rel_time = []
    for perfect_val in new_rel_time_perfect:
        new_rel_time.append(np.round(perfect_val, decimals=1))

    # Setup the parameter space
    x_vector = np.arange(-2000, 2001)
    y_vector = np.arange(-2000, 2001)

    # Solve for the location
    theta_diff, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin = \
        model(x_vector, y_vector, new_rel_time, avg_velocity, geo_locs, ts_syn, xs_syn, ys_syn,
              input_parameters, xs, ys)

    return theta_diff, xs_syn, ys_syn, ts_syn, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin


def test_params(input_parameters):
    """
    Wrapper code for the parameters

    :param input_parameters: [str] Designation for the parameter run
    :return:
    """
    # Check if the combination has already been tested
    if len(glob.glob(f'{output_directory}{input_parameters}/*.csv')) > 0:
        print(f'{input_parameters} already teststed! Skipping...')
        return

    theta_diff_vector = []
    syn_source_vector = []
    rand_start_vector = []
    sgd_end_vector = []

    # Setup the random start
    np.random.seed(seed=seed_number)
    xs_syn_vector = np.random.choice(np.arange(-2000, 2000, 1), num_examples)
    ys_syn_vector = np.random.choice(np.arange(-2000, 2000, 1), num_examples)
    ts_syn_vector = np.random.choice(np.arange(-50, 0, 0.1), num_examples)
    x_vector = np.arange(-2000, 2000)
    y_vector = np.arange(-2000, 2000)
    xs_vector = np.random.choice(x_vector, num_examples)
    ys_vector = np.random.choice(y_vector, num_examples)

    for example_ind in np.arange(num_examples):
        theta_diff_out, xs_syn_out, ys_syn_out, ts_syn_out, xs_start_out, ys_start_out, ts_start_out, \
        xs_fin_out, ys_fin_out, ts_fin_out = synthetic_wrapper(input_parameters, xs_syn_vector[example_ind],
                                                               ys_syn_vector[example_ind], ts_syn_vector[example_ind],
                                                               xs_vector[example_ind], ys_vector[example_ind])

        theta_diff_vector.append(np.round(theta_diff_out, 3))
        syn_source_vector.append((xs_syn_out, ys_syn_out, np.round(ts_syn_out, 1)))
        rand_start_vector.append((xs_start_out, ys_start_out, ts_start_out))
        sgd_end_vector.append((np.round(xs_fin_out), np.round(ys_fin_out), np.round(ts_fin_out, 1)))

    theta_average = np.mean(theta_diff_vector)
    combined_data = list(zip(syn_source_vector, rand_start_vector, sgd_end_vector, theta_diff_vector))
    df = pd.DataFrame(combined_data, columns=['syn_source', 'rand_start_vector', 'sgd_end_vector', 'theta_diff'])
    df.to_csv(f'{output_directory}{input_parameters}/theta_num{num_examples}_avgtheta_{np.round(theta_average, 1)}.csv',
              index=False)
    print(f'Finished saving parameters for {input_parameters}!')

    return


def assess_accuracy(input_dir):
    """
    Assesses the accuracy within all the parameters

    :param input_dir: [str] Path to directory where the individual results are kept
    :return:
    """
    # Get a list of all the parameters
    param_list = sorted(glob.glob(f'{input_dir}*'))
    params = []
    for param in param_list:
        params.append(os.path.basename(param))

    # Cycle through each param and get the accuracy
    mean_difference, stdev_difference = [], []
    for param in params:
        result_file = glob.glob(f'{input_dir}{param}/*.csv')[0]
        df = pd.read_csv(result_file)
        theta_differences = df['theta_diff'].values
        mean_difference.append(np.mean(theta_differences))
        stdev_difference.append(np.std(theta_differences))

        fig, axs = plt.subplots(1, 1, figsize=(16, 8), num=20, clear=True)
        axs.errorbar(np.arange(len(theta_differences)), theta_differences, ecolor='black', marker='o', mfc='blue')
        axs.set_xticks(np.arange(len(theta_differences)))
        axs.set_ylabel('Theta Difference', fontweight='bold')
        axs.set_yscale('log')
        axs.set_xlabel('Synthetic Test Number', fontweight='bold')
        axs.set_title(f'Synthetic results for run {param}: mean={np.mean(theta_differences)}, stdev={np.std(theta_differences)}', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{final_results_directory}{param}_accuracy.png')
        # fig.savefig(f'{final_results_directory}{param}_accuracy.eps')

    # Find the set of parameters with lowest error rate
    least_min_index = np.where(mean_difference == np.min(mean_difference))[0][0]
    best_run = params[least_min_index]

    # Plot the result
    fig20, axs = plt.subplots(1, 1, figsize=(16, 8), num=20, clear=True)
    axs.errorbar(params, mean_difference, yerr=stdev_difference, ecolor='black', marker='o', mfc='blue')
    axs.set_ylabel('Mean', fontweight='bold')
    axs.set_yscale('log')
    axs.set_xlabel('Runs', fontweight='bold')
    fig20.autofmt_xdate()
    axs.set_title(f'Mean and standard deviation for all runs. Best: {best_run}', fontweight='bold')
    fig20.tight_layout()

    fig20.savefig(f'{final_results_directory}allrun_accuracy.png')
    fig20.savefig(f'{final_results_directory}allrun_accuracy.eps')

    return


# Main
outdir = 'C:/data/lunar_output/'
output_directory_out = f'{outdir}results/synthetic_verification_velvar/'
final_results_directory_out = f'{outdir}results/synthetic_results_velvar/'
seed_number = 1
if not os.path.exists(output_directory_out):
    os.mkdir(output_directory_out)
if not os.path.exists(final_results_directory_out):
    os.mkdir(final_results_directory_out)

# Select the number of cpu cores
num_cores = 10

# Find a number of examples to iterate
num_examples = 20

# Initialize the velocity values that we will test against our presumed velocity of 34 m/s
velvalues = np.arange(20, 110, 10)

for velval in velvalues:
    output_directory = f'{output_directory_out}v{velval}/'
    final_results_directory = f'{final_results_directory_out}v{velval}/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(final_results_directory):
        os.mkdir(final_results_directory)

    # Select the space and time steps to cycle through
    # space_step_windows = [50, 75, 100, 125, 150, 175, 200]
    # time_step_windows = [0.05, 0.1, 0.15, 0.2, 0.25]

    space_step_windows = [200]
    time_step_windows = [0.05]

    # Find all the parameter combinations
    param_combinations = []
    for space_step_win in space_step_windows:
        for time_step_win in time_step_windows:
            if not os.path.exists(f'{output_directory}{space_step_win}_{time_step_win}'):
                os.mkdir(f'{output_directory}{space_step_win}_{time_step_win}')
            param_combinations.append(f'{space_step_win}_{time_step_win}')

    if num_cores == 1:
        for param_combination in param_combinations:
                test_params(param_combination)
    else:
        Parallel(n_jobs=num_cores)(delayed(test_params)(param_combination)
                                   for param_combination in param_combinations)

    assess_accuracy(output_directory)



