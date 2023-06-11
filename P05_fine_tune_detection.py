"""
This code fine-tunes the detections originally produced by the Civilini 2021 paper.

"""
# import packages
import pickle
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from matplotlib import cm
from obspy.signal.trigger import classic_sta_lta, plot_trigger
import os
import datetime
import pandas as pd
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Setup functions
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


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


def run_highpass_filter(input_data, input_corner_freq, sr):
    """
    Passes a highpass filter on the data

    :param input_data: [obspy trace] Seismic input trace in Obspy format
    :param input_corner_freq: [int] Corner frequency of highpass filter
    :param sr: [float] Sampling rate
    :return:
    """
    # Remove the mean and run a cosine taper
    trace_nomean = input_data.data - np.mean(input_data.data)
    N = len(trace_nomean)
    taper_function = cosine_taper(N, p=0.1)
    trace_taper = trace_nomean * taper_function

    # Highpass filterbound in Hertz
    data_filt = highpass(trace_taper, input_corner_freq, sr, corners=4, zerophase=False)

    return data_filt


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def f_sta_lta2(t, f, Sxx, avg_Sxx, rel_detection, win, time, data, in_geophone_name, evid, specmax=1e-6):
    """
    Conducts a sta-lta on the frequency-domain representaiton of the trace around a small window around the detection.
    :param t: [vector] Time values from the spectrogram
    :param f: [vector] Frequency values from the spectrogram
    :param Sxx: [vector] Spectrogram values
    :param avg_Sxx: [vector] Average spectrogram values
    :param rel_detection: [int] Arrival of trace in seconds
    :param win: [int] Number of seconds before and after the relative detection that we want to alter the pick
    :param specmax: [float] Spectrogram color maximum
    :return:
    """

    # Run a median filter on the average frequency
    sr = 1 / (t[2] - t[1])
    mv_avg_secs = mov_med_win_val
    avg_samples = sr * mv_avg_secs
    num_samples = int(np.round(mv_avg_secs * avg_samples))

    mv_med_avg_Sxx = running_median(np.array(avg_Sxx), num_samples)

    # Compute a moving average of the gradient slope
    grad = np.gradient(mv_med_avg_Sxx)
    grad_moving_average_sec = grad_mov_avg_val
    grad_moving_average_pts = int(grad_moving_average_sec * sr)
    d = pd.Series(grad)
    grad_moving_avg = d.rolling(grad_moving_average_pts).mean()

    # Change the grad moving averages that are zero to NANs. They are equivalent to the NAN values in the time series.
    zero_grad_moving_indices = np.where(abs(grad_moving_avg) < 10e-30)[0]
    if len(zero_grad_moving_indices) > 0:
        grad_moving_avg.values[zero_grad_moving_indices] = np.nan

    # If the values of the moving grad are not AT LEAST greater than a certain value (i.e. they are super small) do not
    # use them
    grad_2_binary_multiplyer = mult_grad_bin_val
    positive_grad_ind = np.where(grad_moving_avg > grad_2_binary_multiplyer * np.mean(abs(grad_moving_avg)))[0]

    # Do the same thing with the negative values
    negative_grad_ind = np.where(grad_moving_avg < -1 * grad_2_binary_multiplyer * np.mean(abs(grad_moving_avg)))[0]

    # Create a vector such that every positive element is zeros and every positive element is one.
    grad_binary = np.empty(len(grad))
    grad_binary[:] = np.NaN
    grad_binary[positive_grad_ind] = 1
    grad_binary[negative_grad_ind] = 0

    # Fill in the nans that are just due to gaps.
    # First, find Nans. If the closest real values on both sides are less than 5 seconds AND they both have the same
    # value (i.e. both zeros or both ones), fill in.
    gap_value_seconds = gap_fill_time_val
    gap_value_num = int(gap_value_seconds * sr)
    # grad_interp = grad_binary.copy()

    # Find the nan indices and cycle through them
    nan_indices = np.argwhere(np.isnan(grad_binary))
    grad_interp = []
    for binary_index in np.arange(len(grad_binary)):
        # If it's the start or end of the trace and it's an NAN, add NAN.
        if binary_index == 0 or binary_index == len(grad_binary):
            if np.isnan(grad_binary[binary_index]):
                grad_interp.append(np.nan)
                continue
            else:
                grad_interp.append(grad_binary[binary_index])
                continue

        # If it's a non-NAN value, just add it
        if not np.isnan(grad_binary[binary_index]):
            grad_interp.append(grad_binary[binary_index])
            continue

        # If it's a NAN value but it's between two of the same binary values within the gap index, interpolate it
        if np.isnan(grad_binary[binary_index]):
            if binary_index - gap_value_num <= 0:
                lower_elements = grad_binary[0:binary_index]
            else:
                lower_elements = grad_binary[binary_index-gap_value_num: binary_index]
            if binary_index + gap_value_num > len(grad_binary):
                upper_elements = grad_binary[binary_index:len(grad_binary)]
            else:
                upper_elements = grad_binary[binary_index:binary_index + gap_value_num]

            # Find the first non-nan values of each bound. If they are the same, add the value. If not, add nan
            first_nonnan_lower_element = []
            for lower_element_ind in np.arange(len(lower_elements)):
                if not np.isnan(lower_elements[lower_element_ind]):
                    first_nonnan_lower_element.append(lower_elements[lower_element_ind])

            first_nonnan_upper_element = []
            for upper_element_ind in np.arange(len(upper_elements)):
                if not np.isnan(upper_elements[upper_element_ind]):
                    first_nonnan_upper_element.append(upper_elements[upper_element_ind])

            if len(first_nonnan_lower_element) == 0 or len(first_nonnan_upper_element) == 0:
                grad_interp.append(np.nan)
                continue
            else:
                if first_nonnan_lower_element[0] == 0 and first_nonnan_upper_element[0] == 0:
                    grad_interp.append(0)
                    continue
                elif first_nonnan_lower_element[0] == 1 and first_nonnan_upper_element[0] == 1:
                    grad_interp.append(1)
                    continue
                else:
                    grad_interp.append(np.nan)
                    continue

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(t, grad_binary)
    # plt.xlim((t[0], t[-1]))
    # plt.subplot(2, 1, 2)
    # plt.plot(t, grad_interp)
    # plt.xlim((t[0], t[-1]))

    sustained_seconds = 2
    sustained_elements = int(sustained_seconds * sr)

    repeating_positive = []
    for grad_binary_ind in np.arange(len(grad_interp) - sustained_elements):
        if sum(grad_interp[grad_binary_ind:grad_binary_ind + sustained_elements]) == \
                len(grad_interp[grad_binary_ind:grad_binary_ind + sustained_elements]):
            repeating_positive.append(grad_binary_ind)

    # In some cases, the repeated values may not just be at the event arrival (e.g. multiple events in window)
    # Therefore, only look for the start that is in the window of the relative arrival
    rel_detection_upper_ind = find_nearest(t, rel_detection + win)
    rel_detection_lower_ind = find_nearest(t, rel_detection - win)
    repeating_positive_upper_pass_ind = np.where(repeating_positive < rel_detection_upper_ind)
    repeating_positive_lower_pass_ind = np.where(repeating_positive > rel_detection_lower_ind)
    repeating_positive_win = np.intersect1d(repeating_positive_upper_pass_ind, repeating_positive_lower_pass_ind)

    # Add condition of when there is no repeating positive in the window
    if len(repeating_positive) == 0 or len(repeating_positive_win) == 0:
        new_arrival = rel_detection
        new_arrival_index = find_nearest(t, rel_detection)
    else:
        new_arrival = t[repeating_positive[repeating_positive_win[0]]]
        new_arrival_index = repeating_positive[repeating_positive_win[0]]

    # Find the repeating negative values
    repeating_negative = []
    for grad_binary_ind in np.arange(len(grad_interp) - sustained_elements):
        if sum(grad_interp[grad_binary_ind:grad_binary_ind + sustained_elements]) == 0:
            repeating_negative.append(grad_binary_ind + sustained_elements)

    # Find the end of the repeating negative elements closest to the new arrival index
    # First, cut out all the events before the new arrival index
    if len(np.where(repeating_negative < new_arrival_index)[0]) > 0:
        repeating_negative_cut = np.delete(repeating_negative, np.where(repeating_negative < new_arrival_index)[0])
    else:
        repeating_negative_cut = repeating_negative

    # Next, find the index of repeating negative which does NOT increase by one.
    # The index just before that element should be the end of the trace.
    non_incremental_indices = []
    for increment_ind in np.arange(len(repeating_negative_cut) - 1):
        if repeating_negative_cut[increment_ind + 1] - repeating_negative_cut[increment_ind] > 1:
            non_incremental_indices.append(increment_ind)

    # This is just for signals with low signal to noise ratios. If there isn't such a problem, then it means that
    # the signal is impulsive and we can just use the end of the original negative vector
    if len(non_incremental_indices) == 0:
        if len(repeating_negative) > 0:
            envelope_end_ind = repeating_negative[-1]
        else:
            envelope_end_ind = new_arrival_index
    else:
        envelope_end_ind = repeating_negative_cut[non_incremental_indices[0]]

    if len(repeating_negative_cut) == 0:
        envelope_length = 0
        envelope_end_time = new_arrival + envelope_length
    else:
        envelope_length = t[envelope_end_ind] - new_arrival
        envelope_end_time = new_arrival + envelope_length

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(t, grad_moving_avg)
    # plt.axvline(t[new_arrival_index], c='r')
    # plt.axvline(t[envelope_end_ind], c='g')
    # plt.xlim((t[0], t[-1]))
    # plt.subplot(2, 1, 2)
    # plt.plot(t, grad_interp)
    # plt.axvline(t[new_arrival_index], c='r')
    # plt.axvline(t[envelope_end_ind], c='g')
    # plt.xlim((t[0], t[-1]))

    # Find the maximum of the envelope by looking at the maximum of the GRADIENT
    # If the envelope length is zero, then keep the same detection and don't go any further
    if envelope_length <= 1:
        emergence = 0
        max_envelope_time = t[new_arrival_index]
    else:
        max_envelope_index = np.where(grad_moving_avg == np.max(grad_moving_avg
                                                                [new_arrival_index:envelope_end_ind]))[0]

        # Somtimes the maximum value occurs over a small number of samples. If this happens, just take the first element
        # of the vector
        if len(max_envelope_index) > 1:
            max_envelope_index = max_envelope_index[0]
        max_envelope_time = t[max_envelope_index]

        # Calculate emergence by dividing the time that it takes to reach the max envelope time by the envelope length
        emergence = max_envelope_time-t[new_arrival_index]
        if not emergence.dtype == 'float64':
            emergence = emergence[0]

    # Find the signal-to-noise ratio of the time domain
    # Find the pre-event RMS and absolute noise. We start 30 seconds prior to the event and stop 5 wseconds short of the
    # arrival time
    noise_window_sec = 30
    sr_time_domain = 1 / (time[2] - time[1])
    noise_window_pts = int(noise_window_sec * sr_time_domain)

    # Find the RMS and maximum of the noise window.
    # If the noise window is greater than the start of the file, make the window start the beginning of the file
    if new_arrival_index - noise_window_pts <= 0:
        noise_rms = np.sqrt(np.mean(np.absolute(data[0:new_arrival_index - int(5 * sr)]) ** 2))
        noise_max = np.nanmax(abs(data[0:new_arrival_index - int(5 * sr)]))
    else:
        noise_rms = np.sqrt(np.mean(np.absolute(data[new_arrival_index - noise_window_pts:
                                                     new_arrival_index - int(5 * sr)]) ** 2))
        noise_max = np.nanmax(abs(data[new_arrival_index - noise_window_pts:
                                       new_arrival_index - int(5 * sr)]))

    # Find the maximum amplitude of the time domain
    if envelope_end_ind - new_arrival_index <= 0:
        maxamp = data[new_arrival_index]
    else:
        maxamp = np.max(data[new_arrival_index:envelope_end_ind])

    # Find both the snr (using noise rms) and practical SNR (using noise max)
    snr_from_rms = float(str(np.round(maxamp / noise_rms, decimals=1)))
    snr_from_max = float(str(np.round(maxamp / noise_max, decimals=1)))

    # Create a stats vector with all the information
    stats_vector = [new_arrival, envelope_length, envelope_end_time, emergence, max_envelope_time,
                    noise_rms, snr_from_rms, noise_max, snr_from_max]

    fig = plt.figure(figsize=(13, 14), num=2, clear=True)
    ax0 = fig.add_subplot(6, 1, 1)
    ax0.plot(time, data)
    ax0.set_title(f'{in_geophone_name} evid:{evid}', fontweight='bold')
    ax0.axvspan(rel_detection, rel_detection + win, alpha=0.5, color='gray')
    ax0.axvspan(rel_detection, rel_detection - win, alpha=0.5, color='gray')
    ax0.axvline(x=new_arrival, c='r')
    ax0.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax0.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax0.set_xlim((t[0], t[-1]))

    ax1 = fig.add_subplot(6, 1, 2)
    ax1.pcolormesh(t, f, Sxx, cmap=cm.jet, vmax=specmax, shading='auto')
    ax1.axvline(x=new_arrival, c='r')
    ax1.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax1.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax1.set_title('Spectrogram')
    ax1.set_xlim((t[0], t[-1]))

    ax2 = fig.add_subplot(6, 1, 3)
    ax2.plot(t, avg_Sxx)
    ax2.axvspan(rel_detection, rel_detection + win, alpha=0.5, color='gray')
    ax2.axvspan(rel_detection, rel_detection - win, alpha=0.5, color='gray')
    ax2.axvline(x=new_arrival, c='r')
    ax2.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax2.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax2.set_title('Average Spectrogram')
    ax2.set_xlim((t[0], t[-1]))

    ax3 = fig.add_subplot(6, 1, 4)
    ax3.plot(t, mv_med_avg_Sxx, color='orange')
    ax3.axvspan(rel_detection, rel_detection + win, alpha=0.5, color='gray')
    ax3.axvspan(rel_detection, rel_detection - win, alpha=0.5, color='gray')
    ax3.set_title('FFT moving median')
    ax3.axvline(x=new_arrival, c='r')
    ax3.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax3.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax3.set_xlim((t[0], t[-1]))

    ax4 = fig.add_subplot(6, 1, 5)
    ax4.plot(t, grad_moving_avg)
    ax4.axvspan(rel_detection, rel_detection + win, alpha=0.5, color='gray')
    ax4.axvspan(rel_detection, rel_detection - win, alpha=0.5, color='gray')
    ax4.set_title('Gradient Moving Avg.')
    ax4.axvline(x=new_arrival, c='r')
    ax4.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax4.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax4.set_xlim((t[0], t[-1]))

    ax5 = fig.add_subplot(6, 1, 6)
    ax5.scatter(t, grad_interp, s=0.1)
    ax5.axvspan(rel_detection, rel_detection + win, alpha=0.5, color='gray')
    ax5.axvspan(rel_detection, rel_detection - win, alpha=0.5, color='gray')
    ax5.set_title('Gradient Binary')
    ax5.axvline(x=new_arrival, c='r')
    ax5.axvline(x=t[envelope_end_ind], c='g', linewidth=2.5)
    ax5.axvline(x=max_envelope_time, c='magenta', linewidth=1.5)
    ax5.set_xlim((t[0], t[-1]))

    fig.tight_layout()

    fig.savefig(f'{output_directory}combined_images_full/{evid}/{in_geophone_name}_{evid}_spectr_analysis.png')
    # fig.savefig(f'{output_directory}combined_images_full/{evid}/{in_geophone_name}_{evid}_spectr_analysis.eps')
    # plt.close()

    return stats_vector, mv_med_avg_Sxx, grad_moving_avg, grad_interp


def check_exceptions(evid):
    """
    Certain exceptions need to be made on the data from human assessment.
    This function checks whether or not the evid requires different actions

    except_param = 0 (No exceptions found)
    except_param = 1 (Use a smaller window for finetuning)

    :param evid: [str] Event id
    :return:
    """
    except_param = 0
    # For the following evids, keep the original detections
    if evid == '761029-15-M1' or evid == '770125-05-M1' or evid == '770422-09-M1':
        except_param = 1

    return except_param

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


def running_median(seq, win):
    """
    Conducts a running median on the data

    @param seq: [Vector] Input data
    @param win: [Integer] Size of the window (in samples)
    @return:
    """

    samples = np.arange(len(seq))
    medians = []

    window_middle = int(np.ceil(win / 2))

    for ind in np.arange(len(seq)):

        if ind <= window_middle:
            medians.append(np.median(abs(seq[0:win])))

        if ind >= len(seq) - window_middle:
            medians.append(np.median(abs(seq[len(seq) - win:len(seq)])))

        if window_middle < ind < len(seq) - window_middle:
            medians.append(np.median(abs(seq[ind - int(np.floor(win / 2)):ind + int(np.floor(win / 2))])))

    return np.array(medians)


def f_finetune(time_vector, data_vector, rel_detection, in_geophone_name, evid):
    """
    Returns the new detection after computing a spectrogram
    :param time_vector: [vector] Time vector
    :param data_vector: [vector] Seismic trace vector
    :param rel_detection: [int] Time of original detection
    :param in_geophone_name: [str] Name of the geophone
    :param evid: [str] Event ID
    :return:
    """

    # Samplerate is not an integer
    sr = 1 / (time_vector[1] - time_vector[0])
    trace_filtered = run_highpass_filter(data_vector, 10.0, sr)
    # f, t, Sxx = signal.spectrogram(trace_filtered, sr, nperseg=16)
    f, t, Sxx = signal.spectrogram(trace_filtered, sr, nperseg=8)
    t = t + time_vector[0]

    # Get the spacing between samples of the frequency domain
    deltat_freq = t[2] - t[1]

    # Find the average spectrogram for all traces
    # Make the average Sxx only between frequencies of 10-20 Hz
    # In this case, it's only for frequencies 14.7 Hz and 22.0 Hz, indices 2 and 3 in f
    avg_Sxx = []
    for time_ind in np.arange(len(t)):
        avg_Sxx.append(np.mean(Sxx[:, time_ind]))
        # avg_Sxx.append(np.mean(Sxx[2:3, time_ind]))

    # Check exception files
    # Select a window-length (seconds) of +- around the original detection to fine-tune the pick
    except_param = check_exceptions(evid)
    if except_param == 1:
        print(f'Exception for {evid}! Use a smaller detection window.')
        win = 5
    else:
        win = 30

    stats_vector, mv_med_avg_Sxx, grad_moving_avg, grad_binary = f_sta_lta2(t, f, Sxx, avg_Sxx, rel_detection, win,
                                                                            time_vector, data_vector, in_geophone_name,
                                                                            evid)

    return t, f, Sxx, avg_Sxx, stats_vector, mv_med_avg_Sxx, grad_moving_avg, grad_binary


def plot_combined_full(time_array_cut, data_array_cut, spec_t_array, spec_f_array, stats_array,
                       Sxx_array, mv_med_avg_Sxx_array, grad_moving_avg_array, avg_Sxx_array, evid,
                       rel_det_vector, grad_binary_array):
    """

    :param time_array_cut: [np_array] 2D array of time series time
    :param data_array_cut: [np_array] 2D array of time series data
    :param spec_t_array: [np_array] 2D array of spectrogram time
    :param spec_f_array: [np_array] 2D array of spectrogram frequency
    :param stats_array: [np_array] Array of parameters:
            [0] new_arrival, [1] envelope_length, [2] envelope_end_time, [3] emergence, [4] max_envelope_time,
            [5] noise_rms, [6] snr_from_rms, [7] noise_max, and [8] snr_from_max
    :param Sxx_array: [np_array] 3D array of sepctrogram values, with depth correpsonding to geophone
    :param mv_med_avg_Sxx_array: [np_array] 2D array of the moving median of the average spectogram values
    :param grad_moving_avg_array: [np_array] 2D array of the Gradient Moving Averages
    :param avg_Sxx_array: [np_array] 2D array of the average spectrogram values
    :param evid: [str] event ID
    :param rel_det_vector: [vector] Relative detections (i.e. original detections)
    :return:
    """

    # We will plot by iterating across each geophone
    # This way our codes will be flexible when it comes to events with grade lower than A (i.e. don't have 4 geophones)
    num_geophones = np.shape(time_array_cut)[1]

    # Check exception files
    # Select a window-length (seconds) of +- around the original detection to fine-tune the pick
    except_param = check_exceptions(evid)
    if except_param == 1:
        print(f'Exception for {evid}! Use a smaller detection window.')
        win = 5
    else:
        win = 30

    # Find a good ylimit for plotting the traces
    # Cut the traces for what we are plotting below
    max_trace_ylim = []
    for geophone_ind in np.arange(num_geophones):
        max_trace_ylim.append(np.max([np.max(data_array_cut[:, geophone_ind]), abs(np.min(data_array_cut[:, geophone_ind]))]))
    trace_plot_ymax = np.max(max_trace_ylim)

    # Start the iteration and plot
    fig, axs = plt.subplots(6, num_geophones, figsize=(16, 8), num=5, clear=True)
    for geophone_ind in np.arange(num_geophones):

        # Plot the time series
        axs[0, geophone_ind].plot(time_array_cut[:, geophone_ind], data_array_cut[:, geophone_ind], "b-")
        axs[0, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]), np.max(time_array_cut[:, geophone_ind])))
        axs[0, geophone_ind].set_ylabel('Amplitude (counts)', fontweight='bold')
        axs[0, geophone_ind].set_xlabel('Time (s)', fontweight='bold')

        axs[0, geophone_ind].set_ylim((-1*trace_plot_ymax, trace_plot_ymax))
        axs[0, geophone_ind].axvline(x=rel_det_vector[geophone_ind], c='k')
        axs[0, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[0, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[0, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)
        axs[0, geophone_ind].set_title(f'Geo{geophone_ind+1} (t={np.round(stats_array[0][geophone_ind], 3)} sec)')

        # Plot the spectrogram
        specmax = 1e-6
        specax = axs[1, geophone_ind].pcolormesh(spec_t_array[:, geophone_ind], spec_f_array[:, geophone_ind],
                                                 Sxx_array[:, :, geophone_ind], cmap=cm.jet, vmax=specmax,
                                                 shading='auto')
        axs[1, geophone_ind].set_ylabel('Frequency (Hz)', fontweight='bold')
        axs[1, geophone_ind].set_xlabel('Time (sec)', fontweight='bold')
        axs[1, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]), np.max(time_array_cut[:, geophone_ind])))

        axs[1, geophone_ind].axvline(x=rel_det_vector[geophone_ind], c='k')
        axs[1, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[1, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[1, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)

        # Plot the average spectrogram
        axs[2, geophone_ind].plot(spec_t_array[:, geophone_ind], avg_Sxx_array[:, geophone_ind], "b-")
        axs[2, geophone_ind].set_ylabel('Avg. Spectr.', fontweight='bold')
        axs[2, geophone_ind].set_xlabel('Time (s)', fontweight='bold')
        axs[2, geophone_ind].axvspan(rel_det_vector[geophone_ind], rel_det_vector[geophone_ind] + win,
                                     alpha=0.5, color='gray')
        axs[2, geophone_ind].axvspan(rel_det_vector[geophone_ind], rel_det_vector[geophone_ind] - win,
                                     alpha=0.5, color='gray')
        axs[2, geophone_ind].axvline(x=rel_det_vector[geophone_ind], c='k')
        axs[2, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[2, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)
        axs[2, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[2, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]),
                                       np.max(time_array_cut[:, geophone_ind])))

        # Plot the median values
        axs[3, geophone_ind].plot(spec_t_array[:, geophone_ind],
                                  mv_med_avg_Sxx_array[:, geophone_ind], color="orange")
        axs[3, geophone_ind].set_ylabel('Moving Median', fontweight='bold')
        axs[3, geophone_ind].set_xlabel('Time (s)', fontweight='bold')
        axs[3, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[3, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[3, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)
        axs[3, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]),
                                       np.max(time_array_cut[:, geophone_ind])))

        axs[4, geophone_ind].plot(spec_t_array[:, geophone_ind],
                                  grad_moving_avg_array[:, geophone_ind], color="gray")
        axs[4, geophone_ind].set_ylabel('Gradient Moving Avg.', fontweight='bold')
        axs[4, geophone_ind].set_xlabel('Time (s)', fontweight='bold')
        axs[4, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[4, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[4, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)
        axs[4, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]),
                                       np.max(time_array_cut[:, geophone_ind])))

        axs[5, geophone_ind].plot(spec_t_array[:, geophone_ind],
                                  grad_binary_array[:, geophone_ind], color="blue")
        axs[5, geophone_ind].set_ylabel('Gradient Binary', fontweight='bold')
        axs[5, geophone_ind].set_xlabel('Time (s)', fontweight='bold')
        axs[5, geophone_ind].axvline(x=stats_array[0][geophone_ind], c='r')
        axs[5, geophone_ind].axvline(x=stats_array[4][geophone_ind], c='magenta')
        axs[5, geophone_ind].axvline(x=stats_array[2][geophone_ind], c='green', linewidth=3.0)
        axs[5, geophone_ind].set_xlim((np.min(time_array_cut[:, geophone_ind]),
                                       np.max(time_array_cut[:, geophone_ind])))

    # Do tight_layout BEFORE adjusting the subplots to add the colorbar
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f'{evid}', fontweight='bold')
    # cbar_ax = fig.add_axes([0.30, 0.08, 0.4, 0.03])
    # cbar = fig.colorbar(specax, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('Spectral Energy', fontweight='bold')
    fig.savefig(f"{output_directory}combined_images_full/{evid}/GradeA_evid_{evid}.png")
    fig.savefig(f"{output_directory}combined_images_full/GradeA_evid_{evid}.png")
    # fig.savefig(f"{output_directory}combined_images_full/GradeA_evid_{evid}.eps")

    # plt.close('all')
    return


def finetune_wrapper(input_file):
    """
    Wrapper file for the finetuning process

    :param input_file: [str] Path to the input file to finetune
    :return:
    """

    # Get the evid from the filename
    input_file_bn = os.path.basename(input_file)
    evid = input_file_bn.split('_')[1].split('.')[0]

    # If it already exists, skip it
    if os.path.exists(f'{cat_stats_dir}GradeA_evid_{evid}_stats.csv'):
        print(f'Event {evid} has already been processed...')
        return

    # Load the file
    with open(input_file, 'rb') as f:
        time_array_cut, data_array_cut, abs_trace_start, abs_trace_end, \
        rel_det_vector, input_info, data_geophone_list = pickle.load(f)

    # Check how empty it is. If at least one of the geophones has missing data > 20% of the trace, skip it.
    geophone_mia_data_vector = np.zeros((4,1))
    for geophone_ind in np.arange(np.shape(time_array_cut)[1]):
        mia_percentage = len(np.where(abs(data_array_cut[:, geophone_ind]) < 1e-100)[0]) / \
                         len(data_array_cut[:, geophone_ind])
        if mia_percentage > 0.2:
            geophone_mia_data_vector[geophone_ind] = 1
    if np.sum(geophone_mia_data_vector) > 0:
        print(f'Missing data for {evid}! Skipping...')
        return

    # Finetune the detection
    num_geophones = np.shape(time_array_cut)[1]
    stats_array = np.zeros((9, num_geophones))
    for geophone_ind in np.arange(num_geophones):

        spec_t, spec_f, spec_Sxx, avg_Sxx, stats_vector, mv_med_avg_Sxx, grad_moving_avg, grad_binary = \
            f_finetune(time_array_cut[:, geophone_ind], data_array_cut[:, geophone_ind], rel_det_vector[geophone_ind],
                       data_geophone_list[geophone_ind], evid)

        # Combine each vector of information into arrays for plotting
        for element_ind in np.arange(len(stats_vector)):
            stats_array[element_ind, geophone_ind] = stats_vector[element_ind]
        if geophone_ind == 0:
            spec_t_array = np.zeros((len(spec_t), num_geophones))
            spec_f_array = np.zeros((len(spec_f), num_geophones))
            Sxx_array = np.zeros((np.shape(spec_Sxx)[0], np.shape(spec_Sxx)[1], num_geophones))
            mv_med_avg_Sxx_array = np.zeros((len(mv_med_avg_Sxx), num_geophones))
            grad_moving_avg_array = np.zeros((len(grad_moving_avg), num_geophones))
            avg_Sxx_array = np.zeros((len(avg_Sxx), num_geophones))
            grad_binary_array = np.zeros((len(avg_Sxx), num_geophones))

        # Now place each vector or array into their representative array (or 3D array in the case of Sxx)
        spec_t_array[:, geophone_ind] = spec_t
        spec_f_array[:, geophone_ind] = spec_f
        Sxx_array[:, :, geophone_ind] = spec_Sxx
        mv_med_avg_Sxx_array[:, geophone_ind] = mv_med_avg_Sxx
        grad_moving_avg_array[:, geophone_ind] = grad_moving_avg
        avg_Sxx_array[:, geophone_ind] = avg_Sxx
        grad_binary_array[:, geophone_ind] = grad_binary


    plot_combined_full(time_array_cut, data_array_cut, spec_t_array, spec_f_array, stats_array, Sxx_array,
                       mv_med_avg_Sxx_array, grad_moving_avg_array, avg_Sxx_array, evid, rel_det_vector, grad_binary_array)

    # Create a file with the updated catalog times
    # We will also create new variables for the original absolute time for debugging purposes
    # (comparison with original catalog)
    # We will also create an evid list that makes our life easier later for integration into the old catalog
    """
    [np_array] Array of parameters:
            [0] new_arrival, [1] envelope_length, [2] envelope_end_time, [3] emergence, [4] max_envelope_time,
            [5] noise_rms, [6] snr_from_rms, [7] noise_max, and [8] snr_from_max
    """
    # Make a vector list of all the parameters.
    # This list needs to be separated because elements will have different types of data.
    evid_list = []
    new_arrival_vector = []
    envelope_len_vector = stats_array[1][:]
    envelope_end_time_vector = []
    emergence_vector = stats_array[3][:]
    max_envelope_time_vector = []
    noise_rms_vector = stats_array[5][:]
    snr_rms_vector = stats_array[6][:]
    noise_max_vector = stats_array[7][:]
    snr_max_vector = stats_array[8][:]

    # Get absolute date values for the relative times in stats_array
    for geophone_ind in np.arange(num_geophones):
        orig_abs_time = datetime.datetime.strptime(cat.loc[(cat['evid'] == evid) &
                                                           (cat['station'] == f'geo{geophone_ind + 1}')]
                                                   ['abs_time'].values[0], "%Y-%m-%dT%H:%M:%S")
        new_absolute_arrival_time = orig_abs_time + datetime.timedelta(seconds=stats_array[0][geophone_ind] -
                                                                               rel_det_vector[geophone_ind])

        envelope_end_time = new_absolute_arrival_time + datetime.timedelta(seconds=envelope_len_vector[geophone_ind])
        max_envelope_time = new_absolute_arrival_time + datetime.timedelta(seconds=stats_array[4][geophone_ind] -
                                                                                   stats_array[0][geophone_ind])
        new_arrival_vector.append(new_absolute_arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        envelope_end_time_vector.append(envelope_end_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        max_envelope_time_vector.append(max_envelope_time.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        evid_list.append(evid)

    # Save the spectrogram information for plotting in P2.
    with open(f'{output_directory}ft_spectrograms/evid_spectr_{evid}.pkl', 'wb') as f:
        pickle.dump([spec_t_array, spec_f_array, Sxx_array], f)

    list_of_tuples = list(zip(evid_list, data_geophone_list, new_arrival_vector, envelope_len_vector,
                              envelope_end_time_vector, emergence_vector, max_envelope_time_vector, noise_rms_vector,
                              snr_rms_vector, noise_max_vector, snr_max_vector))
    df = pd.DataFrame(list_of_tuples, columns=['evid', 'geophone', 'ft_arrival_time', 'envelope_length',
                                               'envelope_end_time', 'emergence', 'max_envelope_time', 'noise_rms',
                                               'snr_from_rms', 'noise_max', 'snr_from_max'])
    df.to_csv(f'{cat_stats_dir}GradeA_evid_{evid}_stats.csv', index=False)

    print(f'Finished fine-tuning event {evid}...')

    return


# Main
data_outdir = 'C:/data/lunar_output/'
rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'
input_directory = f'{data_outdir}original_images/combined_files/'

# Setup the output directories
output_directory = f'{data_outdir}fine-tuned/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
cat_stats_dir = f'{data_outdir}fine-tuned/cat_stats/'
if not os.path.exists(cat_stats_dir):
    os.mkdir(cat_stats_dir)
if not os.path.exists(f'{output_directory}combined_images_full/'):
    os.mkdir(f'{output_directory}combined_images_full/')
if not os.path.exists(f'{output_directory}ft_spectrograms/'):
    os.mkdir(f'{output_directory}ft_spectrograms/')
if not os.path.exists(cat_stats_dir):
    os.mkdir(cat_stats_dir)

# Set the number of cores
num_cores = 3

# Get the original catalog
original_cat = f'{rundir}fc_lspe_dl_catalog.csv'
cat = pd.read_csv(original_cat)

# Create a list of all the geophones
total_geophone_list = ['geo1', 'geo2', 'geo3', 'geo4']

# Find the list of best parameters obtained from P4.
mov_med_win_val = 5
grad_mov_avg_val = 3
mult_grad_bin_val = 2
gap_fill_time_val = 10

# For presentation
filelist = glob.glob(f'{input_directory}*.pkl')

# For each event in filelist, create an output folder of the evid
for evt_file in filelist:
    input_bn = os.path.basename(evt_file)
    evid_name = input_bn.split('_')[1].split('.')[0]
    if not os.path.exists(f'{output_directory}combined_images_full/{evid_name}'):
        os.mkdir(f'{output_directory}combined_images_full/{evid_name}')

if num_cores == 1:
    for evt_file in filelist:
        print(f'Working on {evt_file}')
        finetune_wrapper(evt_file)
else:
    Parallel(n_jobs=num_cores)(delayed(finetune_wrapper)(evt_file)
                               for evt_file in filelist)

