"""
This code processes the LSPE geophone data from decompressed volts to physical units (nm/s)

"""
# Import Packages
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from obspy.core import read
from obspy.signal.invsim import cosine_taper
from scipy.signal import butter, lfilter, resample
from scipy import fftpack
import pickle
import math
import os
import datetime as dt
from obspy.io.sac.sactrace import SACTrace
import time
from obspy.signal.detrend import polynomial
from joblib import Parallel, delayed


# Functions
def find_nearest(array, value):
    """
    Finds the nearest index in an array for a value
    :param array: [vector] Input array to search
    :param value: [float] Value to find
    :return:
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


def preprocess(in_data):
    """
    Removes the mean and passes a cosine taper

    :param in_data: [vector] Input data
    :return:
    """

    trace_nomean = in_data - np.mean(in_data)
    n = len(trace_nomean)
    taper_function = cosine_taper(n, p=0.001)
    trace_taper = trace_nomean * taper_function

    return trace_taper


def read_sac(flist, ad_plots, day, filename, image_directory):
    """
    Converts the data from ascii to a pandas dataframe.

    Note: The time variable at this point is decimal day.

    :param flist: [list] List of SAC files to process
    :param ad_plots : [boolean] Whether to save additional images of each processing step.
    :param day : [str] The day of the ascii file
    :param image_directory: [str] Path to image directory output folder
    :return:
    """
    # Load the files for each geophone
    st_g1 = read(flist[0])
    st_g2 = read(flist[1])
    st_g3 = read(flist[2])
    st_g4 = read(flist[3])
    tr_g1 = st_g1[0]
    tr_g2 = st_g2[0]
    tr_g3 = st_g3[0]
    tr_g4 = st_g4[0]

    # Process the data
    g1_data_prep = preprocess(tr_g1.data)
    g2_data_prep = preprocess(tr_g2.data)
    g3_data_prep = preprocess(tr_g3.data)
    g4_data_prep = preprocess(tr_g4.data)

    # Get the time vector in seconds
    timevector = tr_g1.times()

    # Save the dataframe
    df = pd.DataFrame()
    df.insert(loc=0, column=f'Time', value=timevector)
    df.insert(loc=1, column=f'Geo1', value=g1_data_prep)
    df.insert(loc=2, column=f'Geo2', value=g2_data_prep)
    df.insert(loc=3, column=f'Geo3', value=g3_data_prep)
    df.insert(loc=4, column=f'Geo4', value=g4_data_prep)

    # Save the plots
    if ad_plots:

        fig = plt.figure(figsize=(11, 6), num=1, clear=True)
        ax0 = plt.subplot(2, 2, 1)
        ax0.plot(df['Time'].values, df['Geo1'].values)
        ax0.set_xlim((df['Time'].values[0], df['Time'].values[-1]))
        ax0.set_title('Geophone 1')
        ax0.set_xlabel('Time (s)')
        ax1 = plt.subplot(2, 2, 2)
        ax1.plot(df['Time'].values, df['Geo2'].values)
        ax1.set_xlim((df['Time'].values[0], df['Time'].values[-1]))
        ax1.set_title('Geophone 2')
        ax1.set_xlabel('Time (s)')
        ax2 = plt.subplot(2, 2, 3)
        ax2.plot(df['Time'].values, df['Geo3'].values)
        ax2.set_xlim((df['Time'].values[0], df['Time'].values[-1]))
        ax2.set_title('Geophone 3')
        ax2.set_xlabel('Time (s)')
        ax3 = plt.subplot(2, 2, 4)
        ax3.plot(df['Time'].values, df['Geo4'].values)
        ax3.set_xlim((df['Time'].values[0], df['Time'].values[-1]))
        ax3.set_title('Geophone 4')
        ax3.set_xlabel('Time (s)')
        fig.suptitle(f'{filename} uncompressed voltage', fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f'{image_directory}additional_images/{day}/{filename}_original.png')
        # plt.savefig(f'{output_folder}additional_images/{day}/{filename}_original.eps')

    return df


def butter_bandpass(in_lowcut, in_highcut, fs, in_bp_order):
    """
    Runs a bandpass butter filter

    @param in_lowcut: [Scalar] Bottom corner frequency
    @param in_highcut: [Scalar] Upper corner frequency
    @param fs: [Scalar] Sampling frequency
    @param in_bp_order: [Integer] Filter order
    @return:
    """
    nyq = 0.5 * fs
    low = in_lowcut / nyq
    high = in_highcut / nyq
    b, a = butter(in_bp_order, [low, high], btype='band')
    return b, a


def rm_freq_resp(df, ad_plots, day, filename, output_folder, resp_f, resp_vals, lowcut, highcut, bp_order, wl_cutoff,
                 image_directory):
    """
    Removes the frequency response from the data of all geophones. Geophone data is taken into the frequency domain and
    multiplied with the inverse of the response function. An inverse FFT is then used to bring the deconvoluted data
    back into the time domain.


    full-value:   Removes the full instrument response from the data. Careful filtering is required as it boosts the
                  instrument noise prevalent in the lower frequencies. Band-pass filtering and water-level corrections
                  are needed to mitigate this effecft.

    :param df: [pd df] Input pandas dataframe
    :param ad_plots : [boolean] Whether to save additional images of each processing step.
    :param day : [str] The day of the ascii file
    :param filename : [str] The name of the ascii file
    :param output_folder: [str] Path to output folder
    :param resp_f: [vector] Response function frequency values
    :param resp_vals: [vector] Response function amplitude values
    :param lowcut: [float] Lower corner value of bandpass filter (Hz)
    :param highcut: [float] Upper corner value of bandpass filter (Hz)
    :param bp_order: [int] Bandpass filter order
    :param wl_cutoff: [float] Water-level cutoff (Hz)
    :param image_directory: [str] Path to save image directory
    """

    # Load the time from the geophone
    geophone_time_seconds = df['Time'].values

    # Plot the instrument response used
    # We only want to do this once, so first check if it's present
    if not os.path.exists(f'{output_folder}images/p0_original_instrument_response.png'):
        fig3 = plt.figure(figsize=(6, 8), num=3, clear=True)
        ax0 = plt.subplot(3, 1, 1)
        ax0.plot(resp_f[0:len(resp_f) // 2], resp_vals[0:len(resp_f) // 2].real, color='red')
        ax0.set_xlim((resp_f[0:len(resp_f) // 2][0], resp_f[0:len(resp_f) // 2][-1]))
        ax0.set_title('Real Part (Amplitude)')
        ax1 = plt.subplot(3, 1, 2)
        ax1.plot(resp_f[0:len(resp_f) // 2], resp_vals[0:len(resp_f) // 2].imag, color='green')
        ax1.set_xlim((resp_f[0:len(resp_f) // 2][0], resp_f[0:len(resp_f) // 2][-1]))
        ax1.set_title('Imaginary Part (Phase)')
        ax2 = plt.subplot(3, 1, 3)
        ax2.plot(resp_f[0:len(resp_f) // 2], abs(resp_vals[0:len(resp_f) // 2]), color='blue')
        ax2.set_xlim((resp_f[0:len(resp_f) // 2][0], resp_f[0:len(resp_f) // 2][-1]))
        ax2.set_title('Absolute (Power)')
        fig3.suptitle('Original Response', fontweight='bold')
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig3.savefig(f'{image_directory}images/original_instrument_response.png')
        # plt.savefig(f'{image_directory}images/original_instrument_response.eps')

    # Remove the instrument response
    # This process will be conducted on one geophone at a time
    for geophone_ind in np.arange(4):
        geophone_data = df[f'Geo{geophone_ind + 1}'].values

        # Remove the mean from the geophone data
        geophone_data = geophone_data - np.mean(geophone_data)

        # Do an FFT on the geophone data
        # delta_target = 0.0085
        # sr_target = 1/delta_target
        sr_target = 117.7667
        delta_target = 1. / sr_target
        fft_f = fftpack.fftfreq(len(geophone_data), d=delta_target)
        fft_full = fftpack.fft(geophone_data)

        # Interpolate the instrument response according to the number of points within the data fft
        if len(resp_vals) < len(fft_f):
            # ffilt = interp1d(resp_f, resp_vals)
            resp_vals_interp = resample(resp_vals, len(fft_f))
            resp_f_interp = fft_f
        if len(resp_vals) > len(fft_f):
            resp_vals_interp = resample(resp_vals, len(fft_f))
            resp_f_interp = fft_f
        if len(resp_vals) == len(fft_f):
            resp_vals_interp = resp_vals
            resp_f_interp = fft_f

        # If you want to check the  interpolation with the data, uncomment below
        if ad_plots:
            fig4 = plt.figure(num=4, clear=True)
            ax0 = plt.subplot(1, 1, 1)
            ax0.plot(resp_f, resp_vals, c='r')
            ax0.plot(resp_f_interp, resp_vals_interp, c='b')
            ax0.legend(['Original response', 'Interpolated response'])
            ax0.set_title(f'Interpolated instrument response ({len(resp_f_interp)} pts)')
            fig4.savefig(f'{image_directory}additional_images/{day}/{filename}_interpolated_resp.png')
            # plt.savefig(f'{image_directory}additional_images/{day}/{filename}_interpolated_resp.eps')

        # There is one more step before we remove the instrument response: Running a bandpass filter on the area of the
        # frequency domain area that we care about.
        b, a = butter_bandpass(lowcut, highcut, 1 / delta_target, bp_order)
        geophone_data_bp = lfilter(b, a, geophone_data)
        fft_full_bp = fftpack.fft(geophone_data_bp)

        # Check the effect of the bandpass filter on the data
        if ad_plots:
            fig5 = plt.figure(num=5, clear=True)
            ax0 = plt.subplot(2, 2, 1)
            ax0.plot(geophone_data, color='red')
            ax0.set_xlim((np.arange(len(geophone_data))[0], np.arange(len(geophone_data))[-1]))
            ax1 = plt.subplot(2, 2, 2)
            ax1.plot(fft_f[0:len(fft_f)//2], abs(fft_full[0:len(fft_f)//2]), color='red')
            ax1.set_xlim((fft_f[0:len(fft_f)//2][0], fft_f[0:len(fft_f)//2][-1]))
            ax1.axvline(x=lowcut, color='black')
            ax1.axvline(x=highcut, color='black')
            ax2 = plt.subplot(2, 2, 3)
            ax2.plot(geophone_data_bp, color='blue')
            ax2.set_xlim((np.arange(len(geophone_data_bp))[0], np.arange(len(geophone_data_bp))[-1]))
            ax3 = plt.subplot(2, 2, 4)
            ax3.plot(fft_f[0:len(fft_f) // 2], abs(fft_full_bp[0:len(fft_f) // 2]), color='blue')
            ax3.set_xlim((fft_f[0:len(fft_f) // 2][0], fft_f[0:len(fft_f) // 2][-1]))
            ax3.axvline(x=lowcut, color='black')
            ax3.axvline(x=highcut, color='black')
            fig5.suptitle(f'{filename} Geo{geophone_ind + 1} {lowcut}-{highcut} Hz Bandpass', fontweight='bold')
            fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig5.savefig(f'{image_directory}additional_images/{day}/{filename}_geo{geophone_ind + 1}_filtered.png')
            # plt.savefig(f'{image_directory}additional_images/{day}/{filename}_geo{geophone_ind + 1}_filtered.eps')

        # -----------------------------------------------
        # Apply the "water level" to the frequency response.
        # Find the Fourier domain number at a particular threshold frequency, then apply it to every frequency less than
        # that. The main purpose of this technique is to prevent amplitude values at low frequencies from skyrocketing
        # during the deconvolution.
        # Check out this reference for more information
        # https://docs.obspy.org/tutorial/code_snippets/seismometer_correction_simulation.html

        # Apply the water-lvl correction
        # Note: The image only shows the positive side, but the same correction was applied to the negative side
        fig6 = plt.figure(num=6, clear=True)
        ax0 = plt.subplot(1, 1, 1)
        ax0.loglog(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_interp[0:len(resp_f_interp) // 2]),
                   color='red')

        water_lvl_value_ind_positive = find_nearest(resp_f_interp, wl_cutoff)
        water_lvl_value_ind_negative = find_nearest(resp_f_interp, np.negative(wl_cutoff))
        water_lvl_value_positive = resp_vals_interp[water_lvl_value_ind_positive]
        water_lvl_value_negative = resp_vals_interp[water_lvl_value_ind_negative]

        # Overwrite the positive and negative frequency values
        resp_vals_interp[0:water_lvl_value_ind_positive] = water_lvl_value_positive
        resp_vals_interp[water_lvl_value_ind_negative:-1] = water_lvl_value_negative

        ax0.loglog(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_interp[0:len(resp_f_interp) // 2]),
                   color='blue')
        ax0.legend(['Original Response (abs)', 'Water-level Corrected Response (abs)'])
        if not os.path.exists(f'{image_directory}images/0_p1_wl_correction.png'):
            fig6.savefig(f'{image_directory}images/p1_wl_correction.png')
            # plt.savefig(f'{image_directory}images/p1_wl_correction.eps')

        # -----------------------------------------------
        # Plot the real, imaginary, and absolute parts of the insturment response
        if not os.path.exists(f'{image_directory}images/p2_full_instrument_response.png'):
            fig7 = plt.figure(figsize=(6, 8), num=7, clear=True)
            ax0 = plt.subplot(3, 1, 1)
            ax0.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals_interp[0:len(resp_f_interp) // 2].real,
                     color='red')
            ax0.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax0.set_title('Real Part (Amplitude)')
            ax1 = plt.subplot(3, 1, 2)
            ax1.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals_interp[0:len(resp_f_interp) // 2].imag,
                     color='green')
            ax1.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax1.set_title('Imaginary Part (Phase)')
            ax2 = plt.subplot(3, 1, 3)
            ax2.plot(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_interp[0:len(resp_f_interp) // 2]),
                     color='blue')
            ax2.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax2.set_title('Absolute (Power)')
            fig7.suptitle('Single-Value Instrument Response', fontweight='bold')
            fig7.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig7.savefig(f'{image_directory}images/full_instrument_response.png')
            # plt.savefig(f'{image_directory}images/full_instrument_response.eps')

        # Invert the instrument response
        resp_vals_inverted = 1.0 / resp_vals_interp

        # Compute the phase of both the regular response and the inverted response
        resp_phase = (180 / math.pi) * np.angle(resp_vals_interp)
        resp_phase_inverted = (180 / math.pi) * np.angle(resp_vals_inverted)

        # Plot to check the inversion
        if not os.path.exists(f'{image_directory}images/p3_instrument_response_inverted_log.png'):
            fig8 = plt.figure(figsize=(10, 4), num=8, clear=True)
            ax = plt.subplot(1, 1, 1)
            ax.loglog(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_interp[0:len(resp_vals_interp) // 2]),
                      color='red')
            ax.set_ylabel('Absolute Response', color='tab:red')
            axb = ax.twinx()
            axb.loglog(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_inverted[0:len(resp_f_interp) // 2]),
                       color='blue')
            axb.set_ylabel('Absolute Inverted Response', color='tab:blue')
            fig8.suptitle('Regular and Inverted Response', fontweight='bold')
            fig8.savefig(f'{image_directory}images/p3_instrument_response_inverted_log.png')
            # plt.savefig(f'{image_directory}images/p3_instrument_response_inverted_log.eps')

        if not os.path.exists(f'{image_directory}images/p4_instrument_response_inverted_linear.png'):
            fig9 = plt.figure(figsize=(6, 8), num=9, clear=True)
            ax0 = plt.subplot(3, 1, 1)
            ax0.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals_inverted[0:len(resp_f_interp) // 2].real,
                     color='red')
            ax0.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax0.set_title('Real Part (Amplitude)')
            ax1 = plt.subplot(3, 1, 2)
            ax1.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals_inverted[0:len(resp_f_interp) // 2].imag,
                     color='green')
            ax1.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax1.set_title('Imaginary Part (Phase)')
            ax2 = plt.subplot(3, 1, 3)
            ax2.plot(resp_f_interp[0:len(resp_f_interp) // 2], abs(resp_vals_inverted[0:len(resp_f_interp) // 2]),
                     color='blue')
            ax2.set_xlim((resp_f_interp[0:len(resp_f) // 2][0], resp_f_interp[0:len(resp_f) // 2][-1]))
            ax2.set_title('Absolute (Power)')
            fig9.suptitle('Inverted Response', fontweight='bold')
            fig9.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig9.savefig(f'{image_directory}images/instrument_response_inverted_linear.png')
            # plt.savefig(f'{image_directory}images/instrument_response_inverted_linear.eps')

        # Remove the instrument response by multiplying the FFT of the geophone data with the inverted response
        resp_removed = np.multiply(fft_full_bp, resp_vals_inverted)

        # Get the new time series by doing an inverse Fourier transform
        ts_ifft = fftpack.ifft(resp_removed)

        # Detrend the data to remove lower frequencies
        # ts_ifft_detrended = detrend(ts_ifft)
        ts_ifft_detrended = polynomial(ts_ifft, order=6)
        tme = np.arange(len(ts_ifft_detrended))*(1/sr_target)
        if ad_plots:
            fig10 = plt.figure(num=10, clear=True)
            ax0 = plt.subplot(2, 1, 1)
            ax0.plot(tme, ts_ifft, color='r')
            ax0.set_xlim((tme[0], tme[-1]))
            ax0.set_title(f'{filename} Detrended Data')
            ax1 = plt.subplot(2, 1, 2)
            ax1.plot(tme, ts_ifft_detrended)
            ax1.set_xlim((tme[0], tme[-1]))
            fig10.savefig(f'{image_directory}images/{day}/{filename}_geo{geophone_ind + 1}_detrended_data.png')
            # plt.savefig(f'{image_directory}images/{day}/{filename}_geo{geophone_ind + 1}_detrended_data.eps')
            # plt.close()

        # Change the units from m/s to nm/second
        ts_ifft = ts_ifft_detrended
        ts_ifft = np.multiply(ts_ifft, 10 ** 9)

        # Plot the final result
        fig11 = plt.figure(figsize=(12, 8), num=11, clear=True)
        ax9 = plt.subplot(3, 2, 1)
        ax9.plot(geophone_time_seconds, geophone_data, 'k')
        ax9.set_title('Original time series', fontweight='bold')
        ax9.set_xlim((np.min(geophone_time_seconds), np.max(geophone_time_seconds)))
        ax10 = plt.subplot(3, 2, 2)
        ax10.plot(fft_f[0:len(fft_f) // 2], abs(fft_full[0:len(fft_full) // 2]), 'k')
        ax10.set_xlim((1, np.max(fft_f)))
        # plt.xscale('log')
        ax10.set_title('Original FFT', fontweight='bold')

        ax0 = plt.subplot(3, 2, 3)
        ax0.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_phase[0:len(resp_phase) // 2], color='green')
        # ax0.set_xscale('log')
        ax0.set_ylabel('Phase', color='tab:green')
        ax0b = ax0.twinx()
        ax0b.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_phase_inverted[0:len(resp_phase) // 2], color='red')
        # ax0b.set_xscale('log')
        ax0b.set_ylabel('Inverted phase', color='tab:red')
        ax0b.set_xlim((1, np.max(fft_f)))
        ax0b.set_title('Response Phase', fontweight='bold')

        ax1 = plt.subplot(3, 2, 4)
        # ax1.loglog(resp_f[0:len(resp_f) // 2], abs(new_resp_vals[0:len(resp_vals) // 2]), color='green')
        ax1.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals[0:len(resp_vals_interp) // 2],
                 color='green')
        # ax1.set_xscale('log')
        # ax1.set_yscale('log')
        ax1.set_ylabel('Inst. Response', color='tab:green')
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Inverted Response', color=color)
        # ax2.loglog(resp_f[0:len(resp_f) // 2], abs(resp_vals_inverted[0:len(resp_vals_inverted) // 2]), color='red')
        ax2.plot(resp_f_interp[0:len(resp_f_interp) // 2], resp_vals_inverted[0:len(resp_vals_inverted) // 2],
                 color='red')
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.set_xlim((1, np.max(fft_f)))
        # plt.axhline(y=abs(resp_inverted_min), color='black')
        ax2.set_title('Response Amplitude', fontweight='bold')

        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(geophone_time_seconds, ts_ifft, color='blue')
        ax5.set_xlim((np.min(geophone_time_seconds), np.max(geophone_time_seconds)))
        ax5.set_title('New Time Series', fontweight='bold')
        ax5.set_ylabel('Velocity (nm/s)')

        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(resp_f_interp[0:len(resp_vals_inverted) // 2], abs(resp_removed[0:len(resp_vals_inverted) // 2]),
                 color='blue')
        ax6.axvline(x=lowcut, color='black')
        ax6.axvline(x=highcut, color='black')
        # plt.xscale('log')
        ax6.set_xlim((1, np.max(fft_f)))
        ax6.set_title(f'New FFT ({lowcut}-{highcut} Hz Filter Order {bp_order})', fontweight='bold')

        fig11.suptitle(f'{filename} Geo{geophone_ind + 1} Response Removed', fontweight='bold')
        fig11.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig11.savefig(f'{image_directory}images/{day}/{filename}_geo{geophone_ind + 1}_resp_rm.png')
        # fig11.savefig(f'{image_directory}images/{day}/{filename}_geo{geophone_ind + 1}_resp_rm.eps')

        # Save the data in sac format
        date = dt.datetime.strptime(filename.split('-')[0], "%Y%m%d")
        starttime = date + dt.timedelta(seconds=(int(filename.split('-')[1]) * 3600))
        tt = starttime.timetuple()
        jday = tt.tm_yday
        header = {'kstnm': 'ST17', 'kcmpnm': f'geo{geophone_ind + 1}', 'nzyear': starttime.year,
                  'nzjday': jday, 'nzhour': starttime.hour, 'nzmin': starttime.minute,
                  'nzsec': starttime.second, 'nzmsec': starttime.microsecond, 'delta': delta_target}

        sac = SACTrace(data=ts_ifft, **header)
        fname_base = filename.split('-')[0]
        fname_base_hour = filename.split('-')[1]
        filename2 = f'{fname_base}_17Geo{geophone_ind + 1}_{fname_base_hour}_ID'
        sac.write(f'{output_folder}sac_data/{day}/{filename2}')

    return


def process_data(input_folder, unique_dayhour, ad_plots, output_folder, resp_f, resp_vals, lowcut, highcut, bp_order,
                 wl_cutoff, image_directory):
    """
    The original workflow to get to digital units is:

    Physical units -> analog voltage -> compressed voltage (by compressor) -> digitized units (by DAS)

    These steps are:
    p_out -> v_in -> v_out -> d_out

    We will be working with the uncompressed (analog) voltage, so we only need to run the last step.

    :param input_folder: [str] Path to input folder of sac data
    :param unique_dayhour: [touple] Unique day-hour touple to get the representative files
    :param ad_plots: [boolean] Whether to save additional images of each processing step.
    :param output_folder: [str] Path to output folder
    :param resp_f: [vector] Response function frequency values
    :param resp_vals: [vector] Response function amplitude values
    :param lowcut: [float] Lower corner value of bandpass filter (Hz)
    :param highcut: [float] Upper corner value of bandpass filter (Hz)
    :param bp_order: [int] Bandpass filter order
    :param wl_cutoff: [float] Water-level cutoff (Hz)
    :param image_directory: [str] Path to save image directory
    :return:
    """
    time_start = time.time()

    # Get the filelist corresponding to the unique dayhour
    flist = glob.glob(f'{input_folder}*/{unique_dayhour[0]}*_{unique_dayhour[1]}_ID')

    # Check that it contains 4 files (the 4 geophones)
    if not len(flist) == 4:
        print(f'Error with number of files for dayhour {unique_dayhour[0]}-{unique_dayhour[1]}! Skipping...')
        return

    # Get the day and filename of the input
    day = unique_dayhour[0]
    hour = unique_dayhour[1]
    filename = f'{day}-{hour}'

    # Check whether or not the data has been processed. If so, skip this.
    sta_check = glob.glob(f'{output_folder}sac_data/{day}/{day}_17G*_{hour}_ID')
    if len(sta_check) == 4:
        print(f'{unique_dayhour[0]}-{unique_dayhour[1]} already processed! Skipping...')
        return

    # Read the data from ascii and turn into a pandas dataframe
    df = read_sac(flist, ad_plots, day, filename, image_directory)

    # Remove instrument response
    rm_freq_resp(df, ad_plots, day, filename, output_folder, resp_f, resp_vals, lowcut, highcut, bp_order, wl_cutoff,
                 image_directory)

    time_end = time.time()
    time_diff = time_end - time_start
    print(f'Finished processing {filename} ({np.round(time_diff, 1)} sec)...')

    return


def get_unique_data(input_folder):
    """
    Finds the unique data within the input folder for processing
    :param input_folder:
    :return:
    """
    # Get a list of all the data for one instrument
    oneinst_data_vector = sorted(glob.glob(f'{input_folder}*/*_17Geo1_*ID'))

    # Extract the unique day-hour combination for that data as a touple
    unique_dayhour_vector = []
    for oneinst_data in oneinst_data_vector:
        bname = os.path.basename(oneinst_data)
        dayval = bname.split('_')[0]
        hrval = bname.split('_')[2]
        unique_dayhour_vector.append((dayval, hrval))

    return unique_dayhour_vector


def main():
    # input_folder is where the data ASCII data is stored
    data_dir = 'C:/data/lunar_data/'
    out_dir = 'C:/data/lunar_output/'
    image_directory = f'{out_dir}phys_convert/'
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)
    input_folder = f'{data_dir}LSPE_sac_hourly/'
    rundir = 'C:/Users/fcivi/Dropbox/NASA_codes/thermal_loc_final4/'

    # We will save the data to SAC format and save a picture of the processing result.
    output_folder = f'{data_dir}LSPE_sac_hourly_phys/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(f'{output_folder}sac_data/'):
        os.mkdir(f'{output_folder}sac_data/')
    if not os.path.exists(f'{image_directory}images/'):
        os.mkdir(f'{image_directory}images/')

    # Load the Apollo 17 response curves from Nunn et al. 2020 (https://doi.org/10.1007/s11214-020-00709-3)
    resp_file = f'{rundir}resp.pkl'
    with open(resp_file, 'rb') as f:
        resp_f, resp_vals = pickle.load(f)

    # Choose the number of cpu cores. If number is more than 1, it will be parallelized.
    num_cores = 4

    # Parameters of the bandpass filter used in the deconvolution step
    # Includes the lower and higher corner frequencies (in Hz) and filter order
    lowcut = 5.0
    highcut = 35.0
    bp_order = 7

    # Define the water-level cutoff (in Hz) of the deconvolution process
    wl_cutoff = 1.0

    # We need to create daily folders for plotting the results of the instrument removed data.
    # The folders for each day need to be created now or the parallelization will crash.
    daylist = glob.glob(f'{input_folder}1*')
    for dayfolder in daylist:
        daystr = os.path.basename(dayfolder)
        if not os.path.exists(f'{image_directory}images/{daystr}'):
            os.mkdir(f'{image_directory}images/{daystr}')
        if not os.path.exists(f'{output_folder}sac_data/{daystr}'):
            os.mkdir(f'{output_folder}sac_data/{daystr}')

    # There is also the ability to save plots showing each step of processing. If not interested, change to False.
    additional_plots = True
    # additional_plots = False
    if additional_plots:
        if not os.path.exists(f'{image_directory}additional_images/'):
            os.mkdir(f'{image_directory}additional_images/')
        for dayfolder in daylist:
            daystr = os.path.basename(dayfolder)
            if not os.path.exists(f'{image_directory}additional_images/{daystr}'):
                os.mkdir(f'{image_directory}additional_images/{daystr}')

    # Get a list of the unique day/hour combinations for processing
    unique_dayhour_vector = get_unique_data(input_folder)

    # If you only want to run this on a particular day-hour combination, uncomment and change below
    # unique_dayhour_vector = [('19760817', '16')]

    # Process data
    if num_cores == 1:
        # Single-core version
        for unique_dayhour in unique_dayhour_vector:
            process_data(input_folder, unique_dayhour, additional_plots, output_folder, resp_f, resp_vals, lowcut,
                         highcut, bp_order, wl_cutoff, image_directory)
    else:
        # Parallel version
        Parallel(n_jobs=num_cores)(delayed(process_data)(input_folder, unique_dayhour, additional_plots,
                                                         output_folder, resp_f, resp_vals, lowcut,
                                                         highcut, bp_order, wl_cutoff, image_directory)
                                   for unique_dayhour in unique_dayhour_vector)

main()
