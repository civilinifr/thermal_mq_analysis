"""
Pick the start of the trace for each geophone of the LSPE for a random sampling of impulsive events.
Only a single pick is done for this particular use, but they can be expanded to any arbitrary number of picks.
"""
# Import packages
import glob
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass, lowpass
from matplotlib import cm
from scipy.signal import butter, lfilter


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


def run_highpass_filter(input_data, input_corner_freq, sr):
    """
    Passes a highpass filter on the data

    :param input_data: [obspy trace] Seismic input trace in Obspy format
    :param input_corner_freq: [int] Corner frequency of highpass filter
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


def run_lowpass_filter(input_data, input_corner_freq, sr, corners):
    """
    Passes a lowpass filter on the data

    :param input_data: [obspy trace] Seismic input trace in Obspy format
    :param input_corner_freq: [int] Corner frequency of highpass filter
    :return:
    """
    # Remove the mean and run a cosine taper
    trace_nomean = input_data.data - np.mean(input_data.data)
    N = len(trace_nomean)
    taper_function = cosine_taper(N, p=0.1)
    trace_taper = trace_nomean * taper_function

    # Highpass filterbound in Hertz
    data_filt = lowpass(trace_taper, input_corner_freq, sr, corners=corners, zerophase=False)

    return data_filt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def onclick(event):
    global ix, iy, evid, geophone
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy})')

    # assign global variable to access outside of function
    global coords, comp_ind
    coords.append((ix, iy))

    ax.axvline(x=coords[0][0], c='r')
    # ax2.axvline(x=coords[0][0], c='r')

    # Disconnect after 2 clicks
    if len(coords) == 1:
        plt.savefig(f'{outdir}images/evid_{evid}_{geophone}_picks.png')
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return


# Main
indir = 'C:/data/lunar_output/'
input_directory = f'{indir}original_images/combined_files/'
output_directory = f'{indir}picking/'
outdir = f'{output_directory}evt_pick_results/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(f'{outdir}images/'):
    os.mkdir(f'{outdir}images/')
if not os.path.exists(f'{outdir}data/'):
    os.mkdir(f'{outdir}data/')

filelist = glob.glob(f'{input_directory}*.pkl')

# Take a random sampling of the events
num_events = 200
good_evids_needed = 40

# List here the emergent events as we find them in the random sampling.
# We can't use them because it's too difficult to accurately pick the arrival time
emr_evts = np.array(['761011-04-M2', '760910-10-M2', '761109-11-M3', '770401-21-M2', '760911-19-M4', '770406-00-M2',
                     '770309-23-M1', '770406-00-M2', '770112-05-M3', '761110-00-M2', '761105-04-M1', '770331-13-M1',
                     '770306-23-M3', '770209-07-M2', '760910-06-M2', '770312-09-M1', '770311-07-M1', '770301-18-M1',
                     '760906-02-M1', '770206-17-M3', '770227-14-M1', '761210-01-M3', '770310-23-M3', '760816-02-M1',
                     '770414-09-M1', '761107-11-M3', '760904-13-M2', '761012-05-M2', '760306-17-M1', '770306-17-M1',
                     '770405-15-M2', '761006-20-M2', '770224-15-M2', '770409-08-M1', '770125-21-M3', '761128-01-M1',
                     '770114-22-M1', '760914-00-M2', '761109-09-M1', '770326-05-M2', '770110-01-M1', '761128-21-M1',
                     '761024-00-M1', '770208-16-M4', '760816-14-M1', '761211-06-M4', '761226-18-M4', '770108-01-M1',
                     '770409-03-M1', '761111-12-M5', '760910-03-M2', '770224-08-M3', '761004-04-M1', '761127-08-M2',
                     '760905-10-M1', '761129-21-M1', '770310-02-M3', '761108-05-M3', '770102-15-M1', '761113-00-M1',
                     '761009-01-M1', '761203-23-M1', '770106-16-M1', '770405-04-M1', '760915-06-M1', '770107-00-M1',
                     '760915-14-M3', '770207-02-M2', '761009-23-M3', '770204-04-M1', '761128-03-M2', '761229-23-M1',
                     '770424-17-M1', '761109-13-M1', '761107-16-M3'])

# Enable or disable the random seed for a consistent random list
np.random.seed(1)
evts = np.random.choice(filelist, num_events)

good_evids = 0
for evt in evts:
    if good_evids > good_evids_needed:
        print('Finished picking all needed events! Exiting...')
        break

    # Load the pickle file with the data
    with open(evt, 'rb') as f:
        time_array_cut, data_array_cut, abs_trace_start, abs_trace_end, \
        rel_det_vector, input_info, data_geophone_list = pickle.load(f)
    evid = input_info.evid

    if len(np.where(emr_evts == evid)[0]) > 0:
        print(f'Evid {evid} is bad. Skipping...')
        continue

    # If the output file for this event exists already, ignore it.
    if os.path.exists(f'{outdir}data/evid_{evid}_picks.txt'):
        print(f'evid {evid} already processed! Skipping...')
        continue

    # We will take x and y coordinates in separate columns for two points.
    # With the four geophones, it means that the array will be 4x4.
    evt_coords = np.zeros((4, 2))

    # Cycle through each geophone and pick two points (event start and end of the envelope).
    for comp_ind in np.arange(4):
        # Get the name of the geophone from comp_ind
        geophone = f'geo{comp_ind + 1}'

        # Setup the array coords corresponding to the points picked for this particular trace
        coords = []
        x = time_array_cut[:, comp_ind]
        y = data_array_cut[:, comp_ind]

        #  Compute the spectrogram
        sr = 1 / (x[1] - x[0])

        # Pass different filters as needed for the events (just uncomment which one you want to choose).
        # Butter bandpass
        # order = 9
        # b, a = butter_bandpass(10., 15., sr, order=order)
        # trace_filtered = lfilter(b, a, y)

        # Highpass
        # trace_filtered = run_highpass_filter(y, 10.0, sr)

        # Lowpass
        trace_filtered1 = run_lowpass_filter(y, 15, sr, 8)
        b, a = butter_bandpass(5., 10., sr, order=8)
        trace_filtered2 = lfilter(b, a, y)

        # Unfiltered
        # trace_filtered = y

        # Plot the spectrogram
        f, t, Sxx = signal.spectrogram(trace_filtered1, sr, nperseg=8)
        t = t + x[0]

        # Original approxiate arrival
        original_arrival = rel_det_vector[comp_ind]

        # Plot the result and pick
        fig = plt.figure(1)
        ax = fig.add_subplot(311)
        ax.plot(x, trace_filtered1)
        plt.xlim((original_arrival - 20, original_arrival + 20))
        plt.title(f'Evid: {evid} / Station: {geophone}')

        ax1 = fig.add_subplot(312)
        ax1.plot(x, trace_filtered2)
        plt.xlim((original_arrival - 20, original_arrival + 20))


        ax4 = fig.add_subplot(313)
        specmax = 1e-7
        ax4.pcolormesh(t, f, Sxx, cmap=cm.jet, vmax=specmax, shading='auto')
        # ax2.pcolormesh(t, f, Sxx, cmap=cm.jet, shading='auto')
        plt.xlim((original_arrival - 20, original_arrival + 20))
        plt.ylim((0, 40))

        # Call click func
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # Make full screen for picking
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        plt.show()

        # Funky runtime errors happen if we keep the touples, so I split the x and y into separate columns
        evt_coords[comp_ind, 0] = coords[0][0]
        evt_coords[comp_ind, 1] = coords[0][1]

    # Save the result for each evid with each geophone.
    np.savetxt(f'{outdir}data/evid_{evid}_picks.txt', evt_coords)
    good_evids = good_evids + 1
    print(f'Saved coordinates for evid {evid}!')


