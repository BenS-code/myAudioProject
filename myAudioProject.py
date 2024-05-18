# import argparse
# import queue
# import sys
#
# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# import numpy as np
# import sounddevice as sd
# from scipy.fft import fft
# import scipy.signal as signal
#
#
# def int_or_str(text):
#     """Helper function for argument parsing."""
#     try:
#         return int(text)
#     except ValueError:
#         return text
#
#
# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument(
#     '-l', '--list-devices', action='store_true',
#     help='show list of audio devices and exit')
# args, remaining = parser.parse_known_args()
# if args.list_devices:
#     print(sd.query_devices())
#     parser.exit(0)
# parser = argparse.ArgumentParser(
#     description=__doc__,
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     parents=[parser])
# parser.add_argument(
#     'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
#     help='input channels to plot (default: the first)')
# parser.add_argument(
#     '-d', '--device', type=int_or_str,
#     help='input device (numeric ID or substring)')
# parser.add_argument(
#     '-w', '--window', type=float, default=5000, metavar='DURATION',
#     help='visible time slot (default: %(default)s ms)')
# parser.add_argument(
#     '-i', '--interval', type=float, default=1,
#     help='minimum time between plot updates (default: %(default)s ms)')
# parser.add_argument(
#     '-b', '--blocksize', type=int, help='block size (in samples)')
# parser.add_argument(
#     '-r', '--samplerate', type=float, help='sampling rate of audio device')
# parser.add_argument(
#     '-n', '--downsample', type=int, default=1, metavar='N',
#     help='display every Nth sample (default: %(default)s)')
# args = parser.parse_args(remaining)
# if any(c < 1 for c in args.channels):
#     parser.error('argument CHANNEL: must be >= 1')
# mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
# q = queue.Queue()
#
# def audio_callback(indata, frames, time, status):
#     """This is called (from a separate thread) for each audio block."""
#     if status:
#         print(status, file=sys.stderr)
#     q.put(indata[::args.downsample, mapping])
#
#
# def update_plot(frame):
#     """This is called by matplotlib for each plot update."""
#     global plotdata, fftplotdata, filtered_plotdata, filtered_fftplotdata
#     while True:
#         try:
#             data = q.get_nowait()
#         except queue.Empty:
#             break
#         shift = len(data)
#         plotdata = np.roll(plotdata, -shift, axis=0)
#         plotdata[-shift:, :] = data
#
#         # Apply Butterworth band-pass filter
#         filtered_data = signal.sosfilt(sos, data, axis=0)
#         filtered_plotdata = np.roll(filtered_plotdata, -shift, axis=0)
#         filtered_plotdata[-shift:, :] = filtered_data
#
#         # Compute FFT of the filtered signal
#         freq_data = fft(data[:, 0])  # FFT of the original signal
#         fftplotdata = 20 * np.log10(np.abs(freq_data) + 1)
#
#         filtered_freq_data = fft(filtered_data[:, 0])  # FFT of the filtered signal
#         filtered_fftplotdata = 20 * np.log10(np.abs(filtered_freq_data) + 1)
#
#     # Generate frequency axis values
#     freq_axis = np.fft.fftfreq(len(fftplotdata), d=1 / args.samplerate)
#
#     # Update the plots
#     fft_line.set_ydata(fftplotdata)
#     fft_line.set_xdata(freq_axis)
#
#     filtered_fft_line.set_ydata(filtered_fftplotdata)
#     filtered_fft_line.set_xdata(freq_axis)
#
#     for column, line in enumerate(lines):
#         line.set_ydata(plotdata[:, column])
#
#     for column, line in enumerate(filtered_lines):
#         line.set_ydata(filtered_plotdata[:, column])
#
#     return lines + [fft_line] + filtered_lines + [filtered_fft_line]
#
#
# try:
#     if args.samplerate is None:
#         device_info = sd.query_devices(args.device, 'input')
#         args.samplerate = device_info['default_samplerate']
#
#     length = int(args.window * args.samplerate / (1000 * args.downsample))
#     plotdata = np.zeros((length, len(args.channels)))
#     filtered_plotdata = np.zeros((length, len(args.channels)))
#     fftplotdata = np.zeros((length,))
#     filtered_fftplotdata = np.zeros((length,))
#
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     lines = ax1.plot(plotdata)
#     fft_line, = ax2.plot(fftplotdata)
#     filtered_lines = ax3.plot(filtered_plotdata)
#     filtered_fft_line, = ax4.plot(filtered_fftplotdata)
#
#     if len(args.channels) > 1:
#         ax1.legend([f'channel {c}' for c in args.channels],
#                    loc='lower left', ncol=len(args.channels))
#         ax3.legend([f'filtered channel {c}' for c in args.channels],
#                    loc='lower left', ncol=len(args.channels))
#
#     ax1.axis((0, len(plotdata), -1, 1))
#     ax1.set_yticks([0])
#     ax1.yaxis.grid(True)
#     ax1.tick_params(bottom=False, top=False, labelbottom=False,
#                     right=False, left=False, labelleft=False)
#     ax1.set_title('Microphone Signal')
#
#     ax2.set_title('FFT of Microphone Signal')
#     ax2.set_xlabel('Frequency')
#     ax2.set_ylabel('Magnitude (dB)')
#     ax2.set_xlim(0, args.samplerate / 2)
#     ax2.set_ylim(-0.1, 50)  # Adjust the limits as necessary
#
#     ax3.axis((0, len(filtered_plotdata), -1, 1))
#     ax3.set_yticks([0])
#     ax3.yaxis.grid(True)
#     ax3.tick_params(bottom=False, top=False, labelbottom=False,
#                     right=False, left=False, labelleft=False)
#     ax3.set_title('Filtered Signal')
#
#     ax4.set_title('FFT of Filtered Signal')
#     ax4.set_xlabel('Frequency')
#     ax4.set_ylabel('Magnitude (dB)')
#     ax4.set_xlim(0, args.samplerate / 2)
#     ax4.set_ylim(-0.1, 50)  # Adjust the limits as necessary
#
#     fig.tight_layout(pad=0.5)
#
#     # Butterworth band-pass filter design parameters
#     lowcut = 7500.0
#     highcut = 8500.0
#     order = 4
#
#     # Design Butterworth band-pass filter
#     nyquist = 0.5 * args.samplerate
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
#
#     stream = sd.InputStream(
#         device=args.device, blocksize=2048, channels=max(args.channels),
#         samplerate=args.samplerate, callback=audio_callback)
#     ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True, cache_frame_data=False)
#     with stream:
#         plt.show()
# except Exception as e:
#     parser.exit(type(e).__name__ + ': ' + str(e))



import argparse
import queue
import sys
import os

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.fft import fft
import scipy.signal as signal
from scipy.io.wavfile import write


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=5000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=1,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update."""
    global plotdata, fftplotdata, filtered_plotdata, filtered_fftplotdata, original_signal, filtered_signal
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

        # Append the data to the signal arrays
        original_signal = np.append(original_signal, data, axis=0)

        # Apply Butterworth band-pass filter
        filtered_data = signal.sosfilt(sos, data, axis=0)
        filtered_plotdata = np.roll(filtered_plotdata, -shift, axis=0)
        filtered_plotdata[-shift:, :] = filtered_data

        # Append the filtered data to the signal array
        filtered_signal = np.append(filtered_signal, filtered_data, axis=0)

        # Compute FFT of the filtered signal
        freq_data = fft(data[:, 0])  # FFT of the original signal
        fftplotdata = 20 * np.log10(np.abs(freq_data) + 1)

        filtered_freq_data = fft(filtered_data[:, 0])  # FFT of the filtered signal
        filtered_fftplotdata = 20 * np.log10(np.abs(filtered_freq_data) + 1)

    # Generate frequency axis values
    freq_axis = np.fft.fftfreq(len(fftplotdata), d=1 / args.samplerate)

    # Update the plots
    fft_line.set_ydata(fftplotdata)
    fft_line.set_xdata(freq_axis)

    filtered_fft_line.set_ydata(filtered_fftplotdata)
    filtered_fft_line.set_xdata(freq_axis)

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])

    for column, line in enumerate(filtered_lines):
        line.set_ydata(filtered_plotdata[:, column])

    return lines + [fft_line] + filtered_lines + [filtered_fft_line]


def on_key(event):
    if event.key == 'q':
        save_audio()
        plt.close()


def save_audio():
    if not os.path.exists('audiofiles'):
        os.makedirs('audiofiles')
    write('audiofiles/original_signal.wav', int(args.samplerate), original_signal)
    write('audiofiles/filtered_signal.wav', int(args.samplerate), filtered_signal)
    print('Audio files saved.')


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    filtered_plotdata = np.zeros((length, len(args.channels)))
    fftplotdata = np.zeros((length,))
    filtered_fftplotdata = np.zeros((length,))

    original_signal = np.empty((0, len(args.channels)))
    filtered_signal = np.empty((0, len(args.channels)))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    lines = ax1.plot(plotdata)
    fft_line, = ax2.plot(fftplotdata)
    filtered_lines = ax3.plot(filtered_plotdata)
    filtered_fft_line, = ax4.plot(filtered_fftplotdata)

    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                   loc='lower left', ncol=len(args.channels))
        ax3.legend([f'filtered channel {c}' for c in args.channels],
                   loc='lower left', ncol=len(args.channels))

    ax1.axis((0, len(plotdata), -1, 1))
    ax1.set_yticks([0])
    ax1.yaxis.grid(True)
    ax1.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
    ax1.set_title('Microphone Signal')

    ax2.set_title('FFT of Microphone Signal')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_xlim(0, args.samplerate / 2)
    ax2.set_ylim(-0.1, 50)  # Adjust the limits as necessary

    ax3.axis((0, len(filtered_plotdata), -1, 1))
    ax3.set_yticks([0])
    ax3.yaxis.grid(True)
    ax3.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
    ax3.set_title('Filtered Signal')

    ax4.set_title('FFT of Filtered Signal')
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_xlim(0, args.samplerate / 2)
    ax4.set_ylim(-0.1, 50)  # Adjust the limits as necessary

    fig.tight_layout(pad=0.5)

    # Butterworth band-pass filter design parameters
    lowcut = 7500.0
    highcut = 8500.0
    order = 4

    # Design Butterworth band-pass filter
    nyquist = 0.5 * args.samplerate
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')

    stream = sd.InputStream(
        device=args.device, blocksize=2048, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True, cache_frame_data=False)
    fig.canvas.mpl_connect('key_press_event', on_key)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
