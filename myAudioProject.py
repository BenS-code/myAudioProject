#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib, NumPy, and SciPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.fft import fft


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
    '-w', '--window', type=float, default=2000, metavar='DURATION',
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
    global plotdata, fftplotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
        freq_data = fft(data[:, 0])  # Take FFT of the first channel
        fftplotdata = 20 * np.log10(np.abs(freq_data) + 1)

    # Generate frequency axis values
    freq_axis = np.fft.fftfreq(len(fftplotdata), d=1 / args.samplerate)

    # Plot the FFT with correct frequencies on the x-axis
    fft_line.set_ydata(fftplotdata)
    fft_line.set_xdata(freq_axis)
    fft_line.set_marker('o')
    fft_line.set_linestyle("")
    fft_line.set_color('blue')
    fft_line.set_markersize(6)

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])

    return lines + [fft_line]


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    fftplotdata = np.zeros((length,))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    lines = ax1.plot(plotdata)
    fft_line, = ax2.plot(fftplotdata)
    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
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
    ax2.set_ylim(-0.1, 20)  # Adjust the limits as necessary
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
