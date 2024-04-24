import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa

from data.preprocessing import find_peaks


def audio_waveframe(audio_data, sampling_rate, block=False):
    """
    Plot the audio waveform
    """
    # calculate the duration of the audio file
    duration = len(audio_data) / sampling_rate
    # create a time array for plotting
    time = np.arange(0, duration, 1/sampling_rate)
    # plot the waveform
    plt.figure(figsize=(30, 4))
    plt.plot(time, audio_data, color='blue')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plot = plt.show(block=block)
    return plot


def peak_plot(y, sr, FMIN=500, FMAX=12500, max_peaks=10):
    # adapted from: https://www.kaggle.com/code/johnowhitaker/peak-identification
    # PCEN spec
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr,
        fmin=FMIN, fmax=FMAX, n_mels=64)
    pcen = librosa.core.pcen(melspec, sr=sr,
        gain=0.8, bias=10, power=0.25, time_constant=0.06, eps=1e-06)
    librosa.display.specshow(pcen, sr=sr,
        fmin=FMIN, fmax=FMAX,
        x_axis='time', y_axis='mel', cmap='magma_r')
    # plt.title('PCEN-based SNR')
    plt.tight_layout()

    # SNR and a smoothed SNR with kernel 15
    plt.subplot(3, 1, 2)
    pcen_snr = np.max(pcen,axis=0) - np.min(pcen,axis=0)
    pcen_snr = librosa.power_to_db(pcen_snr / np.median(pcen_snr))
    median_pcen_snr = scipy.signal.medfilt(pcen_snr, kernel_size=15)
    times = np.linspace(0, len(y)/sr, num=melspec.shape[1])
    plt.plot(times, pcen_snr, color="orange")
    plt.plot(times, median_pcen_snr, color="blue")
    plt.xlim(times[0], times[-1])
    plt.ylim(0, 10)

    # find peaks
    peak_locs, _ = find_peaks(y, sr, FMIN=FMIN, FMAX=FMAX, max_peaks=max_peaks)

    # And go through, picking some peaks
    for t_peak in peak_locs:
        plt.scatter(times[t_peak], median_pcen_snr[t_peak], c='red', zorder=100)