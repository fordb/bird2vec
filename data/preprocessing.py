import os
from functools import partial

import scipy
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import librosa



def preprocess_function(examples, feature_extractor, device):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=5*16000, truncation=True, return_tensors="pt"
    )
    # Ensure tensors are moved to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def label_to_id(example, label2id):
    example["label"] = label2id[example["label"]]
    return example


def load_data(test_size=0.2, force_reload=False):
    # load data
    dataset = load_dataset("tglcourse/5s_birdcall_samples_top20")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=test_size)
    return dataset


def get_labels(dataset):
    # get labels and useful mappings
    labels = set(dataset["train"]["label"])
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return labels, label2id, id2label


def featurize(dataset, device):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) # cast to 16K audio
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    # Create a partial function that includes label2id parameter
    partial_preprocess_function = partial(
        preprocess_function,
        feature_extractor=feature_extractor,
        device=device,
    )
    featurized_dataset = dataset.map(partial_preprocess_function, batched=True, remove_columns="audio")
    return featurized_dataset, feature_extractor


def map_labels(dataset, label2id):
    # Create a partial function that includes label2id parameter
    partial_label_to_id = partial(label_to_id, label2id=label2id)
    # map train/test labels to IDs
    dataset["train"] = dataset["train"].map(partial_label_to_id)
    dataset["test"] = dataset["test"].map(partial_label_to_id)
    return dataset


def find_peaks(y, sr, FMIN=500, FMAX=12500, max_peaks=10, kernel_size=15):
    n_mels = 64
    hop_length = 512
    # adapted from: https://www.kaggle.com/code/johnowhitaker/peak-identification
    # create the mel spectrogram- 2d array representing mel frequency bands over time
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=FMIN, fmax=FMAX, n_mels=n_mels, hop_length=hop_length)
    # apply PCEN (per-channel energy normalization)- stabilizies dynamics + suppresses background noise
    pcen = librosa.core.pcen(melspec, sr=sr, gain=0.8, bias=10, power=0.25, time_constant=0.06, eps=1e-06)

    # calculate the signal to noise ratio (SNR) as the range of PCEN values over time
    pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
    # normalize ranges (by dividing by the median) and convert to decibels
    pcen_snr = librosa.power_to_db(pcen_snr / np.median(pcen_snr))

    # Apply median filter to SNR to smooth out random fluctuations and emphasize sustained variations
    median_pcen_snr = scipy.signal.medfilt(pcen_snr, kernel_size=kernel_size)

    # identify peaks
    peaks, properties = scipy.signal.find_peaks(median_pcen_snr, prominence=0.1)

    peak_heights = properties["prominences"]
    # combine peaks and their properties for sorting
    peaks_with_properties = list(zip(peaks, peak_heights))
    # sort peaks based on their prominence (or height)
    sorted_peaks = sorted(peaks_with_properties, key=lambda x: x[1], reverse=True)

    # if there are more peaks than max_peaks, keep only the max_peaks number of peaks
    if len(sorted_peaks) > max_peaks:
        sorted_peaks = sorted_peaks[:max_peaks]

    # extract the peak locations
    peak_locs = [p[0] for p in sorted_peaks]

    # convert peak indices to times
    time_per_frame = hop_length / sr
    peak_times = [loc * time_per_frame for loc in peak_locs]

    return peak_locs, peak_times


def create_audio_subsets(y, sr, window_length=5.0, max_peaks=10):
    # find audio peaks
    _, peak_times = find_peaks(y, sr, max_peaks=max_peaks)

    # calculate samples per side
    samples_per_side = int((window_length / 2) * sr)

    # Initialize a list to hold the audio subsets
    audio_subsets = []

    # Iterate over each peak time
    for peak_time in peak_times:
        # Convert peak time to the central sample index
        center_index = int(peak_time * sr)

        # Calculate the start and end indices of the subset
        start = max(0, center_index - samples_per_side)  # Ensure start is not negative
        end = min(len(y), center_index + samples_per_side)  # Ensure end does not exceed the length of y

        # Extract the subset from the audio array
        subset = y[start:end]

        # Append to the list of subsets
        audio_subsets.append(subset)
    
    return audio_subsets


def clean_and_save_audio_file(audio_path, window_length=5.0, max_peaks=10, target_sr=16000, dir="datasets/xeno_canto_clean/"):
    # load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # extract file name
    file_name = audio_path.replace("datasets/xeno_canto/", "")
    file_name = file_name.replace(".mp3", "")
    species_name = file_name.split("/")[0]

    # create audio subsets
    audio_subsets = create_audio_subsets(audio, sr, window_length=window_length, max_peaks=max_peaks)

    # adjust sample rate
    resampled_audio_subsets = [librosa.resample(a, orig_sr=sr, target_sr=target_sr) for a in audio_subsets]

    for i, resampled_audio in enumerate(resampled_audio_subsets):
        # create path
        os.makedirs(os.path.join(dir, species_name), exist_ok=True)

        path = os.path.join(dir, file_name + f"_sample_{i}.wav")
        # save processed audio files
        sf.write(path, resampled_audio, target_sr)