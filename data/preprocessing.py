import os
from functools import partial
import soundfile as sf

import scipy
import numpy as np
from datasets import Dataset
from transformers import AutoFeatureExtractor
import librosa


def preprocess_function(examples, feature_extractor, device):
    inputs = feature_extractor(
        examples["audio"],
        sampling_rate=feature_extractor.sampling_rate,
        max_length=5*16000,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    # Ensure tensors are moved to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def create_label_id_mapping(audio_dir):
    labels = set()
    # collect all species names to create mappings
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.endswith('.wav'):
                species_label = os.path.basename(root)
                labels.add(species_label)

    # create mappings from label names to integers and vice versa
    label2id = {species: idx for idx, species in enumerate(sorted(labels))}
    id2label = {idx: species for species, idx in label2id.items()}

    return label2id, id2label


def load_data(config, test_size=0.2):
    # create label mappings
    label2id, id2label = create_label_id_mapping(config.dataset_dir)

    # load and format audio data
    data = []
    for root, _, files in os.walk(config.dataset_dir):
        for filename in files:
            if filename.endswith(".wav"):
                species_label = os.path.basename(root)
                audio_path = os.path.join(root, filename)
                audio, sr = librosa.load(audio_path, sr=16000)
                data.append({"label": label2id[species_label], "audio": audio})

    # create a Hugging Face dataset
    dataset = Dataset.from_dict(
        {
            "label": [x["label"] for x in data],
            "audio": [x["audio"] for x in data]
        }
    )

    # split into train/test
    dataset = dataset.train_test_split(test_size=test_size)

    return dataset, label2id, id2label


def featurize(dataset, config, device):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    # Create a partial function that includes label2id parameter
    partial_preprocess_function = partial(
        preprocess_function,
        feature_extractor=feature_extractor,
        device=device,
    )
    
    # apply the partial function to the dataset
    featurized_dataset = dataset.map(partial_preprocess_function, batched=True, remove_columns="audio")
    # save to disk
    featurized_dataset.save_to_disk(config.hf_dataset_dir)

    return featurized_dataset, feature_extractor


def find_peaks(y, sr, FMIN=500, FMAX=12500, max_peaks=10, kernel_size=15):
    # adapted from: https://www.kaggle.com/code/johnowhitaker/peak-identification
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
    try:
        audio, sr = librosa.load(audio_path, sr=None)
    except FileNotFoundError:
        print(f"File {audio_path} not found, skipping")
        return
    
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