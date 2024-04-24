from functools import partial

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor


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