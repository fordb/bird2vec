import os

import torch
from datasets import load_from_disk
from transformers import AutoFeatureExtractor

from data.preprocessing import load_data, featurize, create_label_id_mapping
from training.metrics import compute_metrics
from training.training import get_trainer
from model.model import load_model
from config.settings import Config


def train():
    # load config
    config = Config()

    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("MPS is available:", torch.backends.mps.is_available())

    # check if the dataset exists or the user is not forcing
    # a recreation of a dataset. if yes, load directly from disk
    if os.path.exists(config.hf_dataset_dir) and not config.force_create_dataset:
        label2id, id2label = create_label_id_mapping(config.dataset_dir)
        featurized_dataset = load_from_disk(config.hf_dataset_dir)
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    else:
        # load dataset and get labels
        dataset, label2id, id2label = load_data(config, test_size=0.2)
        # featurize data
        featurized_dataset, feature_extractor = featurize(dataset, config, device)

    if config.verbose:
        print(f"Training data has {featurized_dataset["train"].num_rows} rows")
        print(f"Test data has {featurized_dataset["test"].num_rows} rows")
        print(f"There are {len(label2id)} unique labels in the dataset")
    
    # load model
    model = load_model(label2id, id2label, device)

    # get trainer
    trainer = get_trainer(model, featurized_dataset, feature_extractor, compute_metrics, config)

    # train model
    trainer.train()


if __name__ == "__main__":
    train()