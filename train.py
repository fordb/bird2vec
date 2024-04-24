import os

import torch
from datasets import load_from_disk

from data.preprocessing import load_data, get_labels, featurize, map_labels
from training.metrics import compute_metrics
from training.training import get_trainer
from model.model import load_model
from config.settings import Config



def train():
    # load config
    config = Config(force_load=True)

    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("MPS is available:", torch.backends.mps.is_available())

    # if data already exists, load it
    # otherwise, create training/test dataset
    if os.path.exists(config.dataset_dir) and not config.force_load:
        print("Dataset already exists, loading from disk")
        featurized_dataset = load_from_disk(config.dataset_dir)
    else:
        print("Dataset does not exist on disk (or force_load=True), creating it now")
        # load dataset and get labels
        dataset = load_data(test_size=0.2)
        labels, label2id, id2label = get_labels(dataset)
        num_labels = len(labels)
        # featurize data
        featurized_dataset, feature_extractor = featurize(dataset, device)
        # map labels to ints
        featurized_dataset = map_labels(featurized_dataset, label2id)
        # Save the dataset
        featurized_dataset.save_to_disk(config.dataset_dir)
    
    # load model
    model = load_model(num_labels, label2id, id2label, device)

    # get trainer
    trainer = get_trainer(model, featurized_dataset, feature_extractor, compute_metrics, config)

    # train model
    trainer.train()


if __name__ == "__main__":
    train()