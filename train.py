import os
from functools import partial

import torch
from datasets import load_from_disk

from data.preprocessing import create_data, featurize, find_file_paths, compute_class_weights
from training.metrics import compute_metrics
from training.training import get_trainer
from model.model import BirdClassifier, load_model
from config.settings import Config, DataConfig, EvalConfig, ModelConfig


def train():
    # load config
    config = Config(
        data_config = DataConfig(),
        eval_config = EvalConfig(),
        model_config = ModelConfig(),
        verbose=True,
    )

    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("MPS is available:", torch.backends.mps.is_available())

    # check if the dataset exists or the user is not forcing
    # a recreation of a dataset. if yes, load directly from disk
    if os.path.exists(config.data_config.hf_dataset_dir) and not config.data_config.force_create_dataset:
        # get label mappings
        _, _, label2id, id2label = find_file_paths(config.data_config.dataset_dir, config)
        # load dataset from disk
        featurized_dataset = load_from_disk(config.data_config.hf_dataset_dir)
    else:
        # create dataset
        dataset, label2id, id2label = create_data(config, test_size=0.2)
        # featurize data
        featurized_dataset = featurize(dataset, config, device)

    if config.verbose:
        print(f"Training data has {featurized_dataset["train"].num_rows} rows")
        print(f"Test data has {featurized_dataset["test"].num_rows} rows")
        print(f"There are {len(label2id)} unique labels in the dataset")
    
    # get class weights
    class_weights = compute_class_weights(featurized_dataset["train"]["label"])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights.to(device)
    
    # load model
    # model = BirdClassifier(num_labels=len(label2id), config=config, device=device, class_weights=class_weights)
    # model.to(device)
    model = load_model(len(label2id), label2id, id2label, config, device)

    # Create a partial function that includes label2id parameter
    partial_compute_metrics = partial(
        compute_metrics,
        class_names=list(id2label.values()),
    )

    # get trainer
    trainer = get_trainer(model, featurized_dataset, partial_compute_metrics, config)

    # train model
    trainer.train()

if __name__ == "__main__":
    train()