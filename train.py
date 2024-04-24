import torch

from data.preprocessing import load_data, featurize
from training.metrics import compute_metrics
from training.training import get_trainer
from model.model import load_model
from config.settings import Config


def train():
    # load config
    config = Config(force_create_dataset=True)

    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("MPS is available:", torch.backends.mps.is_available())

    # load dataset and get labels
    dataset, label2id, id2label = load_data(config, test_size=0.2)

    if config.verbose:
        print(f"Training data has {dataset["train"].num_rows} rows")
        print(f"Test data has {dataset["test"].num_rows} rows")
        print(f"There are {len(label2id)} unique labels in the dataset")

    # featurize data
    featurized_dataset, feature_extractor = featurize(dataset, device)

    # TODO: move save/load dataset from inside load_data to here
    
    # load model
    model = load_model(label2id, id2label, device)

    # get trainer
    trainer = get_trainer(model, featurized_dataset, feature_extractor, compute_metrics, config)

    # train model
    trainer.train()


if __name__ == "__main__":
    train()