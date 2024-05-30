import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoModelForAudioClassification, AutoConfig


def load_model(num_labels, class_weights, label2id, id2label, config, device):
    model_config = AutoConfig.from_pretrained(
        config.model_config.model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        class_weights=class_weights,
    )

    # Load the pretrained model
    model = AutoModelForAudioClassification.from_pretrained(
        config.model_config.model,
        config=model_config,
    ).to(device)

    if config.verbose:
        print(model.num_parameters(only_trainable=True) / 1e6)

    return model