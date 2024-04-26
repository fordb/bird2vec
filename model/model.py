import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class AttentionPool(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_size, 1))

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=1)
        weighted_average = torch.sum(x * attention_scores, dim=1)
        return weighted_average


class BirdClassifier(nn.Module):
    def __init__(self, num_labels, config, device, class_weights=None):
        super().__init__()
        self.device = device
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        feature_size = self.wav2vec2.config.hidden_size
        # TODO: log these parameters to wandb
        self.attention_pool = AttentionPool(feature_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(config.model_config.dropout_prob),
            nn.Linear(256, num_labels)
        )
        # add a loss function directly
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_values, labels=None):
        outputs = self.wav2vec2(input_values).last_hidden_state
        pooled_output = self.attention_pool(outputs)
        logits = self.classifier(pooled_output)
        logits.to(self.device)
        # calculate loss if labels are provided
        loss = None
        if labels is not None:
            labels.to(self.device)
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}