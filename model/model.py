import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from transformers import AutoModelForAudioClassification


def load_model(num_labels, label2id, id2label, config, device):
    # load pretrained model
    model = AutoModelForAudioClassification.from_pretrained(
        config.model_config.model, num_labels=num_labels, label2id=label2id, id2label=id2label
    ).to(device)
    return model


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, feature_size, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads)
        self.feature_size = feature_size

    def forward(self, x):
        # Transpose batch and sequence dimensions
        x = x.transpose(0, 1)  # x should be (seq_length, batch_size, feature_size)
        # Apply multi-head attention
        # we use x as the query, key, and value for the self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        # Transpose back to (batch_size, seq_length, feature_size)
        attn_output = attn_output.transpose(0, 1)
        # Pooling over the sequence length dimension
        pooled_output = torch.mean(attn_output, dim=1)
        return pooled_output


class BirdClassifier(nn.Module):
    def __init__(self, num_labels, config, device, num_heads=4, class_weights=None):
        super().__init__()
        # TODO: log these parameters to wandb
        self.device = device
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_config.model).to(device)
        feature_size = self.wav2vec2.config.hidden_size
        self.attention_pool = MultiHeadAttentionPool(feature_size, num_heads)
        # self.attention_pool = AttentionPool(feature_size)
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