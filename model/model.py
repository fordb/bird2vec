from transformers import AutoModelForAudioClassification


def load_model(num_labels, label2id, id2label, device):
    # load pretrained model
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
    ).to(device)
    return model