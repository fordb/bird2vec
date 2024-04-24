from dataclasses import dataclass
import torch

@dataclass
class Config:
    output_dir: str = "bird2vec"
    dataset_dir: str = "datasets/birdsv1/"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 1e-5 # 1e-5 to 5e-5
    lr_scheduler_type: str = "cosine" # linear or cosine
    max_grad_norm: float = 0.5 # 0.5 to 1.0
    weight_decay: float = 0.01 # 0.01
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 10 # 3 to 10
    fp16: bool = not torch.backends.mps.is_available()
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    push_to_hub: bool = False
    report_to: str = "wandb"