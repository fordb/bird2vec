from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    min_label_count: int = 50
    force_create_dataset: bool = False
    dataset_dir: str = "datasets/xeno_canto_clean"
    hf_dataset_dir: str = "datasets/birdsv7"

@dataclass
class EvalConfig:
    evaluation_strategy: str = "steps"
    eval_steps: int = 0.01
    logging_strategy: str = "steps"
    logging_steps: float = 0.001
    save_strategy: str = "steps"
    save_steps: float = 0.01
    push_to_hub: bool = False
    report_to: str = "wandb"
    output_dir: str = "huggingface"

@dataclass
class ModelConfig:
    model: str = "facebook/wav2vec2-base-960h"
    learning_rate: float = 1e-5 # 1e-5 to 5e-5
    lr_scheduler_type: str = "cosine" # linear or cosine
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5 # 0.5 to 1.0
    weight_decay: float = 0.001 # 0.01 to 0.05
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 10 # 3 to 10
    fp16: bool = not torch.backends.mps.is_available()
    load_best_model_at_end: bool = True

@dataclass
class Config:
    verbose: bool = True
    data_config: DataConfig = field(default_factory=DataConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
