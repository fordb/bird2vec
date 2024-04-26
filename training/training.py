from transformers import TrainingArguments, Trainer, Wav2Vec2Processor


def get_training_args(config):
    training_args = TrainingArguments(
        output_dir=config.eval_config.output_dir,
        evaluation_strategy=config.eval_config.evaluation_strategy,
        save_strategy=config.eval_config.save_strategy,
        learning_rate=config.model_config.learning_rate,
        lr_scheduler_type=config.model_config.lr_scheduler_type,
        max_grad_norm=config.model_config.max_grad_norm,
        weight_decay=config.model_config.weight_decay,
        per_device_train_batch_size=config.model_config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.model_config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.model_config.per_device_eval_batch_size,
        num_train_epochs=config.model_config.num_train_epochs,
        fp16=config.model_config.fp16,
        warmup_ratio=config.model_config.warmup_ratio,
        # warmup_steps=config.model_config.warmup_steps,
        logging_steps=config.eval_config.logging_steps,
        load_best_model_at_end=config.model_config.load_best_model_at_end,
        metric_for_best_model=config.eval_config.metric_for_best_model,
        push_to_hub=config.eval_config.push_to_hub,
        report_to=config.eval_config.report_to,
    )

    return training_args


def get_trainer(model, featurized_dataset, compute_metrics, config):
    # load feature extractor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # get training args
    training_args = get_training_args(config)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=featurized_dataset["train"],
        eval_dataset=featurized_dataset["test"],
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    return trainer