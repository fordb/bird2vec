from transformers import TrainingArguments, Trainer


def get_training_args(config):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        push_to_hub=config.push_to_hub,
        report_to=config.report_to,
    )

    return training_args


def get_trainer(model, featurized_dataset, feature_extractor, compute_metrics, config):
    # get training args
    training_args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=featurized_dataset["train"],
        eval_dataset=featurized_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    return trainer