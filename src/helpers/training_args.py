from transformers import TrainingArguments

def get_training_args(output_dir: str = "./results") -> TrainingArguments:
    """
    Returns the training arguments for the model training process.
    
    Args:
        output_dir: Directory where the model and training logs are saved.
    
    Returns:
        TrainingArguments: Configuration for training process.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=50,
        load_best_model_at_end=True,
        fp16=True,  # Changed from True to False
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_steps=100,
        eval_strategy="steps",
        warmup_ratio=0.1,
    )
    
