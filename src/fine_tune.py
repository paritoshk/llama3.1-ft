from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
import torch
from peft import get_peft_model, LoraConfig, TaskType

def tokenize_function(examples, tokenizer, max_length=1024):
    texts = [
        f"Subject ID: {examples['subject_id'][i]}\n"
        f"Hospital Admission ID: {examples['hadm_id'][i]}\n"
        f"Number of Notes: {examples['n_notes'][i]}\n"
        f"Extractive Summary: {examples['extractive_notes_summ'][i]}\n"
        f"Target Text: {examples['target_text'][i]}"
        for i in range(len(examples['subject_id']))
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

def main():
    # Load model and tokenizer
    model_path = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # Prepare dataset
    dataset = load_from_disk("/workspace/llama3finetune/fine_tuning_dataset")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4  # Adjust based on your CPU cores
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="/workspace/llama3finetune/results",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Increased from 1
        gradient_accumulation_steps=4,  # Increased from 2
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_dir="/workspace/llama3finetune/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Add gradient clipping
        max_steps=1000,  # Add max steps
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Set up learning rate scheduler
    total_steps = len(tokenized_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps
    )
    trainer.lr_scheduler = scheduler

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    tokenizer.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")

if __name__ == "__main__":
    main()