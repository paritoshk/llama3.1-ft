import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
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
    model_path = "/workspace/llama3finetune/model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False  # Disable KV cache
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # Ensure all parameters that should require gradients do so
    for name, param in model.named_parameters():
        if "lora" in name or "Lora" in name:
            param.requires_grad = True

    # Prepare dataset
    dataset = load_from_disk("/workspace/llama3finetune/fine_tuning_dataset")
    
     # Use only the first 500 rows of the dataset
    dataset = dataset.select(range(min(500, len(dataset))))
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
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=4,
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        logging_dir="/workspace/llama3finetune/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_steps=100,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    tokenizer.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")

if __name__ == "__main__":
    main()