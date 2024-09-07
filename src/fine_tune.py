import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments
from datasets import load_from_disk
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer

# Define DebugTrainer
class DebugTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")
        return loss

def main():
    # Load model and tokenizer
    model_path = "/workspace/llama3finetune/model"
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Verify if any parameters require gradients
    print("Checking if any parameters require gradients...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} requires gradients.")
            break
    else:
        print("No parameters require gradients.")

    # Check LoRA parameters
    print("Checking LoRA parameters...")
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            print(f"{name}: requires_grad={param.requires_grad}")

    # Prepare dataset
    print("Loading and tokenizing dataset...")
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select(range(min(500, len(dataset))))

    # Tokenize dataset
    def tokenize_function(examples):
        texts = [
            f"Subject ID: {examples['subject_id'][i]}\n"
            f"Hospital Admission ID: {examples['hadm_id'][i]}\n"
            f"Number of Notes: {examples['n_notes'][i]}\n"
            f"Extractive Summary: {examples['extractive_notes_summ'][i]}\n"
            f"Target Text: {examples['target_text'][i]}"
            for i in range(len(examples['subject_id']))
        ]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=1024)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="/workspace/llama3finetune/results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
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
        warmup_ratio=0.1,
    )

    # Initialize DebugTrainer
    print("Initializing DebugTrainer...")
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    print("Saving the fine-tuned model and tokenizer...")
    model.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    tokenizer.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()
