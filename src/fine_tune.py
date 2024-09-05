from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import torch

def tokenize_function(examples):
    # Combine relevant fields into a single text
    texts = [
        f"Subject ID: {examples['subject_id'][i]}\n"
        f"Hospital Admission ID: {examples['hadm_id'][i]}\n"
        f"Number of Notes: {examples['n_notes'][i]}\n"
        f"Extractive Summary: {examples['extractive_notes_summ'][i]}\n"
        f"Target Text: {examples['target_text'][i]}"
        for i in range(len(examples['subject_id']))
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=1024)

# Load model and tokenizer
model_path = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare dataset
dataset = load_from_disk("./fine_tuning_dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Increase this if you need larger effective batch sizes
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
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
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")