import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from helpers.logging import setup_tensorboard, visualize_eval

def preprocess_notes(examples):
    return [" ".join(note.get('description', '') for note in note_list) for note_list in examples['notes']]

def tokenize_function(examples, tokenizer, max_length):
    processed_notes = preprocess_notes(examples)
    inputs = tokenizer(processed_notes, padding="max_length", truncation=True, max_length=max_length)
    targets = tokenizer(examples['target_text'], padding="max_length", truncation=True, max_length=max_length)
    inputs["labels"] = targets["input_ids"]
    return inputs

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    model_path = "/workspace/llama3finetune/model"
    output_dir = "/workspace/llama3finetune/fine_tuned_llama"
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    log_dir = "/workspace/llama3finetune/logs"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.float16
    )

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    dataset = load_from_disk(dataset_path)
    max_length = 512  # Adjust this based on your needs

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split the dataset into train and validation
    train_val_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        fp16=True,
        logging_dir=log_dir,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    visualize_eval(trainer, output_dir=output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()