import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from helpers.utils import print_trainable_parameters, log_gradients_requirements
from helpers.logging import setup_tensorboard, visualize_eval
from llama_recipes.configs.training import train_config

# Custom Trainer class to log gradients and other details
class DebugTrainer(Trainer):
    def training_step(self, model, inputs):
        # Perform a standard training step and print loss for debugging
        loss = super().training_step(model, inputs)
        print(f"Loss: {loss.item()}")
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Param {name} has non-zero grad")
        return loss

# Function to check trainable parameters in the model
def check_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")

# Function to preprocess notes field (handling lists of dictionaries)
def preprocess_notes(examples):
    processed_notes = []
    for note_list in examples['notes']:
        # Concatenate the 'description' field from each note dictionary
        concatenated_note = " ".join(note.get('description', '') for note in note_list)
        processed_notes.append(concatenated_note)
    return processed_notes

# Main training function
def main():
    # Model and Tokenizer paths
    model_path = "/workspace/llama3finetune/model"
    output_dir = "/workspace/llama3finetune/fine_tuned_llama"

    # Configure for 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.float16
    )

    # Load model with quantization and necessary settings
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.float16,
    )

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set EOS as pad token

    # LoRA configuration (low-rank adaptation)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters for debugging
    print_trainable_parameters(model)
    log_gradients_requirements(model)
    check_trainable_parameters(model)

    # Training configuration
    train_config.model_name = model_path
    train_config.output_dir = output_dir
    train_config.num_epochs = 1
    train_config.batch_size_training = 4
    train_config.gradient_accumulation_steps = 4
    train_config.lr = 5e-4
    train_config.use_peft = True
    train_config.peft_method = "lora"
    train_config.quantization = True
    train_config.use_fp16 = True
    train_config.context_length = 512
    train_config.log_dir = "/workspace/llama3finetune/logs"

    # Load dataset
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    dataset = load_from_disk(dataset_path)

    # Check for any non-string data and handle it
    def check_validity(examples):
        if isinstance(examples['target_text'], list):
            # If it's a list, join the elements into a single string
            examples['target_text'] = ' '.join(map(str, examples['target_text']))
        elif not isinstance(examples['target_text'], str):
            # If it's not a string or list, convert it to a string
            examples['target_text'] = str(examples['target_text'])
        return examples

    # Tokenize the dataset, ensure inputs and targets are properly formatted
    def tokenize_function(examples):
        # Preprocess notes field
        processed_notes = preprocess_notes(examples)
        
        # Tokenize inputs (notes) and targets (target_text)
        inputs = tokenizer(processed_notes, padding="max_length", truncation=True, max_length=train_config.context_length)
        targets = tokenizer(examples['target_text'], padding="max_length", truncation=True, max_length=train_config.context_length)
        inputs["labels"] = targets["input_ids"]
        return inputs

    # Apply tokenization and validity check
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(check_validity(x)),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Set correct format for PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split into train and validation sets
    dataset_split = tokenized_datasets.train_test_split(test_size=0.1)
    dataset_train = dataset_split['train']
    dataset_val = dataset_split['test']

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Validation Set Length = {len(dataset_val)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size_training,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.lr,
        fp16=train_config.use_fp16,
        logging_dir=train_config.log_dir,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False  # Ensure columns aren't inadvertently removed
    )

    # Set up TensorBoard logging
    writer = setup_tensorboard(log_dir=train_config.log_dir)

    # Initialize DebugTrainer with train and validation datasets
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training and log the process
    print("Starting training...")
    trainer.train()

    # Visualize evaluation metrics and save results
    visualize_eval(trainer, output_dir=training_args.output_dir)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()
