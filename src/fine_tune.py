import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
from helpers.utils import tokenize_function, print_trainable_parameters, log_gradients_requirements
from helpers.training_args import get_training_args
from helpers.logging import setup_tensorboard, visualize_eval

# Define DebugTrainer to print loss and check requires_grad
class DebugTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Param {name} has non-zero grad")
        return loss

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

def main():
    # Load model and tokenizer
    model_path = "/workspace/llama3finetune/model"
    
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare model for int8 training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
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

    # Print trainable parameters
    print_trainable_parameters(model)
    log_gradients_requirements(model)

    # Debug: Check if parameters require gradients
    check_trainable_parameters(model)

    # Prepare dataset
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select(range(min(500, len(dataset))))  # Subset for testing

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # Training arguments
    training_args = get_training_args(output_dir="/workspace/llama3finetune/results")

    # TensorBoard setup
    writer = setup_tensorboard(log_dir="/workspace/llama3finetune/logs")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Visualize evaluation and save metrics
    visualize_eval(trainer, output_dir=training_args.output_dir)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    tokenizer.save_pretrained("/workspace/llama3finetune/fine_tuned_llama")
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()