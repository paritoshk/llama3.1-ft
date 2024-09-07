import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
from peft import get_peft_model, LoraConfig, TaskType
from helpers.utils import tokenize_function, print_trainable_parameters, log_gradients_requirements
from helpers.training_args import get_training_args
from helpers.logging import setup_tensorboard, visualize_eval

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
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    print_trainable_parameters(model)
    log_gradients_requirements(model)

    # Prepare dataset
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select(range(min(500, len(dataset))))

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
