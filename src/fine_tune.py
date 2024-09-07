import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from helpers.utils import print_trainable_parameters, log_gradients_requirements
from helpers.logging import setup_tensorboard, visualize_eval

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
    output_dir = "/workspace/llama3finetune/fine_tuned_llama"

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
    check_trainable_parameters(model)

    # Set up training configuration
    train_config.model_name = model_path
    train_config.output_dir = output_dir
    train_config.dataset = "custom_dataset"
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

    # Set up dataset configuration
    dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    train_split = "train"
    test_split = "test"

    # Load and preprocess the dataset
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select(range(min(500, len(dataset))))  # Subset for testing

    # Preprocess the dataset
    dataset_train = dataset[train_split]
    dataset_val = dataset[test_split]

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
    )

    # TensorBoard setup
    writer = setup_tensorboard(log_dir=train_config.log_dir)

    # Initialize DebugTrainer
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Visualize evaluation and save metrics
    visualize_eval(trainer, output_dir=training_args.output_dir)

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()