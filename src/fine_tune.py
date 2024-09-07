import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.train_utils import train, get_dataloader_kwargs
from llama_recipes.configs.training import train_config
from llama_recipes.configs.datasets import dataset_config

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

    # Set up training configuration
    train_config.model_name = model_path
    train_config.output_dir = output_dir
    train_config.dataset = "custom_dataset"  # We'll use this to identify our custom dataset
    train_config.num_epochs = 3
    train_config.batch_size_training = 4
    train_config.gradient_accumulation_steps = 4
    train_config.lr = 2e-4
    train_config.use_peft = True
    train_config.peft_method = "lora"
    train_config.quantization = True
    train_config.use_fp16 = True
    train_config.context_length = 512
    train_config.log_dir = "/workspace/llama3finetune/logs"

    # Set up dataset configuration
    dataset_config.dataset_path = "/workspace/llama3finetune/fine_tuning_dataset"
    dataset_config.train_split = "train"
    dataset_config.test_split = "test"

    # Load and preprocess the dataset
    dataset = load_from_disk(dataset_config.dataset_path)
    dataset = dataset.select(range(min(500, len(dataset))))  # Subset for testing

    # Preprocess the dataset
    dataset_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    dataset_val = get_preprocessed_dataset(tokenizer, dataset_config, split="test")

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Validation Set Length = {len(dataset_val)}")

    # Create DataLoader for training
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    # Start training
    results = train(model, train_dataloader, tokenizer, eval_dataloader=None, train_config=train_config)

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()