import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to generate responses from the model
def generate_response(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test function to compare fine-tuned vs non-fine-tuned model
def compare_models():
    # Paths to the base and fine-tuned models
    base_model_path = "/workspace/llama3finetune/model"  # Path to the base (non-fine-tuned) model
    fine_tuned_model_path = "/workspace/llama3finetune/fine_tuned_llama"  # Path to the fine-tuned model
    
    # Load base model and tokenizer
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load fine-tuned model and tokenizer
    print("Loading fine-tuned model...")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, device_map="auto", torch_dtype=torch.float16)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    # Test prompt
    prompt = "Explain the concept of quantum computing in simple terms."

    # Generate response from base model
    print("\nGenerating response from base model...")
    base_response = generate_response(base_model, base_tokenizer, prompt)
    print(f"Base Model Response: {base_response}\n")

    # Generate response from fine-tuned model
    print("\nGenerating response from fine-tuned model...")
    fine_tuned_response = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt)
    print(f"Fine-Tuned Model Response: {fine_tuned_response}\n")

if __name__ == "__main__":
    compare_models()
