import torch
from transformers import AutoTokenizer
from typing import List, Dict

# Tokenization function
def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int = 1024) -> Dict[str, List[int]]:
    """
    Tokenizes the input data with the provided tokenizer.
    
    Args:
        examples: A dictionary containing the raw data for tokenization.
        tokenizer: The tokenizer to use for encoding.
        max_length: The maximum length of tokens per input sequence.
    
    Returns:
        Tokenized input sequences in dictionary format.
    """
    texts = [
        f"Subject ID: {examples['subject_id'][i]}\n"
        f"Hospital Admission ID: {examples['hadm_id'][i]}\n"
        f"Number of Notes: {examples['n_notes'][i]}\n"
        f"Extractive Summary: {examples['extractive_notes_summ'][i]}\n"
        f"Target Text: {examples['target_text'][i]}"
        for i in range(len(examples['subject_id']))
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

# Helper function to print and log trainable parameters
def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.
    
    Args:
        model: The model to check for trainable parameters.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}%")

# Helper function to check if parameters require gradients
def log_gradients_requirements(model: torch.nn.Module) -> None:
    """
    Logs the parameters that require gradients for training.
    
    Args:
        model: The model to check.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} requires gradient.")
