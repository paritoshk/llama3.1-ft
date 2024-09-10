# LLaMA 3 70B Model Fine-tuning on RunPod

This project demonstrates how to host and fine-tune the LLaMA 3 70B model using a custom dataset on RunPod.

## Project Structure
```
/workspace/llama3finetune/
├── fine_tuned_llama/ # Directory for the fine-tuned model
├── fine_tuning_dataset/ # Custom dataset for fine-tuning
├── llama3.1-ft/ # Main project repository
├── llama3_ft_env/ # Virtual environment
├── logs/ # Training logs
├── model/ # Pre-trained LLaMA 3 70B model
└── results/ # Fine-tuning results
```


## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/paritoshk/llama3.1-ft.git
   cd llama3.1-ft
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv llama3_ft_env
   source llama3_ft_env/bin/activate
   ```

3. Install required packages:
   ```
   pip install torch perft accelerate transformers datasets runpod sentencepiece huggingface_hub
   ```

4. Set up Hugging Face authentication:
   ```
   export HF_AUTH_TOKEN='your_token_here'
   ```

5. Download the pre-trained model:
   ```
   python src/download_model.py
   ```

6. Prepare the dataset:
   ```
   python src/prepare_dataset.py
   ```

7. Start fine-tuning:
   ```
   python src/fine_tune.py
   nohup python src/fine_tune.py > output.log 2>&1 &
   tail -f output.log
   ```

## Why These Choices?

1. **LLaMA 3 70B Model**: We chose this model for its state-of-the-art performance in natural language processing tasks.

2. **RunPod**: RunPod provides scalable GPU resources, making it ideal for fine-tuning large language models.

3. **Custom Dataset**: We use a custom dataset to tailor the model to specific use cases or domains.

4. **LoRA (Low-Rank Adaptation)**: We implement LoRA for efficient fine-tuning, reducing memory requirements and training time.

5. **8-bit Quantization**: This technique allows us to work with the large model on limited GPU resources.

6. **TensorBoard**: For real-time monitoring of the training process and visualization of metrics.

## Fine-tuning Process

Our `fine_tune.py` script:
1. Loads the pre-trained LLaMA 3 70B model
2. Applies 8-bit quantization
3. Configures LoRA for efficient fine-tuning
4. Prepares the custom dataset
5. Sets up training arguments and initializes the Trainer
6. Runs the fine-tuning process
7. Saves the fine-tuned model and tokenizer

## Monitoring and Evaluation

- Use TensorBoard to monitor training progress:
  ```
  tensorboard --logdir=/workspace/llama3finetune/logs
  ```
- The `DebugTrainer` class provides detailed logging during training.
- After training, use the `visualize_eval` function to analyze model performance.

## Next Steps

- Experiment with different LoRA configurations
- Try various learning rates and batch sizes
- Expand the custom dataset for better performance
- Implement advanced techniques like mixed-precision training

# RunPod Process Automation 
The issue you're experiencing is related to your SSH connection being interrupted when you close your laptop, which is causing the training process to stop. This isn't a problem with the VM itself - the VM is still running, but the process you started over SSH is terminated when the connection drops.
To keep your process running even when you disconnect, you don't need to use Terraform or Docker (although Docker can be useful for other reasons). Instead, you can use a terminal multiplexer like tmux or screen. These tools allow you to create persistent sessions that continue running even when you disconnect from SSH.
Here's how you can use tmux to keep your process running:

If tmux isn't installed on your VM, install it:
```bash

sudo apt-get update
sudo apt-get install tmux

```

Start a new tmux session:
```bash

tmux new -s training

```

Once inside the tmux session, navigate to your project directory and start your training process as usual.
To detach from the tmux session without stopping it, press Ctrl+B, then D. This will return you to your normal terminal, but the process will keep running in the background.
You can now safely close your laptop or disconnect from SSH. The process will continue running on the VM.
When you want to check on your process later, SSH back into your VM and reattach to the tmux session:

```bash

tmux attach -t training

```



This way, even if your SSH connection drops, your training process will continue running in the background. You can reconnect to it at any time to check its progress.
Alternative approach using nohup:
If you prefer not to use tmux, you can also use the nohup command to run your script:

```bash

nohup python your_script.py > output.log 2>&1 &

```

This will start your script in the background, redirect all output to output.log, and keep it running even if you disconnect. You can check the progress by examining the log file:

```bash

tail -f output.log

```

Remember to use htop or ps aux | grep python to find the process ID if you need to stop it later.

# Compare Models

Key Points:
Model Loading: The script loads both the base model (non-fine-tuned) and the fine-tuned model from the saved paths.

Prompt: You can set any prompt (in this case, it's "Explain the concept of quantum computing in simple terms.").

Comparison: It generates and prints responses from both the base model and the fine-tuned model for comparison.

Saving the Fine-Tuned Model: In your training script, the fine-tuned model is saved at the specified output_dir (/workspace/llama3finetune/fine_tuned_llama), so this script loads it directly from there.

Expected Output:
Base Model Response: The response generated by the original model before fine-tuning.
Fine-Tuned Model Response: The response generated by the model after it has been fine-tuned on your dataset.
How to Use:
After fine-tuning your model using the training script, run this compare_models.py script to see the difference in output between the two models for a given prompt. The difference will help you understand how fine-tuning has impacted the model's performance on your specific task or data.










```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model(model_path, prompt):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize and generate output
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
    
    # Decode and print the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def compare_models(base_model_path, fine_tuned_model_path, prompt):
    print("Testing Base Model:")
    base_model_result = test_model(base_model_path, prompt)
    print(f"Base Model Response: {base_model_result}\n")

    print("Testing Fine-Tuned Model:")
    fine_tuned_result = test_model(fine_tuned_model_path, prompt)
    print(f"Fine-Tuned Model Response: {fine_tuned_result}\n")

if __name__ == "__main__":
    prompt = "Summarize this medical note: [Insert example note here]"
    base_model_path = "/workspace/llama3finetune/base_model"
    fine_tuned_model_path = "/workspace/llama3finetune/fine_tuned_llama"
    
    compare_models(base_model_path, fine_tuned_model_path, prompt)
```