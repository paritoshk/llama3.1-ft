from huggingface_hub import snapshot_download
import os

def download_model():
    model_path = "/workspace/llama3finetune/model"
    model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
    snapshot_download(model_name, local_dir=model_path, token=auth_token)
    auth_token = os.getenv("HF_AUTH_TOKEN")  # Replace with your token
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()