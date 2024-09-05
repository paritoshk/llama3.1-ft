from huggingface_hub import snapshot_download
import os

def download_model():
    model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
    auth_token = os.get_env("HF_AUTH_TOKEN")  # Replace with your token

    local_dir = snapshot_download(repo_id=model_name, use_auth_token=auth_token)
    print(f"Model downloaded to: {local_dir}")

if __name__ == "__main__":
    download_model()