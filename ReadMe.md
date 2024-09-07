# Self trying to host and fine tune LLaMA 3 70B model using dummy dataset on runpod

## Steps

1. Create a new conda environment
2. Install pytorch
3. Install transformers
4. Install datasets
5. Install runpod
6. Install runpod-cli
7. Install sentencepiece


python3 -m venv llama3_ft_env
source llama3_ft_env/bin/activate

pip install torch transformers datasets runpod sentencepiece huggingface_hub
git clone https://github.com/paritoshk/llama3.1-ft.git
cd llama3.1-ft

export HF_AUTH_TOKEN='your_token_here'

python src/download_model.py
python src/prepare_dataset.py
python src/fine_tune.py