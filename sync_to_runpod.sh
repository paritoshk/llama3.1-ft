#!/bin/bash

RUNPOD_IP="69.30.85.120"
RUNPOD_PORT="22196"
REMOTE_DIR="/root/llama3finetune"

rsync -avz -e "ssh -p $RUNPOD_PORT" \
    --exclude '.git' \
    --exclude '*.pth' \
    --exclude '*.bin' \
    --exclude '*.model' \
    --exclude 'results' \
    --exclude 'fine_tuning_dataset' \
    ./ root@$RUNPOD_IP:$REMOTE_DIR