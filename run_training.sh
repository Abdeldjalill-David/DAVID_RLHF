#!/bin/bash

echo "Setting up Qwen 3 8B RLAIF Training..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p qwen3-8b-rlaif-trained

# Run training
echo "Starting RLAIF training..."
python rlaif_training.py

echo "Training completed! Model saved to ./qwen3-8b-rlaif-trained/" 