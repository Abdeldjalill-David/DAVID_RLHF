# Qwen 3 8B RLAIF Training

This project implements Reinforcement Learning from AI Feedback (RLAIF) training for the Qwen 3 8B model using the HHH (Helpful, Harmless, Honest) dataset.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to Weights & Biases (optional, for logging):
```bash
wandb login
```

## Training

Run the RLAIF training:
```bash
python rlaif_training.py
```

## What it does

- Loads Qwen 3 8B model with 8-bit quantization
- Uses LoRA for efficient fine-tuning
- Loads the Anthropic/hh-rlhf dataset
- Implements PPO (Proximal Policy Optimization) for RLHF
- Uses a simple reward model based on the same model
- Trains for 3 epochs with configurable parameters

## Model Output

The trained model will be saved to `./qwen3-8b-rlaif-trained/`

## Key Features

- **Simple Implementation**: Minimal code complexity
- **Efficient Training**: Uses LoRA and 8-bit quantization
- **HHH Dataset**: Uses Anthropic's helpful, harmless, honest dataset
- **PPO Training**: Standard RLHF approach with PPO
- **Wandb Logging**: Optional experiment tracking

## Configuration

Key parameters in `rlaif_training.py`:
- Learning rate: 1e-5
- Batch size: 4
- LoRA rank: 16
- Training epochs: 3
- Max new tokens: 128

## Requirements

- GPU with sufficient VRAM (recommended 24GB+)
- Python 3.8+
- CUDA-compatible PyTorch
