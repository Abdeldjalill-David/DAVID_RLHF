import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
import wandb

def setup_model_and_tokenizer():
    """Setup Qwen 3 8B model and tokenizer"""
    model_name = "Qwen/Qwen2.5-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    return model, tokenizer

def setup_peft_model(model):
    """Setup LoRA configuration for efficient fine-tuning"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model

def prepare_hhh_dataset():
    """Load and prepare HHH dataset"""
    dataset = load_dataset("Anthropic/hh-rlhf")
    
    def format_prompt(example):
        """Format the prompt for training"""
        return {
            "prompt": example["chosen"][:example["chosen"].find("\n\nHuman:")],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
    
    formatted_dataset = dataset.map(format_prompt)
    return formatted_dataset

def create_reward_model(model, tokenizer):
    """Create a simple reward model for RLAIF"""
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    return reward_model

def train_rlaif():
    """Main training function"""
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Setting up LoRA...")
    model = setup_peft_model(model)
    
    print("Loading HHH dataset...")
    dataset = prepare_hhh_dataset()
    
    print("Creating reward model...")
    reward_model = create_reward_model(model, tokenizer)
    
    # PPO Configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        seed=42,
        init_kl_coef=0.2,
        log_with="wandb",
        project_name="qwen3-8b-rlaif",
        tracker_project_name="qwen3-8b-rlaif",
    )
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        peft_config=None,
        reward_model=reward_model,
    )
    
    print("Starting RLAIF training...")
    
    # Training loop
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]
            
            # Get response from model
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                max_new_tokens=128,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.9,
            )
            
            # Compute rewards
            rewards = ppo_trainer.compute_rewards(
                query_tensors, response_tensors
            )
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            print(f"Step completed. Reward: {rewards.mean().item():.4f}")
    
    print("Training completed!")
    
    # Save the model
    model.save_pretrained("./qwen3-8b-rlaif-trained")
    tokenizer.save_pretrained("./qwen3-8b-rlaif-trained")
    print("Model saved to ./qwen3-8b-rlaif-trained")

if __name__ == "__main__":
    train_rlaif() 