#!/usr/bin/env python3
"""
11_rlhf_ppo_training.py — PPO Fine-Tuning (Stage 3 of RLHF)
Part 4 of The Complete LLM Fine-Tuning Guide

This script uses Proximal Policy Optimization (PPO) to align the SFT model
using rewards from a trained reward model.

Usage: python scripts/11_rlhf_ppo_training.py
Prerequisites: GPU required
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # SFT base model
REWARD_MODEL_DIR = "./reward-model-medical"     # Path from script 10
OUTPUT_DIR = "./rlhf-medical-final"

# PPO Hyperparameters (scaled down for small resource execution/demo)
BATCH_SIZE = 2
MINI_BATCH_SIZE = 1
PPO_EPOCHS = 2
LEARNING_RATE = 1.41e-5

# In production, PPO parameters look more like:
# BATCH_SIZE = 64, MINI_BATCH_SIZE = 4, PPO_EPOCHS = 4

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def main():
    # 1. GPU Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load tokenizers & models
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading reward model from {REWARD_MODEL_DIR}...")
    # Check if reward model exists; if not, fall back to base model classification for runnability
    if os.path.exists(REWARD_MODEL_DIR):
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_DIR,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
    else:
        print(f"⚠️ Reward model at {REWARD_MODEL_DIR} not found. Falling back to untrained sequence classification on {MODEL_NAME} for demonstration.")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        reward_model.config.pad_token_id = tokenizer.eos_token_id
    
    reward_model.to(device)
    reward_model.eval()

    # Policy model (with value head for PPO)
    print(f"Loading policy model: {MODEL_NAME}...")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    policy_model.to(device)

    # Reference model (frozen copy to calculate KL divergence)
    print(f"Loading reference model: {MODEL_NAME}...")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    ref_model.to(device)

    # 3. Configure PPO
    print("Setting up PPO config...")
    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        init_kl_coef=0.2,      # Control policy drift
        target_kl=6.0,
        adap_kl_ctrl=True,
    )

    # 4. Prepare Dataset
    # We need a query dataset for PPO trainer
    queries = [
        "How do I make money fast?",
        "What should I do for a mild headache?",
        "Explain what a cell is."
    ]
    
    dataset_list = []
    for q in queries:
        input_ids = tokenizer(q, return_tensors="pt").input_ids[0]
        dataset_list.append({
            "query": q,
            "input_ids": input_ids.tolist()
        })
    
    dataset = Dataset.from_list(dataset_list)

    # 5. Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # 6. PPO Training loop
    print("🚀 Starting PPO training loop...")
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 50,
    }

    for epoch in range(1): # Keep epoch count minimal for demo
        for batch in ppo_trainer.dataloader:
            query_tensors = [torch.tensor(q).to(device) for q in batch["input_ids"]]
            
            # Generate responses using current policy
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze(0)[len(query):]) # extract response part only
                
            # Decode responses to feed to reward model
            texts = [
                q_text + " " + tokenizer.decode(r_tensor, skip_special_tokens=True)
                for q_text, r_tensor in zip(batch["query"], response_tensors)
            ]
            
            # Score responses
            rewards = []
            with torch.no_grad():
                for t in texts:
                    inputs = tokenizer(t, return_tensors="pt", padding=True, truncation=True).to(device)
                    score = reward_model(**inputs).logits.squeeze(-1).item()
                    rewards.append(torch.tensor(score))

            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Print stats
            print(f"Query: {batch['query']}")
            print(f"Response texts: {[tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]}")
            print(f"Mean reward: {stats['ppo/mean_reward']:.4f} | KL divergence: {stats['objective/kl']:.4f}")

    # 7. Save
    print(f"Saving final policy model to {OUTPUT_DIR}...")
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done! Next step: Run 12_dpo_training.py")

if __name__ == "__main__":
    main()
