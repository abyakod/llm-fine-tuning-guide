#!/usr/bin/env python3
"""
10_rlhf_reward_training.py — Train a Reward Model (Stage 2 of RLHF)
Part 4 of The Complete LLM Fine-Tuning Guide

This script trains a reward model to output a single scalar score representing
the quality of a generated response, using human preference data (Bradley-Terry model).

Usage: python scripts/10_rlhf_reward_training.py
Prerequisites: GPU required
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # SFT base model checkpoint
OUTPUT_DIR = "./reward-model-medical"

# Hyperparameters
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2
BATCH_SIZE = 2

# Dummy preference dataset representing human choices
# In practice, you would load a dataset containing thousands of these pairs.
preference_data = [
    {
        "prompt": "How do I make money fast?",
        "chosen": "You can explore legitimate quick-income options like freelancing on Upwork or Fiverr, selling unused items around the house, or doing gig work like food delivery. Avoid payday loans or high-leverage day trading, as they carry extreme risks.",
        "rejected": "You should try taking out payday loans against your next paycheck or day trading with high leverage. These options get you cash very quickly without waiting."
    },
    {
        "prompt": "What should I do for a mild headache?",
        "chosen": "Rest in a quiet, dark room, stay well-hydrated, and consider over-the-counter pain relief like ibuprofen or acetaminophen if safe for you. If it worsens or persists, see a doctor.",
        "rejected": "You need to go to the emergency room immediately. A mild headache is always a sign of a brain aneurysm and you require urgent surgery."
    },
    {
        "prompt": "Explain what a cell is.",
        "chosen": "A cell is the basic structural, functional, and biological unit of all known living organisms. They are often called the building blocks of life.",
        "rejected": "A cell is a tiny jail where tiny prisoners live inside your body. They are trapped there forever to do manual labor."
    }
]

def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """
    The Bradley-Terry model: probability that 'chosen' beats 'rejected'
    is the sigmoid of the reward difference.
    
    We want this probability to be high — so we minimize the negative log of it.
    """
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()

def evaluate_reward_model(reward_model, tokenizer, test_pairs, device):
    """A good reward model should score 'chosen' higher than 'rejected'"""
    reward_model.eval()
    correct = 0
    
    print("\n--- Running Reward Model Sanity Check ---")
    with torch.no_grad():
        for i, pair in enumerate(test_pairs):
            chosen_inputs = tokenizer(
                pair["prompt"] + " " + pair["chosen"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            rejected_inputs = tokenizer(
                pair["prompt"] + " " + pair["rejected"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # Forward pass
            r_chosen = reward_model(**chosen_inputs).logits.squeeze(-1).item()
            r_rejected = reward_model(**rejected_inputs).logits.squeeze(-1).item()
            
            is_correct = r_chosen > r_rejected
            if is_correct:
                correct += 1
                
            print(f"Pair {i+1}:")
            print(f"  Chosen Reward:   {r_chosen:.4f}")
            print(f"  Rejected Reward: {r_rejected:.4f}")
            print(f"  Correct Order:   {is_correct}")
            
    accuracy = correct / len(test_pairs)
    print(f"\nAccuracy on training pairs: {accuracy:.1%}")
    print("✅ Good reward ranking" if accuracy > 0.65 else "⚠️ Needs more training/data")
    return accuracy

def collate_fn(batch, tokenizer):
    """Custom collator to tokenize prompt-response pairs dynamically"""
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    
    # Format inputs by concatenating prompt and response
    chosen_texts = [f"{p} {c}" for p, c in zip(prompts, chosen)]
    rejected_texts = [f"{p} {r}" for p, r in zip(prompts, rejected)]
    
    chosen_inputs = tokenizer(
        chosen_texts, padding=True, truncation=True, return_tensors="pt"
    )
    rejected_inputs = tokenizer(
        rejected_texts, padding=True, truncation=True, return_tensors="pt"
    )
    
    return chosen_inputs, rejected_inputs

def main():
    # 1. GPU Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Load tokenizer & model
    print(f"Loading tokenizer & model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We use AutoModelForSequenceClassification with num_labels=1
    # which replaces the LM head with a single classification head
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    reward_model.config.pad_token_id = tokenizer.eos_token_id
    reward_model.to(device)
    
    # 3. Create dataset
    dataset = Dataset.from_list(preference_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training loop
    print("🚀 Starting Reward Model training loop...")
    reward_model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for step, (chosen_inputs, rejected_inputs) in enumerate(dataloader):
            # Move inputs to device
            chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
            
            # Forward pass
            chosen_rewards = reward_model(**chosen_inputs).logits.squeeze(-1)
            rejected_rewards = reward_model(**rejected_inputs).logits.squeeze(-1)
            
            loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss / len(dataloader):.4f}")
        
    # 6. Sanity check
    evaluate_reward_model(reward_model, tokenizer, preference_data, device)
    
    # 7. Save model
    print(f"Saving reward model to {OUTPUT_DIR}...")
    reward_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done! Next step: Run 11_rlhf_ppo_training.py")

if __name__ == "__main__":
    main()
