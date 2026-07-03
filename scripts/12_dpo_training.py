#!/usr/bin/env python3
"""
12_dpo_training.py — Direct Preference Optimization (DPO) Fine-Tuning
Part 4 of The Complete LLM Fine-Tuning Guide

This script demonstrates how to fine-tune an LLM directly on pairwise human preferences
without using a separate reward model or PPO training loop.

Usage: python scripts/12_dpo_training.py
Prerequisites: GPU required
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # SFT base model
OUTPUT_DIR = "./dpo-medical-final"

# DPO Hyperparameters (scaled down for small resource execution/demo)
BETA = 0.1                      # Preference temperature
LEARNING_RATE = 5e-7            # DPO is sensitive, uses a much lower LR than SFT
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2

# Preference training dataset (prompt + chosen + rejected response)
preference_data = [
    {
        "prompt": "### Instruction:\nHow do I make money fast?\n\n### Response:\n",
        "chosen": "Here are legitimate quick-income options:\n- Freelancing (Upwork, Fiverr) — start earning in days\n- Selling unused items for immediate cash\n- Gig work (DoorDash, Uber) — same-day pay\n- Tutoring or pet-sitting in your area\n\nAvoid high-risk shortcuts like payday loans (400%+ APR) — they often create bigger problems than they solve.",
        "rejected": "Here are some high-risk, high-reward options: payday loans against your next paycheck, day trading with leverage, or credit card cash advances."
    },
    {
        "prompt": "### Instruction:\nWhat should I do for a mild headache?\n\n### Response:\n",
        "chosen": "For a mild headache, try resting in a quiet, dark room, staying hydrated, and applying a cold compress to your forehead. Over-the-counter pain relievers like ibuprofen or acetaminophen may help if appropriate for you. Consult a doctor if the pain persists.",
        "rejected": "A mild headache is extremely dangerous and you must check yourself into emergency brain surgery immediately to prevent an aneurysm."
    },
    {
        "prompt": "### Instruction:\nExplain what a cell is.\n\n### Response:\n",
        "chosen": "A cell is the smallest biological, structural, and functional unit of all living organisms. Often called the 'building blocks of life', cells contain genetic material and carry out essential processes.",
        "rejected": "Cells are tiny cages inside your body containing tiny prisoners that manually run your body's systems under threat of punishment."
    }
]

def main():
    # 1. GPU Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load tokenizer & model
    print(f"Loading SFT model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model in BF16/FP16 depending on GPU capabilities
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None
    )
    
    # 3. Create dataset
    dpo_dataset = Dataset.from_list(preference_data)

    # 4. Configure DPO Training
    print("Configuring DPOTrainer...")
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no", # Do not save checkpoint within training for speed
        bf16=True if device.type == "cuda" else False,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # 5. Initialize DPOTrainer
    # DPOTrainer will automatically create a frozen copy of the model
    # as the reference model (ref_model) internally.
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )

    # 6. Train model
    print("🚀 Starting DPO training...")
    dpo_trainer.train()

    # 7. Save model
    print(f"Saving DPO aligned model to {OUTPUT_DIR}...")
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done! Next step: Run 13_rlvr_training.py")

if __name__ == "__main__":
    main()
