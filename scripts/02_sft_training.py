#!/usr/bin/env python3
"""
02_sft_training.py — Supervised Fine-Tuning for Medical AI
Part 2 of The Complete LLM Fine-Tuning Guide

Usage: python scripts/02_sft_training.py
Prerequisites: Run 01_create_dataset.py first, GPU required
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# Config
MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "medical_training_data.jsonl"
OUTPUT_DIR = "./medical-ai-checkpoints"
FINAL_MODEL_DIR = "./medical-ai-final"
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024


def main():
    # 1. Check GPU
    assert torch.cuda.is_available(), "No GPU! In Colab: Runtime → Change runtime type → T4 GPU"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

    # 2. Load tokenizer & model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False  # Required when using gradient checkpointing
    print(f"✅ {model.num_parameters()/1e9:.2f}B params, {torch.cuda.memory_allocated()/1e9:.2f} GB used")

    # 3. Load dataset
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    split_idx = int(len(data) * 0.9)
    train_data = Dataset.from_list(data[:split_idx])
    val_data = Dataset.from_list(data[split_idx:])
    print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}")

    # 4. Train
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS, learning_rate=LEARNING_RATE,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
        fp16=True, gradient_checkpointing=True, optim="adamw_torch_fused",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=10, report_to="none", dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=train_data,
        eval_dataset=val_data, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer,
    )

    print(f"🚀 Training: LR={LEARNING_RATE}, epochs={NUM_EPOCHS}, batch={BATCH_SIZE*GRAD_ACCUM_STEPS}")
    trainer.train()

    # 5. Save
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"\n✅ Model saved to {FINAL_MODEL_DIR}")
    print("Next: Run 05_inference.py to test your model!")


if __name__ == "__main__":
    main()
