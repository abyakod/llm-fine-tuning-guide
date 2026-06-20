#!/usr/bin/env python3
"""
07_qlora_training.py — QLoRA Fine-Tuning (Quantized Low-Rank Adaptation)
Part 3 of The Complete LLM Fine-Tuning Guide

Train 13B–70B models on a single consumer GPU using 4-bit NF4 quantization + LoRA.
A 70B model that normally needs 560 GB fits in ~35 GB with QLoRA.

Usage: python scripts/07_qlora_training.py
Prerequisites: Run 01_create_dataset.py first, GPU required (24+ GB recommended)
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # Swap to "meta-llama/Llama-2-70b-hf" for the real deal
DATASET_PATH = "medical_training_data.jsonl"
OUTPUT_DIR = "./qlora-checkpoints"
ADAPTER_DIR = "./qlora-adapter"

# LoRA settings (same as 06, but we include MLP layers for bigger models)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
    "gate_proj", "up_proj", "down_proj",          # MLP (recommended for large models)
]

# QLoRA-specific training settings (tuned for memory efficiency)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 1              # Keep at 1 for large models
GRAD_ACCUM_STEPS = 16       # Effective batch = 16
MAX_SEQ_LENGTH = 512        # Shorter context saves memory for big models
USE_PACKING = True          # Pack multiple short examples into one sequence


def main():
    # ── 1. GPU check ────────────────────────────────────
    assert torch.cuda.is_available(), "No GPU! In Colab: Runtime → Change runtime type → T4 GPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    # ── 2. 4-bit quantization config (the QLoRA magic) ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                           # Load weights in 4-bit
        bnb_4bit_quant_type="nf4",                   # NF4 > INT4 (denser near zero)
        bnb_4bit_compute_dtype=torch.bfloat16,       # Compute in bf16 (stable)
        bnb_4bit_use_double_quant=True,              # Quantize the quant constants too (~0.4 bits saved)
    )

    # ── 3. Load model in 4-bit ──────────────────────────
    print(f"Loading {MODEL_NAME} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ── 4. Prepare for QLoRA training ───────────────────
    #   Critical step — enables gradient computation on 4-bit model.
    #   Without this you get: "ValueError: You can't train a model loaded in 8/4-bit precision"
    model = prepare_model_for_kbit_training(model)

    # ── 5. Add LoRA adapters ────────────────────────────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 6. Load dataset ─────────────────────────────────
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]

    split_idx = int(len(data) * 0.9)
    train_data = Dataset.from_list(data[:split_idx])
    val_data = Dataset.from_list(data[split_idx:])
    print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}")

    # ── 7. Train (memory-optimized) ─────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,

        # Critical QLoRA settings:
        fp16=False,                        # Don't use fp16 with 4-bit (causes NaN loss)
        bf16=True,                         # Use bf16 instead
        optim="paged_adamw_8bit",          # 8-bit optimizer saves memory
        gradient_checkpointing=True,       # Trade compute for memory

        max_grad_norm=0.3,                 # Gradient clipping for stability
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        packing=USE_PACKING,
    )

    print(f"🚀 QLoRA training: r={LORA_RANK}, 4-bit NF4, LR={LEARNING_RATE}")
    trainer.train()

    # ── 8. Save adapter ─────────────────────────────────
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n✅ QLoRA adapter saved to {ADAPTER_DIR}")
    print("Next: python scripts/05_inference.py  or  python scripts/08_distillation.py")


if __name__ == "__main__":
    main()
