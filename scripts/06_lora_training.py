#!/usr/bin/env python3
"""
06_lora_training.py — LoRA Fine-Tuning (Low-Rank Adaptation)
Part 3 of The Complete LLM Fine-Tuning Guide

Fine-tune any model using a fraction of its parameters.
LoRA trains ~0.06% of weights while achieving 95-98% of full fine-tuning quality.

Usage: python scripts/06_lora_training.py
Prerequisites: Run 01_create_dataset.py first, GPU required
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # Swap for any HuggingFace model
DATASET_PATH = "medical_training_data.jsonl"
OUTPUT_DIR = "./lora-checkpoints"
ADAPTER_DIR = "./lora-adapter-only"             # Tiny (~50 MB)
MERGED_DIR = "./lora-merged-model"              # Full size, single model

# LoRA hyperparameters
LORA_RANK = 16          # Start here; drop to 8 if RAM-tight, 32 if quality isn't enough
LORA_ALPHA = 32         # Scaling factor — usually 2× rank
LORA_DROPOUT = 0.05     # Light regularization

# Training hyperparameters
LEARNING_RATE = 3e-4     # LoRA tolerates higher LR than full fine-tuning (adapters start from zero)
NUM_EPOCHS = 3
BATCH_SIZE = 4           # LoRA uses less memory → can afford bigger batches
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024

# Which layers to adapt (pick one):
#   "minimal"       → ["q_proj", "v_proj"]                         (fast, ~90% quality)
#   "standard"      → ["q_proj", "k_proj", "v_proj", "o_proj"]     (recommended start)
#   "comprehensive" → attention + MLP layers                       (best quality, more memory)
#   "all-linear"    → every linear layer                           (near full FT quality)
TARGET_PRESET = "standard"

TARGET_MAP = {
    "minimal":       ["q_proj", "v_proj"],
    "standard":      ["q_proj", "k_proj", "v_proj", "o_proj"],
    "comprehensive": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    "all-linear":    "all-linear",
}

# Save mode: "adapter" (tiny, swappable) or "merged" (single deployable model) or "both"
SAVE_MODE = "both"


def main():
    # ── 1. GPU check ────────────────────────────────────
    assert torch.cuda.is_available(), "No GPU! In Colab: Runtime → Change runtime type → T4 GPU"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")

    # ── 2. Load tokenizer & model ───────────────────────
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"✅ Base model: {model.num_parameters()/1e9:.2f}B params")

    # ── 3. Apply LoRA ───────────────────────────────────
    target_modules = TARGET_MAP[TARGET_PRESET]

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # e.g. "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%"

    # ── 4. Load dataset ─────────────────────────────────
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]

    split_idx = int(len(data) * 0.9)
    train_data = Dataset.from_list(data[:split_idx])
    val_data = Dataset.from_list(data[split_idx:])
    print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}")

    # ── 5. Train ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        optim="paged_adamw_32bit",       # Memory-efficient optimizer for LoRA
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
    )

    print(f"🚀 LoRA training: r={LORA_RANK}, α={LORA_ALPHA}, "
          f"targets={TARGET_PRESET}, LR={LEARNING_RATE}")
    trainer.train()

    # ── 6. Save ─────────────────────────────────────────
    if SAVE_MODE in ("adapter", "both"):
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        print(f"✅ Adapter saved to {ADAPTER_DIR}  (tiny, swappable)")

    if SAVE_MODE in ("merged", "both"):
        merged = model.merge_and_unload()
        merged.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)
        print(f"✅ Merged model saved to {MERGED_DIR}  (single deployable model)")

    print("\nDone! Next steps:")
    print("  • Test: python scripts/05_inference.py")
    print("  • Multi-adapter: python scripts/09_adapter_switching.py")


if __name__ == "__main__":
    main()
