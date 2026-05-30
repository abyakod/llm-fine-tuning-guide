#!/usr/bin/env python3
"""
04_full_pipeline.py — Complete CPT → SFT Training Pipeline
Part 2 of The Complete LLM Fine-Tuning Guide

Runs the full two-stage pipeline:
  Stage 1: CPT (domain knowledge) → Stage 2: SFT (task behavior)

Usage: python scripts/04_full_pipeline.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODEL_NAME = "microsoft/phi-2"
SFT_DATA_PATH = "medical_training_data.jsonl"


def main():
    assert torch.cuda.is_available(), "GPU required!"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # ═══════════════════════════════════════
    # STAGE 1: Continued Pre-Training (CPT)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("Stage 1: CPT — Learning medical domain...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False  # Required when using gradient checkpointing

    # NOTE: Replace with your actual medical corpus
    sample_cpt = [{"text": "Medical domain text here..."}] * 50
    cpt_data = Dataset.from_list(sample_cpt)

    cpt_args = TrainingArguments(
        output_dir="./stage1-cpt-checkpoints",
        learning_rate=5e-6, num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        fp16=True, gradient_checkpointing=True,
        logging_steps=50, save_strategy="epoch", report_to="none",
    )

    cpt_trainer = SFTTrainer(
        model=model, args=cpt_args, train_dataset=cpt_data,
        dataset_text_field="text", max_seq_length=2048, tokenizer=tokenizer,
    )
    cpt_trainer.train()
    cpt_trainer.save_model("./stage1-cpt-model")
    tokenizer.save_pretrained("./stage1-cpt-model")
    print("✅ Stage 1 complete!")

    # ═══════════════════════════════════════
    # STAGE 2: Supervised Fine-Tuning (SFT)
    # ═══════════════════════════════════════
    print("\n" + "=" * 60)
    print("Stage 2: SFT — Learning task behavior...")
    print("=" * 60)

    cpt_model = AutoModelForCausalLM.from_pretrained(
        "./stage1-cpt-model", torch_dtype=torch.float16, device_map="auto"
    )
    cpt_model.config.use_cache = False  # Required when using gradient checkpointing

    with open(SFT_DATA_PATH) as f:
        sft_data = [json.loads(line) for line in f]
    split_idx = int(len(sft_data) * 0.9)
    train_data = Dataset.from_list(sft_data[:split_idx])
    val_data = Dataset.from_list(sft_data[split_idx:])

    sft_args = TrainingArguments(
        output_dir="./stage2-sft-checkpoints",
        num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
        fp16=True, gradient_checkpointing=True, optim="adamw_torch_fused",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=10, report_to="none", remove_unused_columns=False,
    )

    sft_trainer = SFTTrainer(
        model=cpt_model, args=sft_args, train_dataset=train_data,
        eval_dataset=val_data, dataset_text_field="text",
        max_seq_length=1024, tokenizer=tokenizer,
    )
    sft_trainer.train()
    sft_trainer.save_model("./stage2-final-medical-ai")
    tokenizer.save_pretrained("./stage2-final-medical-ai")

    print("\n" + "=" * 60)
    print("🎉 Full Pipeline Complete!")
    print("=" * 60)
    print("Model saved to: ./stage2-final-medical-ai")
    print("Next: Run 05_inference.py to test!")


if __name__ == "__main__":
    main()
