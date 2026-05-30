#!/usr/bin/env python3
"""
03_cpt_training.py — Continued Pre-Training for Domain Knowledge
Part 2 of The Complete LLM Fine-Tuning Guide

Teaches your model domain-specific vocabulary and concepts
using raw text (no labels needed).

Usage: python scripts/03_cpt_training.py
"""

import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, load_dataset

MODEL_NAME = "microsoft/phi-2"
CPT_OUTPUT_DIR = "./medical-cpt-checkpoints"
CPT_MODEL_DIR = "./stage1-cpt-model"
CPT_LEARNING_RATE = 5e-6    # Much lower than SFT — prevents forgetting
CPT_EPOCHS = 1              # 1-2 max for CPT
DOMAIN_RATIO = 0.7          # 70% domain, 30% general


def split_into_chunks(text, max_tokens=2048, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = ' '.join(words[i:i + max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks


def prepare_cpt_dataset(text_files, min_length=100, max_length=2048):
    """Prepare raw text files for CPT training"""
    cpt_data = []
    for filepath in text_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = split_into_chunks(content, max_tokens=max_length)
        for chunk in chunks:
            if len(chunk.split()) < min_length:
                continue
            cpt_data.append({"text": chunk.strip()})
    print(f"Prepared {len(cpt_data)} CPT training chunks")
    return cpt_data


def create_mixed_dataset(domain_data, ratio=0.7, general_examples=500):
    """Mix domain data with general text to prevent forgetting"""
    general_data = load_dataset("allenai/c4", "en", split=f"train[:{general_examples}]", streaming=True)
    general_data = list(general_data)
    n_general = len(general_data)
    n_domain = min(int(n_general * ratio / (1 - ratio)), len(domain_data))
    domain_sample = random.sample(domain_data, n_domain)
    mixed = domain_sample + general_data
    random.shuffle(mixed)
    print(f"Mixed: {n_domain} domain + {n_general} general ({n_domain/(n_domain+n_general):.0%} domain)")
    return Dataset.from_list(mixed)


def evaluate_general_knowledge(model, tokenizer):
    """Check if model still knows basic stuff after CPT"""
    prompts = [
        "What is 15% of 200?",
        "What is the capital of France?",
        "Write a simple Python function to add two numbers.",
        "What year did World War II end?"
    ]
    print("Checking general knowledge retention:")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response[len(prompt):].strip()[:100]}...")


def main():
    assert torch.cuda.is_available(), "GPU required!"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False  # Required when using gradient checkpointing

    # Baseline check
    print("\n=== BEFORE CPT ===")
    evaluate_general_knowledge(model, tokenizer)

    # NOTE: Replace these with your actual medical text files
    # For demo, we create sample data
    sample_medical_text = [{"text": "Myocardial infarction (MI), commonly known as a heart attack..."}]
    print(f"\n⚠️  Using sample data. Replace with your medical text files!")
    mixed_data = Dataset.from_list(sample_medical_text * 50)  # Repeat for demo

    # Train CPT
    cpt_args = TrainingArguments(
        output_dir=CPT_OUTPUT_DIR, learning_rate=CPT_LEARNING_RATE,
        num_train_epochs=CPT_EPOCHS, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, fp16=True,
        gradient_checkpointing=True, logging_steps=50,
        save_strategy="epoch", report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cpt_args, train_dataset=mixed_data,
        dataset_text_field="text", max_seq_length=2048, tokenizer=tokenizer,
    )

    print(f"\n🚀 CPT Training: LR={CPT_LEARNING_RATE}, epochs={CPT_EPOCHS}")
    trainer.train()
    trainer.save_model(CPT_MODEL_DIR)
    tokenizer.save_pretrained(CPT_MODEL_DIR)

    print("\n=== AFTER CPT ===")
    evaluate_general_knowledge(model, tokenizer)

    print(f"\n✅ CPT model saved to {CPT_MODEL_DIR}")
    print("Next: Run 02_sft_training.py on this CPT model, or 04_full_pipeline.py")


if __name__ == "__main__":
    main()
