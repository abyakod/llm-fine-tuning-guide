#!/usr/bin/env python3
"""
09_adapter_switching.py — Multi-Adapter Training & Runtime Switching
Part 3 of The Complete LLM Fine-Tuning Guide

Train separate LoRA adapters for different domains, then hot-swap
between specialists at runtime. One 14 GB base model + four 50 MB
adapters replaces four separate 14 GB fine-tuned models.

Usage:
  python scripts/09_adapter_switching.py --mode train     # Train all adapters
  python scripts/09_adapter_switching.py --mode switch    # Demo runtime switching
  python scripts/09_adapter_switching.py --mode both      # Train then demo

Prerequisites: Training data files for each domain, GPU required
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
BASE_MODEL = "microsoft/phi-2"
ADAPTER_BASE_DIR = "./adapters"

# Domain configs — add/remove domains as needed
DOMAINS = {
    "medical": {
        "data": "medical_training_data.jsonl",
        "description": "Clinical triage and medical Q&A",
    },
    "legal": {
        "data": "legal_training_data.jsonl",
        "description": "Legal document review and analysis",
    },
    "code": {
        "data": "code_training_data.jsonl",
        "description": "Code generation and debugging",
    },
}

# LoRA settings (shared across all adapters for consistency)
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training settings
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024


def train_adapters():
    """Train a separate LoRA adapter for each domain."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir="./adapter-checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="no",         # We save manually per adapter
        report_to="none",
        remove_unused_columns=False,
    )

    for domain_name, config in DOMAINS.items():
        save_path = os.path.join(ADAPTER_BASE_DIR, domain_name)

        # Skip if data file doesn't exist
        if not os.path.exists(config["data"]):
            print(f"⚠️  Skipping {domain_name}: {config['data']} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Training {domain_name} adapter: {config['description']}")
        print(f"{'='*50}")

        # Load fresh base model each time (adapters must start clean)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False

        # Apply LoRA
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM", bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load domain data
        with open(config["data"]) as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)

        # Train
        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=dataset, tokenizer=tokenizer,
            dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
        )
        trainer.train()

        # Save adapter only (~50 MB per domain!)
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ {domain_name} adapter saved to {save_path}")

        # Free memory before next domain
        del model, trainer
        torch.cuda.empty_cache()

    print(f"\n✅ All adapters trained! Saved under {ADAPTER_BASE_DIR}/")


def demo_switching():
    """Demonstrate hot-swapping between specialist adapters."""
    print("\n" + "=" * 50)
    print("ADAPTER SWITCHING DEMO")
    print("=" * 50)

    # Load base model ONCE (stays in memory)
    print(f"\nLoading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"✅ Base model loaded (stays in memory)")

    # Test prompts per domain
    test_prompts = {
        "medical": "Assess: Patient has chest pain and shortness of breath",
        "legal":   "Review: The tenant has not paid rent for 3 months",
        "code":    "Write a Python function to merge two sorted lists",
    }

    for domain_name, prompt in test_prompts.items():
        adapter_path = os.path.join(ADAPTER_BASE_DIR, domain_name)
        if not os.path.exists(adapter_path):
            print(f"\n⚠️  {domain_name} adapter not found at {adapter_path}")
            continue

        # Load adapter on top of base model (~2 seconds, not 30!)
        print(f"\n🔄 Switching to {domain_name} specialist...")
        specialist = PeftModel.from_pretrained(base_model, adapter_path)
        specialist.eval()

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        with torch.no_grad():
            output = specialist.generate(
                **inputs, max_new_tokens=150, do_sample=False,
                repetition_penalty=1.2,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"📝 Prompt: {prompt}")
        print(f"🤖 {domain_name.title()} AI: {response[len(prompt):][:300]}")

        del specialist
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Multi-adapter training & switching")
    parser.add_argument("--mode", choices=["train", "switch", "both"], default="both")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU required"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    if args.mode in ("train", "both"):
        train_adapters()
    if args.mode in ("switch", "both"):
        demo_switching()


if __name__ == "__main__":
    main()
