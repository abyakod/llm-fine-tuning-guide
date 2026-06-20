#!/usr/bin/env python3
"""
08_distillation.py — Knowledge Distillation
Part 3 of The Complete LLM Fine-Tuning Guide

Train a small, fast student model to mimic a large teacher model.
The student learns from the teacher's soft probability distributions,
capturing reasoning and uncertainty — not just hard answers.

Usage: python scripts/08_distillation.py
Prerequisites: A trained teacher model (e.g. from 06/07), GPU required
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset as TorchDataset

# ── Config ──────────────────────────────────────────────
TEACHER_PATH = "./lora-merged-model"            # Your trained large model
STUDENT_NAME = "microsoft/phi-2"                # Fresh small model to train
DATASET_PATH = "medical_training_data.jsonl"
OUTPUT_DIR = "./distilled-student"

# Distillation hyperparameters
TEMPERATURE = 4.0    # Higher = softer teacher distribution = more knowledge transferred
ALPHA = 0.5          # Balance: 0 = only hard labels, 1 = only distillation
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 512

# Device placement
TEACHER_DEVICE = "cuda:0"
STUDENT_DEVICE = "cuda:0"      # Use "cuda:1" if you have 2 GPUs


# ── Distillation Loss ───────────────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Combined loss = α × distillation_loss + (1-α) × hard_label_loss

    The teacher's soft probabilities carry rich information:
      Hard label:  "Paris"
      Soft label:  Paris: 92%, Lyon: 4%, Marseille: 2% ...
      → Student learns relationships between answers, not just the right one.

    Temperature scales the softmax to reveal more of the teacher's distribution.
    Multiply by T² to compensate for the gradient magnitude change.
    """
    # Soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence: how different is student from teacher?
    distil_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)

    # Standard cross-entropy with true labels
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * distil_loss + (1 - alpha) * hard_loss


# ── Simple Dataset ──────────────────────────────────────
class TextDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ── Comparison Helper ───────────────────────────────────
def compare_models(teacher, student, tokenizer, prompts, device):
    """Side-by-side comparison of teacher vs distilled student."""
    print("\n" + "=" * 60)
    print("TEACHER vs STUDENT COMPARISON")
    print("=" * 60)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            t_out = teacher.generate(**inputs, max_new_tokens=150, do_sample=False)
            s_out = student.generate(**inputs, max_new_tokens=150, do_sample=False)

        t_text = tokenizer.decode(t_out[0], skip_special_tokens=True)
        s_text = tokenizer.decode(s_out[0], skip_special_tokens=True)

        print(f"\n📝 Prompt: {prompt}")
        print(f"🎓 Teacher: {t_text[len(prompt):][:200]}")
        print(f"🧑‍🎓 Student: {s_text[len(prompt):][:200]}")
        print("-" * 40)


def main():
    assert torch.cuda.is_available(), "GPU required for distillation"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Load teacher (frozen, eval mode) ─────────────
    print(f"Loading teacher from {TEACHER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.float16, device_map=TEACHER_DEVICE,
        trust_remote_code=True,
    )
    teacher.eval()
    print(f"✅ Teacher loaded ({teacher.num_parameters()/1e9:.2f}B params, frozen)")

    # ── 2. Load student (trainable) ─────────────────────
    print(f"Loading student: {STUDENT_NAME}...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_NAME, torch_dtype=torch.float16, device_map=STUDENT_DEVICE,
        trust_remote_code=True,
    )
    student.train()
    print(f"✅ Student loaded ({student.num_parameters()/1e9:.2f}B params, trainable)")

    # ── 3. Prepare data ─────────────────────────────────
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    texts = [item["text"] for item in data]

    dataset = TextDataset(texts, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Dataset: {len(dataset)} examples")

    # ── 4. Train with distillation ──────────────────────
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Distillation: T={TEMPERATURE}, α={ALPHA}, LR={LEARNING_RATE}")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(STUDENT_DEVICE)
            labels = batch["labels"].to(STUDENT_DEVICE)

            # Teacher predictions (no gradient)
            with torch.no_grad():
                t_logits = teacher(input_ids.to(TEACHER_DEVICE)).logits.to(STUDENT_DEVICE)

            # Student predictions
            s_logits = student(input_ids).logits

            # Combined distillation loss
            loss = distillation_loss(s_logits, t_logits, labels, TEMPERATURE, ALPHA)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ── 5. Save distilled student ───────────────────────
    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Distilled student saved to {OUTPUT_DIR}")

    # ── 6. Quick comparison ─────────────────────────────
    test_prompts = [
        "Patient has chest pain and shortness of breath",
        "What are symptoms of appendicitis?",
        "Is it safe to mix ibuprofen and acetaminophen?",
    ]
    compare_models(teacher, student, tokenizer, test_prompts, STUDENT_DEVICE)


if __name__ == "__main__":
    main()
