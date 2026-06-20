# 📚 The Complete LLM Fine-Tuning Guide — Code Repository

> All scripts and code from the Medium blog series: *From zero to training your own AI*

[![Medium](https://img.shields.io/badge/Medium-Blog%20Series-black?style=for-the-badge&logo=medium)](https://medium.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)](https://python.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-Free-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com)

---

## 📖 Blog Series (on Medium)

| Part | Title | Blog Link |
|------|-------|-----------| 
| 1 | **Foundations — Understanding the Landscape** | [Read on Medium →](https://medium.com) |
| 2 | **Hands-On SFT & CPT** | [Read on Medium →](https://medium.com) |
| 3 | **Efficient Training — LoRA, QLoRA & Distillation** | [Read on Medium →](https://medium.com) |
| 4 | **Advanced Alignment — RLHF, DPO, RLVR & RLER** | 🔜 Coming soon |

> **Start with the blog posts** for full context, explanations, and visuals.  
> This repo contains the runnable code referenced in each post.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Colab account (free!) or a local GPU
- HuggingFace account (free)

### Setup

```bash
# Clone the repo
git clone https://github.com/abyakod/llm-fine-tuning-guide.git
cd llm-fine-tuning-guide

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Run the Scripts (Part 2 — SFT & CPT)

```bash
# Step 1: Create the training dataset
python scripts/01_create_dataset.py

# Step 2: Train with SFT
python scripts/02_sft_training.py

# Step 3 (optional): CPT for domain knowledge
python scripts/03_cpt_training.py

# Step 4 (optional): Full CPT → SFT pipeline
python scripts/04_full_pipeline.py

# Step 5: Test your model
python scripts/05_inference.py
python scripts/05_inference.py --interactive   # Chat mode!
```

### Run the Scripts (Part 3 — LoRA, QLoRA & Distillation)

```bash
# LoRA fine-tuning (trains ~0.06% of parameters)
python scripts/06_lora_training.py

# QLoRA fine-tuning (13B–70B models on consumer GPUs)
python scripts/07_qlora_training.py

# Knowledge distillation (compress large → small model)
python scripts/08_distillation.py

# Multi-adapter training & hot-swapping
python scripts/09_adapter_switching.py --mode train    # Train adapters
python scripts/09_adapter_switching.py --mode switch   # Demo switching
python scripts/09_adapter_switching.py --mode both     # Both
```

---

## 📁 Repository Structure

```
llm-fine-tuning-guide/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── LICENSE
└── scripts/
    │
    │  # Part 2: Hands-On SFT & CPT
    ├── 01_create_dataset.py           ← Create & validate training data
    ├── 02_sft_training.py             ← Supervised Fine-Tuning
    ├── 03_cpt_training.py             ← Continued Pre-Training
    ├── 04_full_pipeline.py            ← Complete CPT → SFT pipeline
    ├── 05_inference.py                ← Test your trained model
    │
    │  # Part 3: Efficient Training
    ├── 06_lora_training.py            ← LoRA fine-tuning
    ├── 07_qlora_training.py           ← QLoRA (4-bit quantized LoRA)
    ├── 08_distillation.py             ← Knowledge distillation
    └── 09_adapter_switching.py        ← Multi-adapter train & switch
```

---

## 🧠 What You'll Build

A **specialized Medical AI Assistant** that:

- ✅ Understands medical terminology (from CPT)
- ✅ Responds in structured, helpful format (from SFT)
- ✅ Trains on massive 70B models using consumer hardware (LoRA/QLoRA)
- ✅ Runs multiple specialists from one base model (Adapters)
- ✅ Compresses into a tiny, fast model (Distillation)
- ✅ Runs locally — no API, no subscriptions
- ✅ Costs less than a coffee ($5-15 in compute)

The code adapts to **any domain** — just swap the training data!  
(Legal, financial, educational, coding, cooking, etc.)

---

## 📊 Expected Results

| Model | Terminology | Task Format | Overall |
|---|---|---|---|
| Base model (no tuning) | ❌ Poor | ❌ Generic | ❌ |
| SFT only | ⚠️ Okay | ✅ Good | ⚠️ |
| CPT only | ✅ Great | ❌ Generic | ⚠️ |
| **CPT + SFT** | **✅ Great** | **✅ Great** | **✅** |

| Training Method | 7B VRAM | 70B VRAM | Quality | Speed |
|---|---|---|---|---|
| Full fine-tuning | 112 GB | 560 GB | 100% | Baseline |
| LoRA | 18 GB | 160 GB | 97-98% | 2× faster |
| **QLoRA** | **6 GB** | **40 GB** | **95-97%** | **1.5× faster** |

---

## ⚙️ Hardware Requirements

| Setup | GPU Memory | Works? |
|-------|-----------|--------|
| Google Colab (Free T4) | 15 GB | ✅ Perfect |
| RTX 3060/4060 | 8-12 GB | ✅ Small batch |
| RTX 3090/4090 | 24 GB | ✅ Comfortable |
| CPU only | — | ❌ Too slow |

---

## 🤝 Contributing

Found a bug? Have a suggestion?

1. Fork → Branch → Commit → PR

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Code companion to the [Medium blog series](https://medium.com). Star ⭐ if it helped!* 🚀
