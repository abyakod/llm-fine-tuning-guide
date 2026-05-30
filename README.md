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
| 3 | **Efficient Training — LoRA, QLoRA & Distillation** | 🔜 Coming soon |
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

### Run the Scripts (Part 2)

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

---

## 📁 Repository Structure

```
llm-fine-tuning-guide/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── LICENSE
└── scripts/
    ├── 01_create_dataset.py           ← Create & validate training data
    ├── 02_sft_training.py             ← Supervised Fine-Tuning
    ├── 03_cpt_training.py             ← Continued Pre-Training
    ├── 04_full_pipeline.py            ← Complete CPT → SFT pipeline
    └── 05_inference.py                ← Test your trained model
```

---

## 🧠 What You'll Build

A **specialized Medical AI Assistant** that:

- ✅ Understands medical terminology (from CPT)
- ✅ Responds in structured, helpful format (from SFT)
- ✅ Knows when things are urgent
- ✅ Runs locally — no API, no subscriptions
- ✅ Costs less than a coffee ($5-10 in compute)

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
