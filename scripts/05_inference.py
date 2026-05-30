#!/usr/bin/env python3
"""
05_inference.py — Test Your Fine-Tuned Medical AI
Part 2 of The Complete LLM Fine-Tuning Guide

Usage: python scripts/05_inference.py [--model ./medical-ai-final]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    """Load fine-tuned model"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"✅ Model loaded ({model.num_parameters()/1e9:.2f}B params)")
    return model, tokenizer


def ask_medical_ai(model, tokenizer, question="", symptoms="", temperature=0.7):
    """Query your fine-tuned medical AI assistant"""
    if symptoms:
        prompt = f"### Instruction:\nAssess these symptoms and provide guidance\n\n### Input:\n{symptoms}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=400, temperature=temperature,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.split("### Response:")[-1].strip()


def run_tests(model, tokenizer):
    """Run standard test suite"""
    tests = [
        ("Symptom Assessment",
         {"symptoms": "Runny nose, mild sore throat, sneezing, no fever, started yesterday"}),
        ("Emergency Check",
         {"symptoms": "Sudden severe headache, worst of my life, stiff neck, light sensitivity"}),
        ("Health Question",
         {"question": "How do I know if I'm dehydrated?"}),
    ]

    for name, kwargs in tests:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print(f"{'=' * 60}")
        print(ask_medical_ai(model, tokenizer, **kwargs))


def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n🩺 Medical AI Interactive Mode")
    print("Type 'quit' to exit, 'symptoms:' prefix for symptom assessment\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if user_input.lower().startswith("symptoms:"):
            response = ask_medical_ai(model, tokenizer, symptoms=user_input[9:].strip())
        else:
            response = ask_medical_ai(model, tokenizer, question=user_input)
        print(f"\n🤖 AI: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test your fine-tuned Medical AI")
    parser.add_argument("--model", default="./medical-ai-final", help="Path to model")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        run_tests(model, tokenizer)
        print("\n\nTip: Run with --interactive for chat mode!")


if __name__ == "__main__":
    main()
