#!/usr/bin/env python3
"""
13_rlvr_training.py — Reinforcement Learning with Verifiable Rewards (RLVR)
Part 4 of The Complete LLM Fine-Tuning Guide

This script demonstrates how to train a model using an objective, verifiable
reward function (such as code execution or mathematical correctness checks)
instead of human preference ratings.

Usage: python scripts/13_rlvr_training.py
Prerequisites: GPU required
"""

import os
import re
import subprocess
import tempfile
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"                  # SFT base model
OUTPUT_DIR = "./rlvr-aligned-model"

# Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
MINI_BATCH_SIZE = 1
PPO_EPOCHS = 1

# List of coding/math problems with verification parameters
verifiable_problems = [
    {
        "prompt": "Write a python function `is_prime(n)` that returns True if a number is prime, else False. Answer with only code.",
        "type": "code",
        "test_cases": [
            {"call": "is_prime(7)", "expected": "True"},
            {"call": "is_prime(10)", "expected": "False"},
            {"call": "is_prime(1)", "expected": "False"},
            {"call": "is_prime(2)", "expected": "True"},
        ]
    },
    {
        "prompt": "Solve this equation: 3 * x + 7 = 22. What is the value of x? Answer with a single number.",
        "type": "math",
        "correct_answer": 5.0
    }
]

def verify_code_solution(generated_code: str, test_cases: list) -> float:
    """
    Runs generated code against test cases in a subprocess.
    Returns 1.0 reward if all tests pass, 0.0 otherwise.
    """
    # Clean code: extract python block if wrapped in markdown
    code_cleaned = generated_code
    if "```python" in code_cleaned:
        code_cleaned = code_cleaned.split("```python")[-1].split("```")[0]
    elif "```" in code_cleaned:
        code_cleaned = code_cleaned.split("```")[1].split("```")[0]

    # Create temporary file to execute
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_cleaned)
        f.write("\n\n")
        # Append test case assertions
        for i, test in enumerate(test_cases):
            f.write(f"assert {test['call']} == {test['expected']}, 'Test {i} failed'\n")
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            timeout=5,
            text=True
        )
        if result.returncode == 0:
            return 1.0   # All tests passed
        else:
            return 0.0   # Failed tests
    except Exception:
        return 0.0
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def verify_math_solution(generated_answer: str, correct_answer: float, tolerance: float = 0.001) -> float:
    """Verifies math problem by extracting the final float and comparing to correct answer"""
    try:
        # Find all numbers in the response
        numbers = re.findall(r'-?\d+\.?\d*', generated_answer)
        if not numbers:
            return 0.0
        
        # Take the last number as the final answer
        predicted = float(numbers[-1])
        if abs(predicted - correct_answer) < tolerance:
            return 1.0
        else:
            return 0.0
    except Exception:
        return 0.0

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def main():
    # 1. GPU Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load tokenizer & model
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with PPO value head: {MODEL_NAME}...")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    policy_model.to(device)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    ref_model.to(device)

    # 3. Configure PPO
    print("Setting up PPO config...")
    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        init_kl_coef=0.1,    # Verifiable reward models can use a lower KL penalty
        target_kl=6.0,
        adap_kl_ctrl=True,
    )

    # 4. Prepare Dataset
    dataset_list = []
    for problem in verifiable_problems:
        input_ids = tokenizer(problem["prompt"], return_tensors="pt").input_ids[0]
        dataset_list.append({
            "prompt": problem["prompt"],
            "input_ids": input_ids.tolist(),
            "type": problem["type"],
            "test_cases": problem.get("test_cases", []),
            "correct_answer": problem.get("correct_answer", 0.0)
        })
    
    dataset = Dataset.from_list(dataset_list)

    # 5. Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # 6. RLVR Loop
    print("🚀 Starting RLVR training loop...")
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
    }

    for epoch in range(1):
        for batch in ppo_trainer.dataloader:
            query_tensors = [torch.tensor(q).to(device) for q in batch["input_ids"]]
            
            # Generate answers using current policy
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze(0)[len(query):])
            
            # Evaluate using verifiers
            rewards = []
            for i, response in enumerate(response_tensors):
                generated_text = tokenizer.decode(response, skip_special_tokens=True)
                p_type = batch["type"][i]
                
                # Check correctness
                if p_type == "code":
                    reward = verify_code_solution(generated_text, batch["test_cases"][i])
                elif p_type == "math":
                    reward = verify_math_solution(generated_text, batch["correct_answer"][i])
                else:
                    reward = 0.0
                    
                rewards.append(torch.tensor(reward))

            # Optimize policy
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            for i, q in enumerate(batch["prompt"]):
                text_ans = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
                print(f"Problem: {q}")
                print(f"Generated: {text_ans}")
                print(f"Reward: {rewards[i].item()} | Passed: {rewards[i].item() == 1.0}")
            print(f"Mean reward: {stats['ppo/mean_reward']:.4f} | KL: {stats['objective/kl']:.4f}\n")

    # 7. Save model
    print(f"Saving RLVR aligned model to {OUTPUT_DIR}...")
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done! Next step: Run 14_rler_agent.py")

if __name__ == "__main__":
    main()
