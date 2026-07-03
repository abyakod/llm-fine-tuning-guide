#!/usr/bin/env python3
"""
14_rler_agent.py — Reinforcement Learning with Execution Rewards (RLER)
Part 4 of The Complete LLM Fine-Tuning Guide

This script demonstrates how to train an AI agent to perform multi-step actions
in an environment (like a web browser or API tool) and align its decision-making
using the final execution success signal (execution rewards).

Usage: python scripts/14_rler_agent.py
Prerequisites: GPU required
"""

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./rler-aligned-agent"

# Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
MINI_BATCH_SIZE = 1
PPO_EPOCHS = 1

class MockEnvironment:
    """
    A simple mockup environment representing a simulated flight-booking portal.
    """
    def __init__(self):
        self.task = None
        self.state = "start"
        self.steps_taken = 0
        
    def reset(self, task: str):
        self.task = task
        self.state = "search_page"
        self.steps_taken = 0
        return f"Task: {self.task}. You are on the flight search page. Please enter departure and destination."

    def get_available_actions(self) -> str:
        if self.state == "search_page":
            return "[SEARCH 'NYC' TO 'PARIS', SEARCH 'LA' TO 'SF']"
        elif self.state == "flight_list":
            return "[SELECT FLIGHT 101 ($500), SELECT FLIGHT 202 ($800)]"
        elif self.state == "payment_page":
            return "[CONFIRM BOOKING, CANCEL BOOKING]"
        return "[GO TO HOMEPAGE]"

    def step(self, action: str):
        self.steps_taken += 1
        action_upper = action.upper()
        
        if self.state == "search_page":
            if "SEARCH 'NYC' TO 'PARIS'" in action_upper:
                self.state = "flight_list"
                return "Flight results shown. Flight 101 ($500) or Flight 202 ($800) available.", False, False
            else:
                return "Invalid search destination. Still on search page.", False, False
                
        elif self.state == "flight_list":
            if "SELECT FLIGHT 101" in action_upper:
                self.state = "payment_page"
                return "Flight 101 selected. Please confirm payment.", False, False
            else:
                return "Invalid flight selection. Still on flight list page.", False, False
                
        elif self.state == "payment_page":
            if "CONFIRM BOOKING" in action_upper:
                self.state = "complete"
                return "Booking successful! Confirmation code: TX889", True, True
            else:
                self.state = "cancelled"
                return "Booking cancelled.", True, False
                
        return "Environment in final state.", True, False

class SimpleWebAgent:
    """A minimal RLER agent that maps states to actions using the LLM"""
    def __init__(self, model, tokenizer, environment):
        self.model = model
        self.tokenizer = tokenizer
        self.env = environment
        
    def generate_action(self, observation: str) -> str:
        prompt = f"""### Current State:
{observation}

### Available Actions:
{self.env.get_available_actions()}

### Your Action:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse the action out
        action = generated_text.split("### Your Action:")[-1].strip().split("\n")[0].strip()
        # Fallback if parsing fails or returns empty
        if not action or len(action) < 2:
            # Let's take the first action from available actions as fallback
            available = self.env.get_available_actions()
            action = available.split("[")[-1].split("]")[0].split(",")[0].strip()
        return action

    def run_episode(self, task: str, max_steps: int = 5):
        """Runs a complete action episode and records trajectory"""
        trajectory = []
        observation = self.env.reset(task)
        
        for step in range(max_steps):
            action = self.generate_action(observation)
            next_observation, done, success = self.env.step(action)
            
            trajectory.append({
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
            })
            
            observation = next_observation
            if done:
                reward = 1.0 if success else 0.0
                return trajectory, reward
                
        return trajectory, 0.0

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

    print(f"Loading policy model with PPO value head: {MODEL_NAME}...")
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
        init_kl_coef=0.1,
        target_kl=6.0,
        adap_kl_ctrl=True,
    )

    # Instantiate Environment and Agent
    env = MockEnvironment()
    agent = SimpleWebAgent(policy_model, tokenizer, env)

    # Dummy dataset of task instructions (used for PPOTrainer dataloader setup)
    # RLER constructs query/response pairs dynamically from episode actions
    # We initialize the trainer with a placeholder dataset to build the loop
    placeholder_dataset = Dataset.from_list([
        {"query": "Book flight NYC to Paris", "input_ids": tokenizer("Book flight NYC to Paris").input_ids},
        {"query": "Book flight NYC to Paris", "input_ids": tokenizer("Book flight NYC to Paris").input_ids}
    ])

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=placeholder_dataset,
    )

    # 4. Training Loop (Simulating RLER episodic training)
    print("🚀 Starting RLER agent training...")
    tasks = [
        "Book flight from NYC to Paris",
        "Book flight from NYC to Paris"
    ]

    for episode_num, task in enumerate(tasks):
        print(f"\n--- Episode {episode_num + 1} ---")
        trajectory, final_reward = agent.run_episode(task)
        
        print(f"Episode Reward: {final_reward}")
        print("Trajectory:")
        for idx, step in enumerate(trajectory):
            print(f"  Step {idx+1}: State -> {step['observation'][:40]}... | Action taken: {step['action']}")
            
        # Optimize model parameters on the actions taken (Credit assignment & Reward shaping)
        for i, step in enumerate(trajectory):
            # Simple reward shaping: actions closer to success get more credit
            step_reward = final_reward * (0.9 ** (len(trajectory) - i - 1))
            
            prompt = f"### Current State:\n{step['observation']}\n\n### Available Actions:\n{env.get_available_actions()}\n\n### Your Action:\n"
            query_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
            response_ids = tokenizer(step["action"], return_tensors="pt").input_ids[0].to(device)
            
            # Step trainer parameters
            stats = ppo_trainer.step([query_ids], [response_ids], [torch.tensor(step_reward)])
            print(f"  Step {idx+1} optimized. Loss: {stats.get('ppo/loss/policy', 0):.4f}")

    # 5. Save Model
    print(f"Saving RLER aligned model to {OUTPUT_DIR}...")
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    main()
