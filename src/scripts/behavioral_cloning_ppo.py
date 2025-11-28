#!/usr/bin/env python3
"""
Behavioral Cloning - Supervised learning from expert demonstrations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from models.ppo.ppo_agent import PPOAgent, ActorCritic


def train_behavioral_cloning(expert_data_path: str, epochs: int = 50, save_path: str = "models/bc/ppo_bc_tuned.pth"):
    """Train policy via behavioral cloning (supervised learning)"""
    
    print("🎯 Starting Behavioral Cloning")
    
    # Load expert data
    import pickle
    with open(expert_data_path, 'rb') as f:
        expert_data = pickle.load(f)
    
    expert_obs = np.array(expert_data['observations'], dtype=np.float32)
    expert_actions = np.array(expert_data['actions'], dtype=np.int64)
    
    # Convert to CHW format
    if expert_obs[0].shape == (96, 96, 3):
        expert_obs = np.array([np.transpose(obs, (2, 0, 1)) for obs in expert_obs])
    
    print(f"✅ Loaded {len(expert_obs)} expert samples")
    
    # Initialize policy
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    policy = PPOAgent(obs_shape=obs_shape)
    
    # Convert to PyTorch datasets
    expert_obs_tensor = torch.tensor(expert_obs, dtype=torch.float32)
    expert_actions_tensor = torch.tensor(expert_actions, dtype=torch.long)
    
    dataset = TensorDataset(expert_obs_tensor, expert_actions_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy.net.parameters(), lr=1e-4)
    
    policy.net.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(policy.device)
            batch_actions = batch_actions.to(policy.device)
            
            # Forward pass
            logits, _ = policy.net(batch_obs)
            loss = criterion(logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_actions).sum().item()
            total += batch_actions.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        print(f"BC Epoch {epoch+1:3d}/{epochs}: Loss: {avg_loss:.4f}, Acc: {accuracy:.3f}")
        
        # Test the policy occasionally
        if (epoch + 1) % 10 == 0:
            test_reward = evaluate_bc_policy(policy, env)
            print(f"  Test Reward: {test_reward:.2f}")
    
    # Save the BC model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    policy.save(save_path)
    env.close()
    
    print(f"✅ Behavioral Cloning complete! Saved to {save_path}")
    return policy


def evaluate_bc_policy(policy, env, n_episodes=3):
    """Evaluate BC policy"""
    total_reward = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, _, _ = policy.act(obs_chw)
            
            from models.ppo.ppo_agent import ACTIONS
            obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_episodes


def bc_then_gail(expert_data_path: str):
    """Complete pipeline: BC then GAIL"""
    
    # Step 1: Behavioral Cloning
    bc_model_path = "models/bc/ppo_bc_tuned.pth"
    print("=" * 60)
    print("STEP 1: Behavioral Cloning")
    print("=" * 60)
    
    bc_policy = train_behavioral_cloning(expert_data_path, epochs=50, save_path=bc_model_path)
    
    # Step 2: Evaluate BC
    print("\n" + "=" * 60)
    print("STEP 2: Evaluate BC Policy")
    print("=" * 60)
    
    env = gym.make("CarRacing-v2", render_mode=None)
    bc_reward = evaluate_bc_policy(bc_policy, env, n_episodes=5)
    env.close()
    
    print(f"BC Policy Average Reward: {bc_reward:.2f}")
    
    if bc_reward > 0:
        print("✅ BC successful! Proceeding to GAIL...")
        
        # Step 3: GAIL fine-tuning
        print("\n" + "=" * 60)
        print("STEP 3: GAIL Fine-tuning")
        print("=" * 60)
        
        # We need to modify the GAIL script to start from BC policy
        # For now, just notify
        print("Run GAIL training starting from BC policy:")
        print(f"python src/scripts/train_gail.py --expert-data {expert_data_path} --bc-init {bc_model_path}")
    else:
        print("❌ BC failed. Need different approach.")


if __name__ == "__main__":
    bc_then_gail("data/expert_data/ppo_demos.pkl")