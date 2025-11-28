#!/usr/bin/env python3
"""
GAIL training starting from Behavioral Cloning model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import gymnasium as gym
from models.gail.gail_agent_fixed import FixedGAILAgent
from models.ppo.ppo_agent import PPOAgent, ACTIONS


def train_gail_with_bc(expert_data_path: str, bc_model_path: str, total_epochs: int = 50, save_path: str = "models/gail/gail_bc_tuned.pth"):
    """GAIL training starting from BC model"""
    
    print("🚀 Starting GAIL with BC Initialization")
    print(f"BC Model: {bc_model_path}")
    
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
    
    # Initialize environment and agent
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    # Create GAIL agent with BC-initialized policy
    agent = FixedGAILAgent(obs_shape=obs_shape)
    
    # Load BC model into the policy
    print(f"🔄 Loading BC model: {bc_model_path}")
    agent.policy.load(bc_model_path)
    
    # Test initial performance
    initial_reward = evaluate_policy(agent.policy, env)
    print(f"📊 Initial BC Performance: {initial_reward:.2f}")
    
    print("=" * 60)
    print("Starting GAIL Fine-tuning...")
    print("=" * 60)
    
    best_reward = initial_reward
    
    for epoch in range(1, total_epochs + 1):
        # Collect policy data
        policy_obs, policy_actions, policy_logprobs, policy_values, policy_rewards, policy_dones = [], [], [], [], [], []
        
        obs, info = env.reset()
        steps = 0
        
        while steps < 1024:
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, logprob, value = agent.policy.act(obs_chw)
            action = ACTIONS[action_id]
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            policy_obs.append(obs_chw)
            policy_actions.append(action_id)
            policy_logprobs.append(logprob)
            policy_values.append(value)
            policy_rewards.append(reward)
            policy_dones.append(float(done))
            
            steps += 1
            obs = next_obs
            
            if done:
                obs, info = env.reset()
        
        # Convert to arrays
        policy_obs = np.array(policy_obs, dtype=np.float32)
        policy_actions = np.array(policy_actions, dtype=np.int64)
        policy_logprobs = np.array(policy_logprobs, dtype=np.float32)
        policy_values = np.array(policy_values, dtype=np.float32)
        policy_rewards_env = np.array(policy_rewards, dtype=np.float32)
        policy_dones = np.array(policy_dones, dtype=np.float32)
        
        # Sample batches
        batch_size = min(512, len(expert_obs), len(policy_obs))
        expert_idx = np.random.choice(len(expert_obs), batch_size, replace=False)
        policy_idx = np.random.choice(len(policy_obs), batch_size, replace=False)
        
        # Train discriminator
        disc_stats = agent.update_discriminator(
            expert_obs[expert_idx], expert_actions[expert_idx],
            policy_obs[policy_idx], policy_actions[policy_idx]
        )
        
        # Compute GAIL rewards
        policy_obs_tensor = torch.tensor(policy_obs, dtype=torch.float32, device=agent.device)
        policy_actions_tensor = torch.tensor(policy_actions, dtype=torch.long, device=agent.device)
        gail_rewards = agent.compute_gail_reward(policy_obs_tensor, policy_actions_tensor)
        gail_rewards = gail_rewards.cpu().numpy()
        
        # Update policy (conservative - only 1 epoch to not break BC knowledge)
        policy_loss = agent.update_policy(
            observations=policy_obs,
            actions=policy_actions,
            logprobs_old=policy_logprobs,
            values_old=policy_values,
            rewards=gail_rewards,
            dones=policy_dones,
            epochs=1,  # Conservative to preserve BC knowledge
            minibatch_size=32
        )
        
        mean_env_reward = np.mean(policy_rewards_env)
        
        # Track best reward
        if mean_env_reward > best_reward:
            best_reward = mean_env_reward
        
        print(f"Epoch {epoch:3d}: Disc: {disc_stats['discriminator_loss']:5.3f}, "
              f"Policy: {policy_loss:6.3f}, Env: {mean_env_reward:7.2f}, "
              f"Acc: {disc_stats['total_accuracy']:.3f}, Best: {best_reward:7.2f}")
        
        # Save if performance improves or periodically
        if mean_env_reward > best_reward * 0.95 or epoch % 10 == 0:
            agent.save(save_path)
            print(f"💾 Saved to {save_path}")
        
        # Early stopping if performance drops significantly
        if epoch > 10 and mean_env_reward < best_reward * 0.7:
            print(f"🛑 Performance dropped significantly. Stopping early.")
            break
    
    # Final evaluation
    final_reward = evaluate_policy(agent.policy, env)
    improvement = final_reward - initial_reward
    
    print("\n" + "=" * 60)
    print("GAIL Fine-tuning Complete!")
    print(f"Initial Reward: {initial_reward:.2f}")
    print(f"Final Reward: {final_reward:.2f}")
    print(f"Improvement: {improvement:+.2f}")
    print("=" * 60)
    
    agent.save(save_path)
    env.close()
    
    return agent


def evaluate_policy(policy, env, n_episodes=3):
    """Evaluate policy performance"""
    total_reward = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, _, _ = policy.act(obs_chw)
            obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-data", type=str, required=True, help="Path to expert data")
    parser.add_argument("--bc-model", type=str, required=True, help="Path to BC model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--save-path", type=str, default="models/gail/gail_bc_tuned.pth", help="Output path")
    
    args = parser.parse_args()
    
    train_gail_with_bc(
        expert_data_path=args.expert_data,
        bc_model_path=args.bc_model,
        total_epochs=args.epochs,
        save_path=args.save_path
    )