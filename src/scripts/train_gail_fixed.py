#!/usr/bin/env python3
"""
Improved GAIL with better reward shaping and early stopping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import gymnasium as gym
from models.ppo.ppo_agent import PPOAgent
from models.gail.gail_agent_fixed import FixedGAILAgent
from models.ppo.ppo_agent import ACTIONS


def train_gail_improved(expert_data_path: str, total_epochs: int = 100, save_path: str = "models/gail/gail_improved.pth"):
    """Improved GAIL with better reward handling"""
    
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
    
    # Initialize
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = FixedGAILAgent(obs_shape=obs_shape)
    
    print("🚀 Starting IMPROVED GAIL Training")
    print("Key improvements:")
    print("  • Early stopping for bad behavior")
    print("  • Action diversity enforcement")
    print("  • Progressive difficulty")
    print("=" * 60)
    
    consecutive_bad_epochs = 0
    best_reward = -float('inf')
    
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
        
        # Calculate action diversity
        action_counts = np.bincount(policy_actions, minlength=5)
        action_diversity = np.sum(action_counts > 0) / 5.0  # 0-1 score
        
        mean_env_reward = np.mean(policy_rewards_env)
        
        # EARLY STOPPING FOR BAD BEHAVIOR
        if mean_env_reward < -0.04 and action_diversity < 0.6:
            consecutive_bad_epochs += 1
            print(f"⚠️  Bad behavior detected: low reward & diversity (consecutive: {consecutive_bad_epochs})")
            
            if consecutive_bad_epochs >= 5:
                print("🛑 RESTARTING TRAINING - Policy stuck in bad behavior")
                # Reset policy but keep discriminator
                obs_shape = (3, 96, 96)
                agent.policy = PPOAgent(obs_shape=obs_shape)
                consecutive_bad_epochs = 0
                continue
        else:
            consecutive_bad_epochs = 0
            if mean_env_reward > best_reward:
                best_reward = mean_env_reward
        
        # Sample batches - use smaller batches for more updates
        batch_size = min(256, len(expert_obs), len(policy_obs))
        
        # Train discriminator MULTIPLE times if policy is bad
        disc_updates = 3 if mean_env_reward < -0.03 else 1
        
        disc_stats_list = []
        for _ in range(disc_updates):
            expert_idx = np.random.choice(len(expert_obs), batch_size, replace=False)
            policy_idx = np.random.choice(len(policy_obs), batch_size, replace=False)
            
            disc_stats = agent.update_discriminator(
                expert_obs[expert_idx], expert_actions[expert_idx],
                policy_obs[policy_idx], policy_actions[policy_idx]
            )
            disc_stats_list.append(disc_stats)
        
        # Average discriminator stats
        disc_stats = {
            key: np.mean([s[key] for s in disc_stats_list])
            for key in disc_stats_list[0].keys()
        }
        
        # Compute GAIL rewards with ACTION DIVERSITY BONUS
        policy_obs_tensor = torch.tensor(policy_obs, dtype=torch.float32, device=agent.device)
        policy_actions_tensor = torch.tensor(policy_actions, dtype=torch.long, device=agent.device)
        gail_rewards = agent.compute_gail_reward(policy_obs_tensor, policy_actions_tensor)
        gail_rewards = gail_rewards.cpu().numpy()
        
        # ADD DIVERSITY BONUS to encourage exploration
        diversity_bonus = action_diversity * 0.1  # Small bonus for diverse actions
        gail_rewards = gail_rewards + diversity_bonus
        
        # Update policy with FEWER epochs to prevent overfitting to bad behavior
        policy_loss = agent.update_policy(
            observations=policy_obs,
            actions=policy_actions,
            logprobs_old=policy_logprobs,
            values_old=policy_values,
            rewards=gail_rewards,
            dones=policy_dones,
            epochs=1,  # Only 1 epoch to prevent overfitting
            minibatch_size=32
        )
        
        # Print progress with more details
        print(f"Epoch {epoch:3d}: Disc: {disc_stats['discriminator_loss']:5.3f}, "
              f"Policy: {policy_loss:6.3f}, Env: {mean_env_reward:7.2f}, "
              f"Acc: {disc_stats['total_accuracy']:.3f}, Div: {action_diversity:.3f}")
        
        # Early success
        if mean_env_reward > 50:
            print(f"🎉 EARLY SUCCESS! Reward: {mean_env_reward:.2f}")
            agent.save(save_path)
            break
        
        # Save periodically
        if epoch % 10 == 0:
            agent.save(save_path)
            print(f"💾 Saved to {save_path}")
    
    agent.save(save_path)
    env.close()
    print(f"✅ Improved GAIL training complete! Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    train_gail_improved("data/expert_data/ppo_demos.pkl", total_epochs=100)