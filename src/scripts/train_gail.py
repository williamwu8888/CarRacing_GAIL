#!/usr/bin/env python3
"""
GAIL training script
"""

import sys
import os

# Adjust sys.path to include src directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import time
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from models.gail.gail_agent import GAILAgent
from models.ppo.ppo_agent import ACTIONS


def load_expert_data(data_path: str):
    """Load expert demonstrations from file"""
    import pickle
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert data file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded expert data from {data_path}")
    print(f"   Observations: {len(data['observations'])}")
    print(f"   Actions: {len(data['actions'])}")
    
    return data


def train_gail(
    expert_data_path: str,
    total_epochs: int = 100,
    discriminator_steps: int = 5,
    policy_steps: int = 2048,
    save_path: str = "models/gail/gail_model.pth",
    render: bool = False
):
    """Train GAIL on expert demonstrations"""
    
    # Load expert data
    expert_data = load_expert_data(expert_data_path)
    expert_obs = np.array(expert_data['observations'], dtype=np.float32)
    expert_actions = np.array(expert_data['actions'], dtype=np.int64)
    
    # Convert expert observations to CHW format if needed
    if expert_obs[0].shape == (96, 96, 3):  # HWC format
        expert_obs = np.array([np.transpose(obs, (2, 0, 1)) for obs in expert_obs])
    
    print(f"Expert observations shape: {expert_obs.shape}")
    print(f"Expert actions shape: {expert_actions.shape}")
    
    # Initialize environment and agent
    env = gym.make("CarRacing-v2", render_mode='human' if render else None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = GAILAgent(obs_shape=obs_shape)
    
    print("🚀 Starting GAIL Training")
    print(f"Device: {agent.device}")
    print(f"Expert samples: {len(expert_obs)}")
    print(f"Total epochs: {total_epochs}")
    print(f"Discriminator steps per epoch: {discriminator_steps}")
    print("=" * 60)
    
    # Training statistics
    episode_rewards = deque(maxlen=20)
    start_time = time.time()
    
    for epoch in range(1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} ---")
        
        # Collect policy trajectories
        policy_obs, policy_actions, policy_logprobs, policy_values, policy_rewards, policy_dones = [], [], [], [], [], []
        
        obs, info = env.reset()
        episode_reward = 0
        steps_collected = 0
        
        while steps_collected < policy_steps:
            # Convert observation to CHW format
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            
            # Get action from policy
            action_id, logprob, value = agent.policy.act(obs_chw)
            action = ACTIONS[action_id]
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store policy data
            policy_obs.append(obs_chw)
            policy_actions.append(action_id)
            policy_logprobs.append(logprob)
            policy_values.append(value)
            policy_rewards.append(reward)
            policy_dones.append(float(done))
            
            episode_reward += reward
            steps_collected += 1
            obs = next_obs
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, info = env.reset()
        
        print(f"Collected {steps_collected} policy steps")
        print(f"Recent episode rewards: {[f'{r:.2f}' for r in list(episode_rewards)[-3:]]}")
        
        # Convert to numpy arrays
        policy_obs = np.array(policy_obs, dtype=np.float32)
        policy_actions = np.array(policy_actions, dtype=np.int64)
        policy_logprobs = np.array(policy_logprobs, dtype=np.float32)
        policy_values = np.array(policy_values, dtype=np.float32)
        policy_rewards_env = np.array(policy_rewards, dtype=np.float32)  # Environment rewards
        policy_dones = np.array(policy_dones, dtype=np.float32)
        
        # Sample mini-batches from expert data for discriminator training
        expert_indices = np.random.choice(len(expert_obs), size=min(1024, len(expert_obs)), replace=False)
        policy_indices = np.random.choice(len(policy_obs), size=min(1024, len(policy_obs)), replace=False)
        
        expert_obs_batch = expert_obs[expert_indices]
        expert_actions_batch = expert_actions[expert_indices]
        policy_obs_batch = policy_obs[policy_indices]
        policy_actions_batch = policy_actions[policy_indices]
        
        # Train discriminator
        discriminator_stats = []
        for disc_step in range(discriminator_steps):
            stats = agent.update_discriminator(
                expert_obs_batch,
                expert_actions_batch,
                policy_obs_batch,
                policy_actions_batch
            )
            discriminator_stats.append(stats)
        
        # Average discriminator stats
        avg_disc_stats = {
            key: np.mean([s[key] for s in discriminator_stats])
            for key in discriminator_stats[0].keys()
        }
        
        # Compute GAIL rewards for policy update
        policy_obs_tensor = torch.tensor(policy_obs, dtype=torch.float32, device=agent.device)
        policy_actions_tensor = torch.tensor(policy_actions, dtype=torch.long, device=agent.device)
        gail_rewards = agent.compute_gail_reward(policy_obs_tensor, policy_actions_tensor)
        gail_rewards = gail_rewards.cpu().numpy()
        
        print(f"GAIL rewards - Mean: {gail_rewards.mean():.3f}, Std: {gail_rewards.std():.3f}")
        
        # Update policy using GAIL rewards
        policy_loss = agent.update_policy(
            observations=policy_obs,
            actions=policy_actions,
            logprobs_old=policy_logprobs,
            values_old=policy_values,
            rewards=gail_rewards,
            dones=policy_dones,
            epochs=4,
            minibatch_size=64
        )
        
        # Print epoch statistics
        fps = steps_collected / (time.time() - start_time)
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Policy Loss: {policy_loss:.4f}")
        print(f"  Discriminator Loss: {avg_disc_stats['discriminator_loss']:.4f}")
        print(f"  Expert Accuracy: {avg_disc_stats['expert_accuracy']:.3f}")
        print(f"  Policy Accuracy: {avg_disc_stats['policy_accuracy']:.3f}")
        print(f"  Mean Reward: {mean_reward:.2f}")
        print(f"  FPS: {fps:.1f}")
        
        # Save model periodically
        if epoch % 10 == 0:
            agent.save(save_path)
            print(f"💾 Model saved to: {save_path}")
    
    # Final save
    agent.save(save_path)
    env.close()
    print(f"✅ GAIL training complete! Model saved to: {save_path}")


def evaluate_gail(model_path: str, n_episodes: int = 5, render: bool = True):
    """Evaluate trained GAIL model"""
    env = gym.make("CarRacing-v2", render_mode='human' if render else None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = GAILAgent(obs_shape=obs_shape)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    agent.load(model_path)
    print(f"🤖 Evaluating GAIL model: {model_path}")
    
    eval_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, _, _ = agent.act(obs_chw)
            obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
            done = terminated or truncated
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    print(f"\n📊 GAIL Evaluation Results:")
    print(f"Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Min/Max: {np.min(eval_rewards):.2f}/{np.max(eval_rewards):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-data", type=str, required=True, help="Path to expert data file")
    parser.add_argument("--eval", type=str, help="Evaluate model instead of training")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--save-path", type=str, default="models/gail/gail_model.pth", help="Model save path")
    parser.add_argument("--render", action="store_true", help="Render during training")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_gail(args.eval)
    else:
        train_gail(
            expert_data_path=args.expert_data,
            total_epochs=args.epochs,
            save_path=args.save_path,
            render=args.render
        )