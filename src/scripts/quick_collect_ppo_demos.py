#!/usr/bin/env python3
"""
Quickly collect demonstrations from trained PPO model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pickle
from src.models.ppo.ppo_agent import PPOAgent, ACTIONS


def quick_collect_ppo_demos(model_path, n_episodes=30, output_path="data/expert_data/quick_ppo_demos.pkl"):
    """Quickly collect demonstrations from PPO model"""
    
    print(f"🚗 Collecting {n_episodes} episodes from PPO model: {model_path}")
    
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    # Load PPO agent
    agent = PPOAgent(obs_shape=obs_shape)
    agent.load(model_path)
    
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    total_samples = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        episode_reward = 0
        step = 0
        
        while step < 1000:  # Max steps per episode
            # Convert to CHW format
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            
            # Get action from PPO
            action_id, _, _ = agent.act(obs_chw)
            action = ACTIONS[action_id]
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store data
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action_id)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Add episode to dataset
        dataset['observations'].extend(episode_data['observations'])
        dataset['actions'].extend(episode_data['actions'])
        dataset['rewards'].extend(episode_data['rewards'])
        dataset['dones'].extend(episode_data['dones'])
        
        total_samples += len(episode_data['observations'])
        print(f"Episode {episode + 1}: {step} steps, Reward: {episode_reward:.2f}, Total samples: {total_samples}")
    
    # Save dataset
    os.makedirs('data/expert_data', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    env.close()
    
    print(f"✅ Collected {total_samples} samples from {n_episodes} episodes")
    print(f"✅ Saved to: {output_path}")
    
    # Print statistics
    actions = np.array(dataset['actions'])
    unique, counts = np.unique(actions, return_counts=True)
    print("Action distribution:")
    for action_id, count in zip(unique, counts):
        action_names = ["Nothing", "Left", "Right", "Gas", "Brake"]
        print(f"  {action_names[action_id]}: {count} ({count/total_samples*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to collect")
    parser.add_argument("--output", type=str, default="data/expert_data/ppo_demos.pkl", help="Output file path")
    
    args = parser.parse_args()
    
    quick_collect_ppo_demos(args.model, args.episodes, args.output)