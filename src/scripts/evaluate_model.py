#!/usr/bin/env python3
"""
Comprehensive model evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import time
from models.ppo.ppo_agent import PPOAgent, ACTIONS


def evaluate_model_comprehensive(model_path, n_episodes=10, render=False):
    """Comprehensive evaluation of trained model"""
    
    env = gym.make("CarRacing-v2", render_mode='human' if render else None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = PPOAgent(obs_shape=obs_shape)
    agent.load(model_path)
    
    print(f"🧪 Comprehensive Evaluation: {model_path}")
    print(f"Episodes: {n_episodes}, Rendering: {render}")
    print("=" * 50)
    
    all_rewards = []
    all_steps = []
    action_distribution = {i: 0 for i in range(len(ACTIONS))}
    action_names = ["Nothing", "Left", "Right", "Gas", "Brake"]
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Max steps per episode
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, _, _ = agent.act(obs_chw)
            
            # Track action distribution
            action_distribution[action_id] += 1
            
            obs, reward, terminated, truncated, info = env.step(ACTIONS[action_id])
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
        
        all_rewards.append(episode_reward)
        all_steps.append(steps)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:7.2f}, Steps = {steps:3d}")
    
    env.close()
    
    # Print comprehensive results
    print("\n📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Best Reward: {np.max(all_rewards):.2f}")
    print(f"Worst Reward: {np.min(all_rewards):.2f}")
    print(f"Average Steps: {np.mean(all_steps):.1f}")
    print(f"Success Rate (>0 reward): {sum(r > 0 for r in all_rewards) / n_episodes * 100:.1f}%")
    
    print(f"\n🎯 Action Distribution:")
    total_actions = sum(action_distribution.values())
    for action_id in range(len(ACTIONS)):
        count = action_distribution[action_id]
        percentage = count / total_actions * 100
        print(f"  {action_names[action_id]:8}: {count:5} ({percentage:5.1f}%)")
    
    # Performance assessment
    mean_reward = np.mean(all_rewards)
    if mean_reward > 100:
        assessment = "🎉 EXCELLENT - Professional driving!"
    elif mean_reward > 50:
        assessment = "✅ VERY GOOD - Solid driving skills"
    elif mean_reward > 20:
        assessment = "⚠️  DECENT - Can complete laps"
    elif mean_reward > 0:
        assessment = "🔧 BASIC - Learning but needs improvement"
    else:
        assessment = "❌ POOR - Not learning to drive"
    
    print(f"\n📈 Performance Assessment: {assessment}")
    
    return all_rewards, action_distribution


def quick_evaluate(model_path, n_episodes=5):
    """Quick evaluation without rendering"""
    print(f"🚗 Quick Evaluation: {model_path}")
    
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = PPOAgent(obs_shape=obs_shape)
    agent.load(model_path)
    
    rewards = []
    
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
        
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward:.2f}")
    
    env.close()
    
    mean_reward = np.mean(rewards)
    print(f"\n📊 Average Reward: {mean_reward:.2f}")
    
    if mean_reward > 50:
        print("✅ Model is ready for demonstration collection!")
    else:
        print("⚠️  Model may need more training")
    
    return mean_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation without details")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_evaluate(args.model, args.episodes)
    else:
        evaluate_model_comprehensive(args.model, args.episodes, args.render)