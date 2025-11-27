#!/usr/bin/env python3
"""
PPO training using the proven implementation
"""

import sys
import os

# Adjust sys.path to include src directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import time
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from models.ppo.ppo_agent import PPOAgent, ACTIONS


def train_ppo(total_timesteps=100000, rollout_steps=2048, update_epochs=8, minibatch_size=32, save_path="models/ppo/ppo_model.pth"):
    """Training function using the PPO implementation"""
    
    # Use the original gym environment directly (not our wrapper) to avoid action space issues
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()

    # Get observation shape in CHW format
    obs_shape = (3, obs.shape[0], obs.shape[1])  # (channels, height, width)
    agent = PPOAgent(obs_shape=obs_shape)
    device = agent.device

    # Rollout buffers
    obs_buffer, actions_buffer, logprobs_buffer, rewards_buffer, dones_buffer, values_buffer = [], [], [], [], [], []

    timestep = 0
    episode_rewards = deque(maxlen=100)
    ep_reward = 0
    start_time = time.time()

    print("🚀 Starting PPO Training")
    print(f"Device: {device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {len(ACTIONS)}")
    print(f"Action space: {env.action_space}")
    print(f"Rollout steps: {rollout_steps}")
    print("=" * 60)

    while timestep < total_timesteps:
        # Convert observation to CHW format
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)

        for step in range(rollout_steps):
            action_id, logp, value = agent.act(obs_chw)
            action = ACTIONS[action_id]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            obs_buffer.append(obs_chw)
            actions_buffer.append(action_id)
            logprobs_buffer.append(logp)
            rewards_buffer.append(reward)
            dones_buffer.append(float(done))
            values_buffer.append(value)

            ep_reward += reward
            timestep += 1
            obs = next_obs
            obs_chw = np.transpose(next_obs, (2, 0, 1)).astype(np.float32)

            if done:
                episode_rewards.append(ep_reward)
                print(f"Episode finished | Reward: {ep_reward:.2f} | Timestep: {timestep}")
                ep_reward = 0
                obs, info = env.reset()

            if timestep >= total_timesteps:
                break

        # Compute last value for GAE
        obs_tensor = torch.tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = agent.net(obs_tensor)
            last_value = last_value.item()

        # Convert to numpy arrays
        rewards_np = np.array(rewards_buffer, dtype=np.float32)
        dones_np = np.array(dones_buffer, dtype=np.float32)
        values_np = np.array(values_buffer + [last_value], dtype=np.float32)

        # Compute GAE
        advantages, returns = agent.compute_gae(rewards_np, dones_np, values_np[:-1], last_value)

        # PPO update
        update_loss = agent.update(
            batch_obs=np.array(obs_buffer, dtype=np.float32),
            batch_actions=np.array(actions_buffer, dtype=np.int64),
            batch_logprobs=np.array(logprobs_buffer, dtype=np.float32),
            batch_returns=returns,
            batch_advantages=advantages,
            epochs=update_epochs,
            minibatch_size=minibatch_size
        )

        # Clear rollout buffers
        obs_buffer.clear()
        actions_buffer.clear()
        logprobs_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        values_buffer.clear()

        # Save model periodically
        if timestep % 50000 == 0:
            agent.save(save_path)
            print(f"💾 Model saved at step {timestep}")

        # Print progress
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        fps = timestep / (time.time() - start_time)
        
        print(f"Step {timestep:6d}/{total_timesteps} | "
              f"AvgReward: {avg_reward:7.2f} | "
              f"Update Loss: {update_loss:7.3f} | "
              f"FPS: {fps:5.1f}")

        # Early success detection
        if avg_reward > 80 and timestep > 10000:
            print("🎉 Early success! Model is learning to drive!")
            agent.save(save_path.replace('.pth', '_success.pth'))
            break

    # Final save
    agent.save(save_path)
    env.close()
    print(f"✅ Training complete! Model saved to: {save_path}")
    return agent


def evaluate_ppo_model(model_path, n_episodes=5, render=True):
    """Evaluate the trained PPO model"""
    env = gym.make("CarRacing-v2", render_mode='human' if render else None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = PPOAgent(obs_shape=obs_shape)
    agent.load(model_path)
    
    print(f"🤖 Evaluating PPO model: {model_path}")
    
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
    
    print(f"\n📊 Evaluation Results:")
    print(f"Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Min/Max: {np.min(eval_rewards):.2f}/{np.max(eval_rewards):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, help="Evaluate model instead of training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_ppo_model(args.eval)
    else:
        train_ppo(total_timesteps=args.timesteps)