#!/usr/bin/env python3
"""
Quick test of the optimized PPO implementation
"""

import sys
import os

# From tests directory, go up to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gymnasium as gym
import numpy as np
from src.models.ppo.ppo_agent import PPOAgent


def test_ppo_short():
    """Quick test with minimal timesteps"""
    print("🧪 Testing Optimized PPO Implementation...")
    
    # Use gym environment directly
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    
    # Get observation shape in CHW format
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    # Initialize PPO agent with small parameters for quick test
    agent = PPOAgent(
        obs_shape=obs_shape,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    print(f"Device: {agent.device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: 5")
    print(f"Model parameters: {sum(p.numel() for p in agent.net.parameters()):,}")
    
    # Quick training test
    print("Starting short training test...")
    
    # Training parameters for quick test
    total_timesteps = 2000
    rollout_steps = 128  # Smaller buffer for quick test
    update_epochs = 2
    minibatch_size = 32
    
    # Rollout buffers
    obs_buffer, actions_buffer, logprobs_buffer, rewards_buffer, dones_buffer, values_buffer = [], [], [], [], [], []
    
    timestep = 0
    episode_rewards = []
    ep_reward = 0
    
    while timestep < total_timesteps:
        # Convert observation to CHW format
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        
        for step in range(rollout_steps):
            action_id, logp, value = agent.act(obs_chw)
            from src.models.ppo.ppo_agent import ACTIONS
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
                print(f"  Episode finished | Reward: {ep_reward:.2f} | Timestep: {timestep}")
                ep_reward = 0
                obs, info = env.reset()
            
            if timestep >= total_timesteps:
                break
        
        # Compute last value for GAE
        obs_tensor = torch.tensor(obs_chw, dtype=torch.float32, device=agent.device).unsqueeze(0)
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
        
        # Print progress
        avg_reward = np.mean(episode_rewards[-5:]) if episode_rewards else 0
        print(f"Step {timestep:4d}/{total_timesteps} | "
              f"AvgReward: {avg_reward:6.2f} | "
              f"Update Loss: {update_loss:7.3f}")
    
    env.close()
    print("✅ PPO test completed!")
    print(f"Tested {len(episode_rewards)} episodes")
    if episode_rewards:
        print(f"Final rewards: {episode_rewards[-3:]}")
        print(f"Best reward: {max(episode_rewards):.2f}")


def test_ppo_act_only():
    """Test just the action selection without training"""
    print("🧪 Testing PPO Action Selection...")
    
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    agent = PPOAgent(obs_shape=obs_shape)
    
    print("Testing action selection for 10 steps...")
    for step in range(10):
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        action_id, logprob, value = agent.act(obs_chw)
        
        print(f"Step {step + 1}: Action={action_id}, LogProb={logprob:.3f}, Value={value:.3f}")
        
        from src.models.ppo.ppo_agent import ACTIONS
        action = ACTIONS[action_id]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            print("  Environment reset")
    
    env.close()
    print("✅ Action selection test completed!")


def test_model_save_load():
    """Test model saving and loading functionality"""
    print("🧪 Testing Model Save/Load...")
    
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    
    # Create agent and get initial action
    agent1 = PPOAgent(obs_shape=obs_shape)
    obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
    action1, logprob1, value1 = agent1.act(obs_chw)
    
    # Save model
    save_path = "models/ppo/test_model.pth"
    agent1.save(save_path)
    print(f"✅ Model saved to: {save_path}")
    
    # Create new agent and load model
    agent2 = PPOAgent(obs_shape=obs_shape)
    agent2.load(save_path)
    print(f"✅ Model loaded from: {save_path}")
    
    # Test if actions are the same
    action2, logprob2, value2 = agent2.act(obs_chw)
    
    print(f"Original agent - Action: {action1}, LogProb: {logprob1:.3f}, Value: {value1:.3f}")
    print(f"Loaded agent   - Action: {action2}, LogProb: {logprob2:.3f}, Value: {value2:.3f}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"✅ Test file cleaned up: {save_path}")
    
    env.close()
    print("✅ Save/load test completed!")


if __name__ == "__main__":
    import torch
    
    print("=" * 60)
    print("PPO Implementation Test Suite")
    print("=" * 60)
    
    # Run tests
    test_ppo_act_only()
    print()
    
    test_model_save_load() 
    print()
    
    test_ppo_short()
    
    print("=" * 60)
    print("All tests completed! 🎉")
    print("=" * 60)