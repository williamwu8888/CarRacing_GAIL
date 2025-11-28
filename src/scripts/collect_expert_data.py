#!/usr/bin/env python3
"""
Collect expert demonstrations for GAIL training
"""

import sys
import os

# Adjust sys.path to include src directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import gymnasium as gym
import numpy as np
import pickle
from environments.car_racing_env import CarRacingEnvWrapper
from models.ppo.ppo_agent import PPOAgent, ACTIONS


class ExpertDataCollector:
    def __init__(self):
        self.env = CarRacingEnvWrapper(render_mode='human')
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
    
    def human_control(self):
        """Simple human control for data collection"""
        print("Human Control Mode:")
        print("A/D - Steer Left/Right")
        print("W/S - Gas/Brake") 
        print("Space - Do Nothing")
        print("Q - Quit")
        
        from pynput import keyboard
        
        current_action = 0  # Do nothing
        
        def on_press(key):
            nonlocal current_action
            try:
                if key.char == 'a':
                    current_action = 1  # Steer left
                elif key.char == 'd':
                    current_action = 2  # Steer right
                elif key.char == 'w':
                    current_action = 3  # Gas
                elif key.char == 's':
                    current_action = 4  # Brake
            except AttributeError:
                if key == keyboard.Key.space:
                    current_action = 0  # Do nothing
        
        def on_release(key):
            nonlocal current_action
            if key == keyboard.KeyCode.from_char('q'):
                return False  # Stop listener
            current_action = 0  # Reset to do nothing when key released
        
        return current_action, keyboard.Listener(on_press=on_press, on_release=on_release)
    
    def collect_human_demonstrations(self, n_episodes=10):
        """Collect human demonstrations"""
        print("🚗 Collecting Human Demonstrations for GAIL...")
        
        current_action, listener = self.human_control()
        listener.start()
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            total_reward = 0
            step = 0
            
            while step < 1000:  # Max steps per episode
                # Take action based on human input
                action_id = current_action
                action = ACTIONS[action_id]  # Convert to continuous action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store data (store action_id for discrete learning)
                episode_data['observations'].append(obs)
                episode_data['actions'].append(action_id)  # Store discrete action ID
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                
                obs = next_obs
                total_reward += reward
                step += 1
                
                if done:
                    break
            
            # Store episode data
            self.dataset['observations'].extend(episode_data['observations'])
            self.dataset['actions'].extend(episode_data['actions'])
            self.dataset['rewards'].extend(episode_data['rewards'])
            self.dataset['dones'].extend(episode_data['dones'])
            
            print(f"Episode {episode + 1}: {step} steps, Reward: {total_reward:.2f}")
        
        listener.stop()
        self.env.close()
    
    def collect_ppo_demonstrations(self, model_path, n_episodes=50):
        """Collect demonstrations using a trained PPO model"""
        print("🤖 Collecting PPO Expert Demonstrations...")
        
        # Use the original gym environment to avoid wrapper issues
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, info = env.reset()
        
        # Initialize PPO agent with correct observation shape
        obs_shape = (3, obs.shape[0], obs.shape[1])  # CHW format
        agent = PPOAgent(obs_shape=obs_shape)
        
        # Load the trained model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        agent.load(model_path)
        print(f"✅ Loaded PPO model from: {model_path}")
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            total_reward = 0
            step = 0
            
            while step < 1000:
                # Convert observation to CHW format
                obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
                
                # Get action from PPO agent
                action_id, _, _ = agent.act(obs_chw)
                action = ACTIONS[action_id]
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store data
                episode_data['observations'].append(obs)
                episode_data['actions'].append(action_id)  # Store discrete action ID
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                
                obs = next_obs
                total_reward += reward
                step += 1
                
                if done:
                    break
            
            # Store episode data
            self.dataset['observations'].extend(episode_data['observations'])
            self.dataset['actions'].extend(episode_data['actions'])
            self.dataset['rewards'].extend(episode_data['rewards'])
            self.dataset['dones'].extend(episode_data['dones'])
            
            print(f"PPO Episode {episode + 1}: {step} steps, Reward: {total_reward:.2f}")
        
        env.close()
    
    def save_dataset(self, filename="ppo_demos.pkl"):
        """Save collected demonstrations"""
        os.makedirs('data/expert_data', exist_ok=True)
        filepath = f"data/expert_data/{filename}"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.dataset, f)
        
        print(f"✅ Expert data saved to {filepath}")
        print(f"Total transitions: {len(self.dataset['observations'])}")
        print(f"Observations shape: {self.dataset['observations'][0].shape if self.dataset['observations'] else 'None'}")
        print(f"Actions sample: {self.dataset['actions'][:5] if self.dataset['actions'] else 'None'}")
    
    def load_dataset(self, filename="ppo_demos.pkl"):
        """Load demonstrations"""
        filepath = f"data/expert_data/{filename}"
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
            
        with open(filepath, 'rb') as f:
            self.dataset = pickle.load(f)
        
        print(f"✅ Expert data loaded from {filepath}")
        print(f"Total transitions: {len(self.dataset['observations'])}")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect expert demonstrations for GAIL')
    parser.add_argument('--human', action='store_true', help='Collect human demonstrations')
    parser.add_argument('--ppo', type=str, help='Collect from trained PPO model (provide model path)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default='ppo_demos.pkl', help='Output filename')
    
    args = parser.parse_args()
    
    collector = ExpertDataCollector()
    
    if args.human:
        print("🚗 Starting human demonstration collection...")
        collector.collect_human_demonstrations(n_episodes=args.episodes)
        collector.save_dataset(args.output)
        
    elif args.ppo:
        print(f"🤖 Collecting demonstrations from PPO model: {args.ppo}")
        collector.collect_ppo_demonstrations(model_path=args.ppo, n_episodes=args.episodes)
        collector.save_dataset(args.output)
        
    else:
        # Just show available options
        print("Expert Data Collection Options:")
        print("1. Collect human demonstrations:")
        print("   python scripts/collect_expert_data.py --human --episodes 10")
        print("2. Collect from trained PPO model:")
        print("   python scripts/collect_expert_data.py --ppo models/ppo/optimized_ppo.pth --episodes 20")
        print("3. Load existing dataset:")
        print("   python -c \"from scripts.collect_expert_data import ExpertDataCollector; c = ExpertDataCollector(); c.load_dataset()\"")
        
        # Create directory structure
        os.makedirs('data/expert_data', exist_ok=True)
        print("✅ Ready for expert data collection!")


if __name__ == "__main__":
    main()