#!/usr/bin/env python3
"""
Collect human demonstrations for GAIL training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import gymnasium as gym
import numpy as np
import pickle
from models.ppo.ppo_agent import ACTIONS


def collect_human_demonstrations(n_episodes=10, output_path="data/expert_data/human_demos.pkl"):
    """Collect human driving demonstrations"""
    
    print("🚗 Collecting Human Demonstrations")
    print("Controls:")
    print("  A/D - Steer Left/Right")
    print("  W/S - Gas/Brake")
    print("  SPACE - Do nothing") 
    print("  ESC - Quit")
    print("=" * 50)
    
    pygame.init()
    env = gym.make("CarRacing-v2", render_mode='human')
    
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    for episode in range(n_episodes):
        print(f"\n🎮 Episode {episode + 1}/{n_episodes}")
        print("Get ready to drive! The environment will start in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            pygame.time.wait(1000)
        
        obs, info = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < 1000:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return
            
            # Get keyboard state
            keys = pygame.key.get_pressed()
            
            # Map keys to actions
            action_id = 0  # Default: do nothing
            
            # Map keys to actions (using only the 5 available actions)
            action_id = 0  # Default: do nothing

            if keys[pygame.K_a]:
                action_id = 1  # Left
            elif keys[pygame.K_d]:
                action_id = 2  # Right
            elif keys[pygame.K_w]:
                action_id = 3  # Gas
            elif keys[pygame.K_s]:
                action_id = 4  # Brake
            
            action = ACTIONS[action_id]
            
            # Take step
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
            
            # Display progress
            total_tiles = info.get("all_tiles", 0)
            visited_tiles = info.get("tiles", 0)
            if total_tiles > 0:
                progress = visited_tiles / total_tiles * 100
                print(f"\rStep: {step:3d} | Reward: {episode_reward:6.1f} | Progress: {progress:5.1f}%", end="")
        
        print(f"\n✅ Episode {episode + 1} complete: {step} steps, Reward: {episode_reward:.1f}")
        
        # Add to dataset
        dataset['observations'].extend(episode_data['observations'])
        dataset['actions'].extend(episode_data['actions'])
        dataset['rewards'].extend(episode_data['rewards'])
        dataset['dones'].extend(episode_data['dones'])
        
        # Ask to continue
        if episode < n_episodes - 1:
            print("Press any key to continue to next episode...")
            pygame.event.clear()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        waiting = False
                    if event.type == pygame.QUIT:
                        env.close()
                        return
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    env.close()
    pygame.quit()
    
    print(f"\n🎉 Human demonstrations complete!")
    print(f"✅ Saved {len(dataset['observations'])} samples to {output_path}")
    
    # Print statistics
    actions = np.array(dataset['actions'])
    unique, counts = np.unique(actions, return_counts=True)
    action_names = ["Nothing", "Left", "Right", "Gas", "Brake", "L+Gas", "R+Gas", "L+Brake", "R+Brake"]
    
    print("\n📊 Action Distribution:")
    for action_id, count in zip(unique, counts):
        percentage = count / len(actions) * 100
        print(f"  {action_names[action_id]:8}: {count:5} ({percentage:5.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--output", type=str, default="data/expert_data/human_demos.pkl", help="Output path")
    
    args = parser.parse_args()
    
    collect_human_demonstrations(args.episodes, args.output)