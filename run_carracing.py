#!/usr/bin/env python3
"""
Main entry point for CarRacing Social Robotics Project
"""

import argparse
from src.environments.car_racing_env import CarRacingEnvWrapper

def main():
    parser = argparse.ArgumentParser(description='CarRacing Social Robotics Project')
    parser.add_argument('--mode', type=str, choices=['train_ppo', 'train_gail', 'evaluate'], 
                       default='train_ppo', help='Mode to run')
    parser.add_argument('--render', action='store_true', help='Render environment')
    
    args = parser.parse_args()
    
    # Initialize environment
    env = CarRacingEnvWrapper(render_mode='human' if args.render else None)
    
    print(f"Running in {args.mode} mode")
    
    if args.mode == 'train_ppo':
        from scripts.train_ppo import main as train_ppo
        train_ppo()
    elif args.mode == 'train_gail':
        from scripts.train_gail import main as train_gail
        train_gail()
    else:
        print("Evaluation mode selected")

if __name__ == "__main__":
    main()