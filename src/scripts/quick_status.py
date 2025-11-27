#!/usr/bin/env python3
"""
Quick status check for training progress and model files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import time
import torch
import numpy as np


def check_ppo_status():
    """Check PPO training status"""
    print("🔍 PPO Status Check")
    print("-" * 40)
    
    # Check for PPO models
    ppo_models = glob.glob("models/ppo/*.pth")
    
    if not ppo_models:
        print("❌ No PPO models found")
        print("   Run: python scripts/train_ppo.py --timesteps 200000")
        return
    
    # Find latest model
    latest_model = max(ppo_models, key=os.path.getctime)
    file_size = os.path.getsize(latest_model) / 1024 / 1024  # MB
    modified_time = time.ctime(os.path.getmtime(latest_model))
    
    print(f"✅ Latest PPO model: {os.path.basename(latest_model)}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Modified: {modified_time}")
    
    # Try to load and get basic info
    try:
        checkpoint = torch.load(latest_model, map_location='cpu')
        if 'timestep' in checkpoint:
            print(f"   Training steps: {checkpoint['timestep']:,}")
        if 'episode_rewards' in checkpoint:
            rewards = checkpoint['episode_rewards']
            if rewards:
                recent = rewards[-5:]
                print(f"   Recent rewards: {[f'{r:.1f}' for r in recent]}")
                print(f"   Best reward: {max(rewards):.1f}" if rewards else "   No reward data")
    except:
        print("   (Could not load model details)")


def check_gail_status():
    """Check GAIL training status"""
    print("\n🤖 GAIL Status Check")
    print("-" * 40)
    
    gail_models = glob.glob("models/gail/*.pth")
    
    if not gail_models:
        print("❌ No GAIL models found")
        print("   Run: python scripts/train_gail.py --expert-data data/expert_data/ppo_demos.pkl")
        return
    
    latest_model = max(gail_models, key=os.path.getctime)
    file_size = os.path.getsize(latest_model) / 1024 / 1024
    modified_time = time.ctime(os.path.getmtime(latest_model))
    
    print(f"✅ Latest GAIL model: {os.path.basename(latest_model)}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Modified: {modified_time}")


def check_expert_data():
    """Check expert data availability"""
    print("\n📊 Expert Data Status")
    print("-" * 40)
    
    expert_files = glob.glob("data/expert_data/*.pkl")
    
    if not expert_files:
        print("❌ No expert data found")
        print("   Run: python scripts/collect_expert_data.py --ppo models/ppo/ppo_model.pth --episodes 30")
        return
    
    for file in expert_files:
        file_size = os.path.getsize(file) / 1024 / 1024
        modified_time = time.ctime(os.path.getmtime(file))
        
        try:
            import pickle
            with open(file, 'rb') as f:
                data = pickle.load(f)
            samples = len(data['observations'])
            print(f"✅ {os.path.basename(file)}")
            print(f"   Samples: {samples:,}")
            print(f"   Size: {file_size:.1f} MB")
            print(f"   Modified: {modified_time}")
        except:
            print(f"⚠️  {os.path.basename(file)} (corrupted?)")


def check_training_progress():
    """Check if training is currently running"""
    print("\n🔄 Training Progress")
    print("-" * 40)
    
    # Check for recently modified model files (last 10 minutes)
    recent_models = []
    for pattern in ["models/ppo/*.pth", "models/gail/*.pth"]:
        for file in glob.glob(pattern):
            if time.time() - os.path.getmtime(file) < 600:  # 10 minutes
                recent_models.append(file)
    
    if recent_models:
        print("✅ Training appears active (recent model updates)")
        for model in recent_models:
            print(f"   {os.path.basename(model)}")
    else:
        print("❌ No recent training activity detected")


def main():
    """Main status check function"""
    print("🚗 CarRacing Project - Quick Status")
    print("=" * 50)
    
    check_ppo_status()
    check_gail_status()
    check_expert_data()
    check_training_progress()
    
    print("\n" + "=" * 50)
    print("💡 Next Steps:")
    
    # Suggest next actions based on current state
    ppo_models = glob.glob("models/ppo/*.pth")
    gail_models = glob.glob("models/gail/*.pth")
    expert_files = glob.glob("data/expert_data/*.pkl")
    
    if not ppo_models:
        print("   1. Start PPO training: python scripts/train_ppo.py --timesteps 200000")
    elif not expert_files:
        print("   1. Collect demonstrations: python scripts/collect_expert_data.py --ppo models/ppo/ppo_model.pth")
    elif not gail_models:
        print("   1. Train GAIL: python scripts/train_gail.py --expert-data data/expert_data/ppo_demos.pkl")
    else:
        print("   1. Evaluate models: python scripts/evaluate_model.py --model models/ppo/ppo_model.pth --quick")
        print("   2. Compare with GAIL: python scripts/evaluate_model.py --model models/gail/gail_model.pth --quick")


if __name__ == "__main__":
    main()