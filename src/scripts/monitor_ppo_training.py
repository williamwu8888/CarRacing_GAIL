#!/usr/bin/env python3
"""
Monitor for the optimized PPO training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob


class PPOTrainingMonitor:
    def __init__(self, checkpoint_dir="models/ppo"):
        self.checkpoint_dir = checkpoint_dir
        
    def find_latest_checkpoint(self):
        """Find the most recent PPO checkpoint"""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
        if not checkpoints:
            return None
        
        # Filter out GAIL checkpoints if any
        ppo_checkpoints = [c for c in checkpoints if "gail" not in c.lower()]
        if not ppo_checkpoints:
            return None
            
        return max(ppo_checkpoints, key=os.path.getctime)
    
    def get_training_status(self):
        """Get current training status"""
        latest = self.find_latest_checkpoint()
        if not latest:
            return "No checkpoints found"
        
        # For this optimized version, we don't store episode rewards in checkpoints
        # So we'll just report the file info
        file_size = os.path.getsize(latest) / 1024 / 1024  # MB
        modified_time = time.ctime(os.path.getmtime(latest))
        
        status = f"Latest: {os.path.basename(latest)}\n"
        status += f"Size: {file_size:.1f} MB\n"
        status += f"Modified: {modified_time}"
        
        return status


def live_monitor(update_interval=30):
    """Simple live monitoring"""
    monitor = PPOTrainingMonitor()
    
    print("🔍 PPO Training Monitor")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            status = monitor.get_training_status()
            print(f"\r{status}", end="", flush=True)
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")


if __name__ == "__main__":
    live_monitor()