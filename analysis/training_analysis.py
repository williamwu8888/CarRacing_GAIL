# analysis/training_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

def plot_training_progress():
    """Plot training rewards and losses over time"""
    # Create individual plots instead of subplots
    
    # PPO Training Results
    ppo_rewards = [-54.4, -12.8, 45.6, 128.9, 212.3, 285.7, 
                325.8, 348.2, 360.1, 366.8, 370.3, 372.97]
    ppo_losses = [2.8, 2.3, 1.7, 1.2, 0.8, 0.5, 
                0.3, 0.2, 0.15, 0.12, 0.09, 0.07]

    # Behavioral Cloning from PPO Results
    bc_ppo_rewards = [-51.8, 68.3, 195.2, 285.7, 335.4, 356.8, 
                    365.2, 369.1, 371.0, 371.8, 372.3, 372.6]
    bc_ppo_losses = [1.2, 0.7, 0.4, 0.25, 0.18, 0.14, 
                    0.11, 0.09, 0.08, 0.075, 0.072, 0.07]
    epochs = range(1, len(ppo_rewards) + 1)

    # Plot 1: Reward Progression Comparison
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, ppo_rewards, 'b-', linewidth=2, label='PPO')
    plt.plot(epochs, bc_ppo_rewards, 'g-', linewidth=2, label='BC-PPO')
    plt.title('Training - Average Reward Progression')
    plt.xlabel('Training timesteps (x10K)')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/training_analysis/reward_progression_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Loss Progression Comparison
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, ppo_losses, 'r-', linewidth=2, label='PPO Loss')
    plt.plot(epochs, bc_ppo_losses, 'm-', linewidth=2, label='BC-PPO Loss')
    plt.title('Training - Average Loss Progression')
    plt.xlabel('Training timesteps (x10K)')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/training_analysis/loss_progression_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Behavioral Cloning Performance
    plt.figure(figsize=(8, 6))
    bc_epochs = list(range(1, 51))
    bc_losses = [0.7764, 0.6042, 0.5726, 0.5329, 0.5086, 0.4890, 0.4706, 0.4412, 0.4192, 0.4027,
                0.3960, 0.3863, 0.3773, 0.3688, 0.3621, 0.3533, 0.3500, 0.3449, 0.3402, 0.3332,
                0.3296, 0.3262, 0.3174, 0.3152, 0.3116, 0.3067, 0.3006, 0.3019, 0.2947, 0.2899,
                0.2837, 0.2836, 0.2748, 0.2748, 0.2665, 0.2630, 0.2577, 0.2521, 0.2492, 0.2456,
                0.2434, 0.2373, 0.2325, 0.2293, 0.2221, 0.2207, 0.2197, 0.2122, 0.2098, 0.2006]

    bc_accuracies = [0.722, 0.772, 0.782, 0.787, 0.793, 0.804, 0.816, 0.833, 0.841, 0.849,
                    0.851, 0.855, 0.858, 0.860, 0.861, 0.861, 0.866, 0.867, 0.868, 0.870,
                    0.873, 0.875, 0.877, 0.879, 0.880, 0.881, 0.884, 0.883, 0.886, 0.888,
                    0.890, 0.888, 0.894, 0.896, 0.896, 0.898, 0.901, 0.901, 0.904, 0.906,
                    0.907, 0.910, 0.911, 0.912, 0.917, 0.914, 0.917, 0.919, 0.919, 0.923]
    
    # Rewards de test aux epochs 10, 20, 30, 40, 50
    test_epochs = [10, 20, 30, 40, 50]
    test_rewards = [657.60, 702.53, 800.85, 587.60, 839.70]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Courbe de loss et accuracy
    ax1.plot(bc_epochs, bc_losses, 'r-', linewidth=2, label='BC Loss', alpha=0.7)
    ax1.plot(bc_epochs, bc_accuracies, 'g-', linewidth=2, label='BC Accuracy')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss / Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Courbe des rewards de test
    ax2 = ax1.twinx()
    ax2.plot(test_epochs, test_rewards, 'purple', linewidth=3, marker='o', markersize=6, 
            label='Test Reward', linestyle='--')
    ax2.set_ylabel('Test Reward', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(500, 900)

    plt.title('Behavioral Cloning Performance')
    plt.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    plt.tight_layout()
    plt.savefig('results/training_analysis/bc_performance.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Action Distribution Comparison
    plt.figure(figsize=(10, 6))
    methods = ['PPO', 'BC from PPO', 'BC from Human']
    nothing =   [16.0, 19.9, 60.5]
    left =      [15.6, 17.3, 18.3]
    right =     [15.4, 14.9, 7.7 ]
    gas =       [24.7, 25.0, 10.7]
    braking =   [28.4, 22.8, 2.8 ]
    
    x = np.arange(len(methods))
    width = 0.15  # Made thinner by reducing the width

    plt.bar(x - width*2, nothing, width, label='No Action', alpha=0.8)
    plt.bar(x - width, left, width, label='Left', alpha=0.8)
    plt.bar(x, gas, width, label='Gas', alpha=0.8)
    plt.bar(x + width, right, width, label='Right', alpha=0.8)
    plt.bar(x + width*2, braking, width, label='Braking', alpha=0.8)
    
    plt.title('Action Distribution by Method')
    plt.xlabel('Training Method')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_analysis/action_distribution.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_comparison():
    """Compare performance of different methods"""
    methods = ['PPO Only', 'BC (PPO Demos)', 'BC (Human Demos)', 'Human Expert']
    mean_rewards =  [297.72, 382.46, 702.38, 820.0]   # Human expert value is estimated
    std_rewards =   [209.34, 214.24, 249.40, 150.0]   # Estimated std for human expert
    success_rates = [12.0  , 18.0  , 81.0  , 95.0 ]   # in percentage
    
    # Plot 1: Mean Reward Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, mean_rewards, yerr=std_rewards, 
                   capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('Average Reward by Method')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/training_analysis/mean_reward_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Success Rate Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, success_rates, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('Success Rate by Method')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 1, f'{rate}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/training_analysis/success_rate_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_expert_data():
    """Analyze collected expert demonstrations"""
    try:
        with open('data/expert_data/human_demos.pkl', 'rb') as f:
            human_data = pickle.load(f)
        
        with open('data/expert_data/ppo_demos.pkl', 'rb') as f:
            ppo_data = pickle.load(f)
        
        # Plot 1: Dataset sizes
        plt.figure(figsize=(8, 6))
        datasets = ['Human Demos', 'PPO Demos']
        sizes = [len(human_data['observations']), len(ppo_data['observations'])]
        
        bars = plt.bar(datasets, sizes, color=['orange', 'blue'], alpha=0.7)
        plt.title('Expert Dataset Sizes')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        for i, size in enumerate(sizes):
            plt.text(i, size + max(sizes)*0.01, f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/training_analysis/dataset_sizes.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Action distribution comparison
        plt.figure(figsize=(10, 6))
        human_actions = np.array(human_data['actions'])
        ppo_actions = np.array(ppo_data['actions'])
        
        action_names = ["Nothing", "Left", "Right", "Gas", "Brake"]
        
        human_counts = [np.sum(human_actions == i) for i in range(5)]
        ppo_counts = [np.sum(ppo_actions == i) for i in range(5)]
        
        x = np.arange(len(action_names))
        width = 0.35
        
        plt.bar(x - width/2, human_counts, width, label='Human', alpha=0.7)
        plt.bar(x + width/2, ppo_counts, width, label='PPO', alpha=0.7)
        plt.title('Action Distribution in Expert Data')
        plt.xlabel('Actions')
        plt.ylabel('Count')
        plt.xticks(x, action_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_analysis/expert_action_distribution.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Reward distribution
        plt.figure(figsize=(10, 6))
        human_rewards = human_data['rewards']
        ppo_rewards = ppo_data['rewards']
        
        plt.hist(human_rewards, bins=50, alpha=0.7, label='Human', density=True)
        plt.hist(ppo_rewards, bins=50, alpha=0.7, label='PPO', density=True)
        plt.title('Reward Distribution in Expert Data')
        plt.xlabel('Reward')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_analysis/expert_reward_distribution.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.close()
        
    except FileNotFoundError as e:
        print(f"Could not load expert data: {e}")

def plot_learning_curves():
    """Plot learning curves for different methods"""
    plt.figure(figsize=(12, 8))
    
    # Example data
    timesteps = np.arange(0, 120000, 10000)
    
    # PPO learning curve
    ppo_rewards = [-54.4, -12.8, 45.6, 128.9, 212.3, 285.7, 
                   325.8, 348.2, 360.1, 366.8, 370.3, 372.97]
    # BC learning curves
    bc_ppo_rewards = [-51.8, 68.3, 195.2, 285.7, 335.4, 356.8, 365.2, 369.1, 371.0, 371.8, 372.3, 372.6]
    bc_human_rewards = [505.2, 580.3, 620.5, 650.7, 680.1, 700.4, 720.6, 735.2, 745.8, 755.0, 760.3, 765.0]
    
    plt.plot(timesteps, ppo_rewards, 'b-', linewidth=3, label='PPO', marker='o')
    plt.plot(timesteps, bc_ppo_rewards, 'g-', linewidth=3, label='BC (PPO Demos)', marker='s')
    plt.plot(timesteps, bc_human_rewards, 'orange', linewidth=3, label='BC (Human Demos)', marker='^')
    
    # Add human expert baseline
    plt.axhline(y=800, color='red', linestyle='--', linewidth=2, label='Human Expert')

    plt.ylim(-80, 850)    
    plt.title('Learning Curves: PPO vs Behavioral Cloning', fontsize=14, fontweight='bold')
    plt.xlabel('Training Timesteps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/training_analysis/learning_curves.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all plots"""
    # Create results directory
    os.makedirs('results/training_analysis', exist_ok=True)
    
    print("Generating training progress plots...")
    plot_training_progress()
    
    print("Generating evaluation comparison plots...")
    plot_evaluation_comparison()
    
    print("Generating expert data analysis plots...")
    analyze_expert_data()
    
    print("Generating learning curves...")
    plot_learning_curves()
    
    print("✅ All plots saved as .eps files in the 'results/training_analysis' folder!")
    
    # List generated files
    result_files = os.listdir('results/training_analysis')
    print(f"\nGenerated {len(result_files)} files:")
    for file in sorted(result_files):
        print(f"  - {file}")

if __name__ == "__main__":
    main()