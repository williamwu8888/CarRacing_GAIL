#!/usr/bin/env python3
"""
GAIL (Generative Adversarial Imitation Learning) Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
import torch.nn.functional as F

from ..ppo.ppo_agent import PPOAgent, ActorCritic


class Discriminator(nn.Module):
    """Discriminator network that distinguishes expert from policy trajectories"""
    
    def __init__(self, obs_shape: Tuple[int], action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        # Input: observation (flattened) + one-hot action
        c, h, w = obs_shape
        self.obs_flatten_size = c * h * w
        self.action_dim = action_dim
        input_size = self.obs_flatten_size + action_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator
        Returns probability that (s,a) comes from expert (close to 1) vs policy (close to 0)
        """
        # Flatten observations
        batch_size = observations.size(0)
        obs_flat = observations.view(batch_size, -1)
        
        # One-hot encode actions
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # Concatenate observations and actions
        x = torch.cat([obs_flat, actions_one_hot], dim=-1)
        
        # Get discriminator logits
        logits = self.network(x)
        
        return torch.sigmoid(logits).squeeze(-1), logits.squeeze(-1)


class GAILAgent:
    """GAIL agent that learns from expert demonstrations"""
    
    def __init__(
        self, 
        obs_shape: Tuple[int], 
        action_dim: int = 5,
        lr_discriminator: float = 3e-4,
        lr_policy: float = 2.5e-4,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Policy network (PPO agent)
        self.policy = PPOAgent(obs_shape, action_dim)
        
        # Discriminator network
        self.discriminator = Discriminator(obs_shape, action_dim).to(self.device)
        
        # Optimizers
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_discriminator
        )
        
        # We'll use the policy's existing optimizer
        
        # Training statistics
        self.discriminator_losses = []
        self.policy_losses = []
        self.expert_accuracies = []
        self.policy_accuracies = []
    
    def compute_gail_reward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GAIL rewards using discriminator
        Reward = -log(1 - D(s,a)) as in original GAIL paper
        """
        with torch.no_grad():
            d_prob, _ = self.discriminator(observations, actions)
            # Avoid log(0) by adding small epsilon
            rewards = -torch.log(1 - d_prob + 1e-8)
            return rewards
    
    def update_discriminator(
        self,
        expert_obs: np.ndarray,
        expert_actions: np.ndarray,
        policy_obs: np.ndarray, 
        policy_actions: np.ndarray
    ) -> Dict[str, float]:
        """
        Update discriminator to distinguish expert from policy trajectories
        """
        # Convert to tensors
        expert_obs_t = torch.tensor(expert_obs, dtype=torch.float32, device=self.device)
        expert_actions_t = torch.tensor(expert_actions, dtype=torch.long, device=self.device)
        policy_obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=self.device)
        policy_actions_t = torch.tensor(policy_actions, dtype=torch.long, device=self.device)
        
        # Get discriminator predictions
        expert_probs, expert_logits = self.discriminator(expert_obs_t, expert_actions_t)
        policy_probs, policy_logits = self.discriminator(policy_obs_t, policy_actions_t)
        
        # Discriminator loss (binary cross-entropy)
        expert_loss = F.binary_cross_entropy(expert_probs, torch.ones_like(expert_probs))
        policy_loss = F.binary_cross_entropy(policy_probs, torch.zeros_like(policy_probs))
        discriminator_loss = expert_loss + policy_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        # Calculate accuracies for monitoring
        expert_accuracy = (expert_probs > 0.5).float().mean()
        policy_accuracy = (policy_probs < 0.5).float().mean()
        total_accuracy = (expert_accuracy + policy_accuracy) / 2
        
        # Store statistics
        self.discriminator_losses.append(discriminator_loss.item())
        self.expert_accuracies.append(expert_accuracy.item())
        self.policy_accuracies.append(policy_accuracy.item())
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'expert_accuracy': expert_accuracy.item(),
            'policy_accuracy': policy_accuracy.item(),
            'total_accuracy': total_accuracy.item()
        }
    
    def update_policy(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        logprobs_old: np.ndarray,
        values_old: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        epochs: int = 4,
        minibatch_size: int = 64
    ) -> float:
        """
        Update policy using GAIL rewards with PPO
        """
        # Compute advantages and returns using the GAIL rewards
        values_np = np.array(values_old + [0.0], dtype=np.float32)  # Add dummy last value
        advantages, returns = self.policy.compute_gae(rewards, dones, values_old, 0.0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy using PPO
        policy_loss = self.policy.update(
            batch_obs=observations,
            batch_actions=actions,
            batch_logprobs=logprobs_old,
            batch_returns=returns,
            batch_advantages=advantages,
            epochs=epochs,
            minibatch_size=minibatch_size
        )
        
        self.policy_losses.append(policy_loss)
        return policy_loss
    
    def act(self, obs: np.ndarray):
        """Get action from policy (same interface as PPOAgent)"""
        return self.policy.act(obs)
    
    def save(self, path: str):
        """Save GAIL model (both policy and discriminator)"""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.net.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load GAIL model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.net.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.policy.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'discriminator_loss': np.mean(self.discriminator_losses[-10:]) if self.discriminator_losses else 0,
            'policy_loss': np.mean(self.policy_losses[-10:]) if self.policy_losses else 0,
            'expert_accuracy': np.mean(self.expert_accuracies[-10:]) if self.expert_accuracies else 0,
            'policy_accuracy': np.mean(self.policy_accuracies[-10:]) if self.policy_accuracies else 0,
        }