#!/usr/bin/env python3
"""
FIXED GAIL (Generative Adversarial Imitation Learning) Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
import torch.nn.functional as F

from ..ppo.ppo_agent import PPOAgent, ActorCritic


class Discriminator(nn.Module):
    """Fixed Discriminator with better architecture and training"""
    
    def __init__(self, obs_shape: Tuple[int], action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        c, h, w = obs_shape
        self.obs_flatten_size = c * h * w
        self.action_dim = action_dim
        input_size = self.obs_flatten_size + action_dim
        
        # Better architecture with normalization
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)  # Smaller gain
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.size(0)
        obs_flat = observations.view(batch_size, -1)
        
        # One-hot encode actions
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # Concatenate and normalize
        x = torch.cat([obs_flat, actions_one_hot], dim=-1)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)  # Normalize
        
        logits = self.network(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities.squeeze(-1), logits.squeeze(-1)


class FixedGAILAgent:
    """Fixed GAIL agent with stable training"""
    
    def __init__(
        self, 
        obs_shape: Tuple[int], 
        action_dim: int = 5,
        lr_discriminator: float = 1e-4,  # Lower learning rate
        lr_policy: float = 2.5e-4,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Policy network
        self.policy = PPOAgent(obs_shape, action_dim)
        
        # Discriminator with lower learning rate
        self.discriminator = Discriminator(obs_shape, action_dim).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_discriminator, weight_decay=1e-5
        )
        
        # Training stats
        self.discriminator_losses = []
        self.policy_losses = []
        
    def compute_gail_reward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Stable reward computation with clipping"""
        with torch.no_grad():
            d_prob, _ = self.discriminator(observations, actions)
            
            # Clipped rewards to prevent explosion
            rewards = -torch.log(1 - d_prob.clamp(0.01, 0.99) + 1e-8)
            
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            return rewards
    
    def update_discriminator(
        self,
        expert_obs: np.ndarray,
        expert_actions: np.ndarray,
        policy_obs: np.ndarray, 
        policy_actions: np.ndarray
    ) -> Dict[str, float]:
        """Stable discriminator update with gradient clipping"""
        # Convert to tensors
        expert_obs_t = torch.tensor(expert_obs, dtype=torch.float32, device=self.device)
        expert_actions_t = torch.tensor(expert_actions, dtype=torch.long, device=self.device)
        policy_obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=self.device)
        policy_actions_t = torch.tensor(policy_actions, dtype=torch.long, device=self.device)
        
        # Get predictions
        expert_probs, expert_logits = self.discriminator(expert_obs_t, expert_actions_t)
        policy_probs, policy_logits = self.discriminator(policy_obs_t, policy_actions_t)
        
        # Label smoothing for stability
        expert_labels = torch.full_like(expert_probs, 0.9)  # Instead of 1.0
        policy_labels = torch.full_like(policy_probs, 0.1)  # Instead of 0.0
        
        # Loss with smoothing
        expert_loss = F.binary_cross_entropy(expert_probs, expert_labels)
        policy_loss = F.binary_cross_entropy(policy_probs, policy_labels)
        discriminator_loss = expert_loss + policy_loss
        
        # Update with gradient clipping
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.discriminator_optimizer.step()
        
        # Calculate accuracies
        expert_accuracy = (expert_probs > 0.5).float().mean()
        policy_accuracy = (policy_probs < 0.5).float().mean()
        
        stats = {
            'discriminator_loss': discriminator_loss.item(),
            'expert_accuracy': expert_accuracy.item(),
            'policy_accuracy': policy_accuracy.item(),
            'total_accuracy': (expert_accuracy.item() + policy_accuracy.item()) / 2
        }
        
        self.discriminator_losses.append(discriminator_loss.item())
        return stats
    
    def update_policy(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        logprobs_old: np.ndarray,
        values_old: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        epochs: int = 2,  # Fewer epochs for stability
        minibatch_size: int = 32
    ) -> float:
        """Policy update with environment rewards as baseline"""
        # Use BOTH GAIL rewards and environment rewards
        env_rewards = np.array(rewards, dtype=np.float32)
        gail_rewards = rewards  # Already computed
        
        # Blend rewards (70% GAIL, 30% environment)
        blended_rewards = 0.7 * gail_rewards + 0.3 * env_rewards
        
        # Normalize
        blended_rewards = (blended_rewards - blended_rewards.mean()) / (blended_rewards.std() + 1e-8)
        
        # Compute advantages
        values_np = np.array(values_old + [0.0], dtype=np.float32)
        advantages, returns = self.policy.compute_gae(blended_rewards, dones, values_old, 0.0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
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
        return self.policy.act(obs)
    
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.net.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.net.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])