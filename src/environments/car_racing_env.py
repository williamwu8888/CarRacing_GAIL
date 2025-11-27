import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any


class CarRacingEnvWrapper:
    """
    Wrapper for CarRacing-v2 environment with preprocessing and utility methods
    """
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 1000):
        self.env = gym.make('CarRacing-v2', render_mode=render_mode, continuous=False)
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self) -> np.ndarray:
        """Reset environment and return preprocessed observation"""
        self.current_step = 0
        obs, info = self.env.reset()
        return self._preprocess_observation(obs), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take step in environment and return preprocessed results"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._preprocess_observation(obs), reward, terminated, truncated, info
    
    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation (normalize, resize, etc.)"""
        # Normalize pixel values to [0, 1]
        obs_normalized = obs.astype(np.float32) / 255.0
        
        # You can add more preprocessing here:
        # - Resize images
        # - Convert to grayscale
        # - Frame stacking
        # - etc.
        
        return obs_normalized
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def get_action_meanings(self) -> list:
        """Get meaning of each action index"""
        return [
            "Do Nothing",
            "Steer Left", 
            "Steer Right",
            "Gas",
            "Brake"
        ]
    
    def sample_random_action(self) -> int:
        """Sample random action from action space"""
        return self.action_space.sample()


def test_environment():
    """Test function to verify environment works correctly"""
    env = CarRacingEnvWrapper(render_mode='human')
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Action space: {env.action_space}")
    
    for step in range(100):
        action = env.sample_random_action()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action={action}, Reward={reward:.3f}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()


if __name__ == "__main__":
    test_environment()