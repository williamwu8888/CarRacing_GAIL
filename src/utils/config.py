import yaml
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration management for training and model parameters"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.data = config_dict or {}
        
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(config_dict)
    
    def save_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            yaml.dump(self.data, file, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        keys = key.split('.')
        value = self.data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        current = self.data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Default configuration for PPO
DEFAULT_PPO_CONFIG = {
    'environment': {
        'name': 'CarRacing-v2',
        'max_steps': 1000,
        'frame_stack': 4
    },
    'model': {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    },
    'training': {
        'total_timesteps': 100000,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'eval_frequency': 10000,
        'save_frequency': 50000
    },
    'network': {
        'hidden_sizes': [64, 64],
        'activation': 'tanh'
    }
}