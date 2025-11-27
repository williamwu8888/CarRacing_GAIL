#!/usr/bin/env python3
"""
Test the PPO implementation
"""

import sys
import os

# From tests directory, go up to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
import gymnasium as gym


def test_gae_computation():
    """Test the GAE computation"""
    print("🧪 Testing GAE Computation...")
    
    from src.models.ppo.ppo_agent import PPOAgent
    
    # Create dummy data
    rewards = np.array([1.0, 0.5, -0.1, 2.0, 0.0], dtype=np.float32)
    values = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    dones = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Last step is done
    
    # Create a PPO agent instance to test GAE
    env = gym.make("CarRacing-v2", render_mode=None)
    obs, info = env.reset()
    obs_shape = (3, obs.shape[0], obs.shape[1])
    agent = PPOAgent(obs_shape=obs_shape)
    
    try:
        # Test the compute_gae method
        advantages, returns = agent.compute_gae(rewards, dones, values, last_value=0.0)
        print("✅ GAE computation successful!")
        print(f"Advantages: {advantages}")
        print(f"Returns: {returns}")
        
        # Test with different last_value
        advantages2, returns2 = agent.compute_gae(rewards, dones, values, last_value=0.3)
        print(f"With last_value=0.3 - Advantages: {advantages2}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ GAE computation failed: {e}")
        env.close()
        return False


def test_action_selection():
    """Test action selection with the new PPO agent"""
    print("\n🧪 Testing Action Selection...")
    
    from src.models.ppo.ppo_agent import PPOAgent, ACTIONS
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, info = env.reset()
        obs_shape = (3, obs.shape[0], obs.shape[1])
        
        agent = PPOAgent(obs_shape=obs_shape)
        
        # Test multiple action selections
        for i in range(5):
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, logprob, value = agent.act(obs_chw)
            
            print(f"  Step {i+1}: Action={action_id}, LogProb={logprob:.3f}, Value={value:.3f}")
            
            # Verify action is valid
            assert action_id in ACTIONS, f"Invalid action ID: {action_id}"
            assert 0 <= action_id < len(ACTIONS), f"Action ID out of range: {action_id}"
            
            # Take the action
            action = ACTIONS[action_id]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("    Environment reset")
        
        env.close()
        print("✅ Action selection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_update():
    """Test the PPO update mechanism"""
    print("\n🧪 Testing PPO Update...")
    
    from src.models.ppo.ppo_agent import PPOAgent
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, info = env.reset()
        obs_shape = (3, obs.shape[0], obs.shape[1])
        
        agent = PPOAgent(obs_shape=obs_shape)
        
        # Create dummy batch data for update
        batch_size = 32
        batch_obs = np.random.randn(batch_size, 3, 96, 96).astype(np.float32)
        batch_actions = np.random.randint(0, 5, size=batch_size).astype(np.int64)
        batch_logprobs = np.random.randn(batch_size).astype(np.float32)
        batch_returns = np.random.randn(batch_size).astype(np.float32)
        batch_advantages = np.random.randn(batch_size).astype(np.float32)
        
        # Test the update method
        loss = agent.update(
            batch_obs=batch_obs,
            batch_actions=batch_actions,
            batch_logprobs=batch_logprobs,
            batch_returns=batch_returns,
            batch_advantages=batch_advantages,
            epochs=2,
            minibatch_size=16
        )
        
        print(f"✅ PPO update successful! Loss: {loss:.4f}")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ PPO update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """Test model saving and loading"""
    print("\n🧪 Testing Model Save/Load...")
    
    from src.models.ppo.ppo_agent import PPOAgent
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, info = env.reset()
        obs_shape = (3, obs.shape[0], obs.shape[1])
        
        # Create first agent and get action
        agent1 = PPOAgent(obs_shape=obs_shape)
        obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
        action1, logprob1, value1 = agent1.act(obs_chw)
        
        # Save model
        test_save_path = "models/ppo/test_save_load.pth"
        agent1.save(test_save_path)
        print(f"✅ Model saved to: {test_save_path}")
        
        # Create second agent and load
        agent2 = PPOAgent(obs_shape=obs_shape)
        agent2.load(test_save_path)
        print(f"✅ Model loaded from: {test_save_path}")
        
        # Test if they produce same output
        action2, logprob2, value2 = agent2.act(obs_chw)
        
        print(f"Agent1 - Action: {action1}, LogProb: {logprob1:.3f}, Value: {value1:.3f}")
        print(f"Agent2 - Action: {action2}, LogProb: {logprob2:.3f}, Value: {value2:.3f}")
        
        # Clean up
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
            print(f"✅ Test file cleaned up: {test_save_path}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Model save/load failed: {e}")
        # Clean up on failure
        test_save_path = "models/ppo/test_save_load.pth"
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        return False


def quick_ppo_integration_test():
    """Quick integration test of the PPO agent"""
    print("\n🧪 Quick PPO Integration Test...")
    
    from src.models.ppo.ppo_agent import PPOAgent, ACTIONS
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, info = env.reset()
        obs_shape = (3, obs.shape[0], obs.shape[1])
        
        agent = PPOAgent(obs_shape=obs_shape)
        
        # Test a few steps of interaction
        total_reward = 0
        steps = 0
        
        for step in range(50):  # Just 50 steps for quick test
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            action_id, logprob, value = agent.act(obs_chw)
            action = ACTIONS[action_id]
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                print(f"  Episode finished at step {step}, reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
        
        print(f"✅ Integration test successful! Completed {steps} steps")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Implementation Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests = [
        test_gae_computation,
        test_action_selection, 
        test_ppo_update,
        test_model_save_load,
        quick_ppo_integration_test
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PPO implementation is working correctly.")
        print("You can now run proper training with: python scripts/train_ppo.py")
    else:
        print("⚠️  Some tests failed. Check the implementation before training.")
    print("=" * 60)