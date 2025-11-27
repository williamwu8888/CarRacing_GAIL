#!/usr/bin/env python3
"""
Simple test script to verify the CarRacing environment works
"""

import sys
import os

# Adjust sys.path to include src directory
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.environments.car_racing_env import CarRacingEnvWrapper

def basic_environment_test():
    """Test basic environment functionality"""
    print("🚗 Testing CarRacing-v2 Environment...")
    
    try:
        # Test 1: Create environment without rendering
        print("1. Creating environment...")
        env = CarRacingEnvWrapper(render_mode=None)
        print("   ✅ Environment created successfully!")
        
        # Test 2: Reset environment
        print("2. Resetting environment...")
        obs, info = env.reset()
        print(f"   ✅ Reset successful! Observation shape: {obs.shape}")
        print(f"   ✅ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test 3: Check action space
        print("3. Checking action space...")
        print(f"   ✅ Action space: {env.action_space}")
        print(f"   ✅ Action meanings: {env.get_action_meanings()}")
        
        # Test 4: Take random actions
        print("4. Testing random actions...")
        total_reward = 0
        for step in range(10):  # Just 10 steps for quick test
            action = env.sample_random_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step}: Action={action}, Reward={reward:.3f}, "
                  f"Obs shape={obs.shape}")
            
            if terminated or truncated:
                print("   Episode ended early!")
                break
        
        print(f"   ✅ Random actions test completed! Total reward: {total_reward:.3f}")
        
        # Test 5: Close environment
        print("5. Closing environment...")
        env.close()
        print("   ✅ Environment closed successfully!")
        
        print("\n🎉 All basic tests passed! Environment is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you installed dependencies: pip install -r requirements.txt")
        print("2. For Box2D issues, try: pip install gymnasium[box2d]")
        print("3. On Ubuntu, you might need: sudo apt-get install swig")
        print("4. On macOS, you might need: brew install swig")
        return False
    
    return True


def rendering_test():
    """Test environment with rendering (optional)"""
    print("\n👀 Testing environment rendering (optional)...")
    
    try:
        env = CarRacingEnvWrapper(render_mode='human')
        obs, info = env.reset()
        print("   ✅ Rendering environment created!")
        
        # Take a few steps to see if rendering works
        for step in range(5):
            action = env.sample_random_action()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {step}: Action={action}, Reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("   ✅ Rendering test completed!")
        
    except Exception as e:
        print(f"   ⚠️  Rendering test failed (this might be OK): {e}")
        print("   ℹ️  Rendering requires a display environment")


if __name__ == "__main__":
    print("=" * 60)
    print("CarRacing Environment Test Suite")
    print("=" * 60)
    
    # Run basic tests
    success = basic_environment_test()
    
    # Only run rendering test if basic tests passed
    if success:
        rendering_test()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed successfully!")
        print("You can now proceed with implementing PPO training.")
    else:
        print("❌ Some tests failed. Please check the troubleshooting tips above.")
    print("=" * 60)