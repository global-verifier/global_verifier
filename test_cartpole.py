#!/usr/bin/env python3
"""
Quick test script for CartPole environment.
Tests basic functionality without requiring LLM.
"""

import sys
import gymnasium as gym
from env_adaptors.cartPole_adaptor import CartPoleAdaptor
from env_adaptors.cartPole_llama_adaptor import CartPoleLlamaAdaptor

def test_basic_adaptor():
    """Test basic CartPole adaptor functionality."""
    print("=" * 80)
    print("Test 1: Basic CartPole Adaptor")
    print("=" * 80)
    
    adaptor = CartPoleAdaptor("cartpole_test")
    
    # Initialize
    print("\n1. Initializing environment...")
    adaptor.initialize_env()
    print(adaptor.get_env_description())
    
    # Get initial state
    print("\n2. Getting initial state...")
    state = adaptor.get_state()
    print(f"Initial state: {state}")
    
    # Test actions
    print("\n3. Testing actions...")
    for i in range(5):
        action = i % 2  # Alternate between 0 and 1
        print(f"\n  Step {i+1}: Action {action}")
        adaptor.step(action)
        new_state = adaptor.get_state()
        print(f"  New state: {new_state}")
        print(f"  Reward: {adaptor.reward}")
        print(f"  Done: {adaptor.is_finished_state(new_state)}")
        
        if adaptor.is_finished_state(new_state):
            print("  Episode finished!")
            break
    
    # Get experience
    print("\n4. Getting experience...")
    try:
        exp = adaptor.get_experience()
        print(f"Experience ID: {exp['id']}")
        print(f"Action path: {exp['action_path']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Final score
    print(f"\n5. Final score: {adaptor.extract_reward_score()}")
    
    print("\n✅ Basic adaptor test passed!")
    return True

def test_llama_adaptor():
    """Test LLaMA adaptor (prompt generation only, no LLM inference)."""
    print("\n" + "=" * 80)
    print("Test 2: LLaMA Adaptor (Prompt Generation)")
    print("=" * 80)
    
    adaptor = CartPoleLlamaAdaptor("cartpole_llama_test")
    
    # Initialize
    print("\n1. Initializing environment...")
    adaptor.initialize_env()
    
    # Run a few steps
    print("\n2. Running a few steps to generate experiences...")
    for i in range(3):
        action = i % 2
        adaptor.step(action)
    
    # Get current state
    state = adaptor.get_state()
    print(f"\n3. Current state: {state}")
    
    # Test prompt generation without experiences
    print("\n4. Generating prompt (no experiences)...")
    prompt_no_exp = adaptor.get_action_prompt([])
    print(f"Prompt length: {len(prompt_no_exp)} characters")
    print(f"\nPrompt preview (first 500 chars):\n{prompt_no_exp[:500]}...")
    
    # Create fake experiences
    print("\n5. Generating prompt (with fake experiences)...")
    fake_experiences = [
        {
            'action': 0,
            'st': state,
            'st1': {'x_bin': 1, 'theta_bin': 2, 'x_dot_sign': -1, 'theta_dot_sign': 0}
        },
        {
            'action': 1,
            'st': state,
            'st1': {'x_bin': 5, 'theta_bin': 6, 'x_dot_sign': 1, 'theta_dot_sign': 1}  # Failure state
        }
    ]
    prompt_with_exp = adaptor.get_action_prompt(fake_experiences)
    print(f"Prompt length: {len(prompt_with_exp)} characters")
    print(f"\nChecking for experience warnings in prompt...")
    if "DANGEROUS" in prompt_with_exp:
        print("✓ Found DANGEROUS action warning")
    if "SUCCESSFUL" in prompt_with_exp:
        print("✓ Found SUCCESSFUL action info")
    
    # Test action formatting
    print("\n6. Testing action formatting...")
    test_responses = ["0", "1", "Action 0", "I choose 1", "0\n"]
    for response in test_responses:
        try:
            formatted = adaptor.format_action(response)
            print(f"  '{response}' → {formatted} ✓")
        except Exception as e:
            print(f"  '{response}' → Error: {e}")
    
    print("\n✅ LLaMA adaptor test passed!")
    return True

def test_state_reconstruction():
    """Test state reconstruction from experience."""
    print("\n" + "=" * 80)
    print("Test 3: State Reconstruction")
    print("=" * 80)
    
    adaptor = CartPoleAdaptor("cartpole_reconstruct_test")
    adaptor.initialize_env()
    
    # Run a few steps and collect experience
    print("\n1. Running steps and collecting experience...")
    action_sequence = [1, 0, 1]
    for action in action_sequence:
        adaptor.step(action)
    
    exp = adaptor.get_experience()
    print(f"Experience action path: {exp['action_path']}")
    print(f"Experience st: {exp['st']}")
    print(f"Experience st1: {exp['st1']}")
    
    # Test reconstruction
    print("\n2. Testing reconstruction...")
    success, error = adaptor.reconstruct_state(exp)
    if success:
        print("✓ Reconstruction successful")
        reconstructed_state = adaptor.get_state()
        print(f"Reconstructed state: {reconstructed_state}")
        print(f"Expected state: {exp['st']}")
        if reconstructed_state == exp['st']:
            print("✓ States match!")
        else:
            print("✗ States don't match!")
    else:
        print(f"✗ Reconstruction failed: {error}")
    
    print("\n✅ State reconstruction test passed!")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CartPole Adaptor Test Suite")
    print("=" * 80)
    
    try:
        test_basic_adaptor()
        test_llama_adaptor()
        test_state_reconstruction()
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print("\nCartPole environment is ready to use with Explorer!")
        print("Next steps:")
        print("  1. Update config.py to use cartpole_llama")
        print("  2. Run: python explore20times.py")
        print("  3. Or use: playground_cartPole.ipynb")
        
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

