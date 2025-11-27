import sys
import os
import gymnasium as gym
from .base_env_adaptor import BaseEnvAdaptor
from .env_config import mountaincar_config
from utils import get_timestamp_ms

class MountainCarAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.seed = mountaincar_config.get('random_seed')
        
        # Create MountainCar environment
        self.env = gym.make(mountaincar_config['id'])
        
        # Apply custom parameters if provided
        if mountaincar_config.get('goal_position'):
            self.env.unwrapped.goal_position = mountaincar_config['goal_position']
        if mountaincar_config.get('goal_velocity'):
            self.env.unwrapped.goal_velocity = mountaincar_config['goal_velocity']
        if mountaincar_config.get('force'):
            self.env.unwrapped.force = mountaincar_config['force']
        if mountaincar_config.get('gravity'):
            self.env.unwrapped.gravity = mountaincar_config['gravity']
        
        self.reproduce_method = "action_path"
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
        # History records
        self.action_path = []
        self.st = None
        self.prev_action = None
        self.st1 = None
        self.terminated = False
        self.truncated = False
        self.reward = None
    
    def initialize_env(self):
        """Reset environment for a new episode."""
        observation, info = self.env.reset()
        print(f"Observation: {observation}")
        print(f"Info: {info}")
        
        # Reset episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        
        # Reset history
        self.st = None
        self.prev_action = None
        self.st1 = self.get_state()
        self.action_path = []
        self.terminated = False
        self.truncated = False
        self.reward = None
    
    def step(self, action):
        """Execute one step in the environment."""
        # Record history
        self.st = self.st1
        self.prev_action = action
        self.action_path.append(action)
        
        # Execute action
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update tracking
        self.episode_reward += reward
        self.episode_length += 1
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward
        
        # Update state
        self.st1 = self.get_state()
    
    def get_state(self):
        """
        Get discretized state representation.
        
        State consists of:
        - position: car's position on the hill (-1.2 to 0.6)
        - velocity: car's velocity (-0.07 to 0.07)
        
        We discretize these into bins for easier learning.
        """
        obs = self.env.unwrapped.state
        position, velocity = obs
        
        # Discretize position into bins
        # Key positions: valley bottom (~-0.5), left edge (-1.2), goal (0.5+)
        position_bins = [-1.2, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.45, 0.5, 0.6]
        
        # Discretize velocity into bins
        # Negative = moving left, Positive = moving right, Zero = stationary
        velocity_bins = [-0.07, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.07]
        
        position_bin = self._get_bin(position, position_bins)
        velocity_bin = self._get_bin(velocity, velocity_bins)
        
        return {
            'position_bin': position_bin,
            'velocity_bin': velocity_bin,
            'position': round(position, 3),  # For debugging/display
            'velocity': round(velocity, 4)   # For debugging/display
        }
    
    def _get_bin(self, value, bins):
        """Helper to find which bin a value falls into."""
        for i, threshold in enumerate(bins):
            if value < threshold:
                return i
        return len(bins)
    
    def get_env_description(self):
        """Get environment description for LLM."""
        return f"""=== Mountain Car Environment ===
Goal: Drive the car to reach the flag at position 0.5 (top of the right hill)

Challenge: The car's engine is not strong enough to climb the hill in one go.
You must build momentum by driving back and forth.

Strategy: Drive left to gain momentum, then use that momentum to climb right.

Position Range: -1.2 (far left) to 0.6 (beyond goal)
Velocity Range: -0.07 (max left) to 0.07 (max right)
Goal Position: 0.5 (flag location)

Actions:
  0 = Push LEFT (accelerate left)
  1 = NO PUSH (coast, no acceleration)
  2 = Push RIGHT (accelerate right)
"""
    
    def get_instruction(self):
        """Get task instruction."""
        return "Reach the flag at position 0.5 by building momentum"
    
    def get_available_actions(self):
        """Return available actions."""
        return [0, 1, 2]  # Left, Nothing, Right
    
    def get_action_path(self):
        return self.action_path.copy()
    
    def get_experience(self):
        """
        Create an experience record.
        Format: {id, reproduce_method, action_path, st, action, st1}
        """
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[mountainCar_adaptor] In get_experience(), history not set")
        
        experience = {
            "id": f"{get_timestamp_ms()}_mountaincar_{'-'.join(map(str, self.action_path))}",
            "reproduce_method": self.reproduce_method,
            "action_path": self.get_action_path(),
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience
    
    def is_valid_action(self, action):
        """Check if action is valid (0, 1, or 2)."""
        return action in self.get_available_actions()
    
    def is_finished_state(self, state):
        """Check if the episode is finished."""
        # Episode finishes when car reaches goal position (0.5)
        # Or when truncated (max steps reached)
        return self.terminated or self.truncated
    
    def extract_reward_score(self):
        """
        Extract final score.
        For MountainCar, negative number of steps (closer to 0 is better).
        If reached goal, return negative step count.
        """
        if self.terminated:
            # Reached goal - return negative step count (closer to 0 is better)
            return -self.episode_length
        else:
            # Failed to reach goal - return large negative number
            return -200  # Max episode length penalty
    
    def reconstruct_state(self, exp):
        """
        Reconstruct state from experience by replaying actions.
        Returns (success: bool, error_message: str)
        """
        assert exp['action'] == exp['action_path'][-1]
        assert len(exp['action_path']) > 0
        
        self.initialize_env()
        
        try:
            # Replay all actions except the last one
            for i in range(len(exp['action_path']) - 1):
                action = exp['action_path'][i]
                self.step(action)
                if self.is_finished_state(self.get_state()):
                    return False, f"Episode terminated early at step {i}"
        except Exception as e:
            return False, str(e)
        
        # Check if reconstructed state matches expected state
        current_state = self.get_state()
        if current_state != exp['st']:
            return False, f"State mismatch: expected {exp['st']}, got {current_state}"
        
        return True, None
    
    def format_action(self, raw_action):
        """Format action from LLM output."""
        # Try to extract integer action from string
        raw_action = str(raw_action).strip()
        
        # Remove common prefixes
        if raw_action.lower().startswith('action'):
            raw_action = raw_action[6:].strip()
        if raw_action.startswith(':'):
            raw_action = raw_action[1:].strip()
        
        # Extract first digit
        for char in raw_action:
            if char.isdigit():
                action = int(char)
                if action in [0, 1, 2]:
                    return action
        
        # Default to no push if parsing fails
        return 1

