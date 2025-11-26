import sys
import os
import random
import gymnasium as gym
from .base_env_adaptor import BaseEnvAdaptor
from .env_config import cartpole_config
from utils import get_timestamp_ms

class CartPoleAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.seed = cartpole_config.get('random_seed')
        
        # Create CartPole environment with seed
        self.env = gym.make(cartpole_config['id'])
        
        # Apply custom physics parameters if provided
        if cartpole_config.get('force_mag'):
            self.env.unwrapped.force_mag = cartpole_config['force_mag']
        if cartpole_config.get('gravity'):
            self.env.unwrapped.gravity = cartpole_config['gravity']
        if cartpole_config.get('masscart'):
            self.env.unwrapped.masscart = cartpole_config['masscart']
        if cartpole_config.get('masspole'):
            self.env.unwrapped.masspole = cartpole_config['masspole']
            # Recalculate derived parameters
            self.env.unwrapped.total_mass = self.env.unwrapped.masspole + self.env.unwrapped.masscart
            self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.env.unwrapped.length
        if cartpole_config.get('length'):
            self.env.unwrapped.length = cartpole_config['length']
            self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.env.unwrapped.length
        if cartpole_config.get('tau'):
            self.env.unwrapped.tau = cartpole_config['tau']
        
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
        # observation, info = self.env.reset(seed=self.seed)
        observation, info = self.env.reset()
        print(f"Observation: {observation}")
        print(f"Info: {info}")
        # Reset history
        self.st = None
        self.prev_action = None
        self.st1 = self.get_state()
        self.action_path = []
        self.terminated = False
        self.truncated = False
        self.reward = None
        self.episode_reward = 0
        self.episode_length = 0

    def step(self, action):
        """Execute one step in the environment."""
        # Record history
        self.st = self.st1
        self.prev_action = action
        self.action_path.append(action)
        
        # Execute action
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        
        # Update current state
        self.st1 = self.get_state()

    def is_finished_state(self, state):
        """
        Check if the state is terminal.
        CartPole terminates when:
        1. Pole angle exceeds threshold (±12°)
        2. Cart position exceeds threshold (±2.4)
        3. Episode reaches max steps
        """
        return self.terminated or self.truncated

    def extract_reward_score(self):
        """
        Return the total reward for the episode.
        In CartPole, this is the number of steps the pole stayed balanced.
        """
        return self.episode_reward

    # Getters
    def get_env_description(self):
        return f"""-----
Init new environment:
[ENV] CartPole-v1
[Goal] Balance the pole on the cart for as long as possible
[Max Steps] {self.env.spec.max_episode_steps}
[Physics] force_mag={self.env.unwrapped.force_mag}, gravity={self.env.unwrapped.gravity}
-----"""
    
    def get_instruction(self):
        return "Balance the pole on the cart for as long as possible by moving left or right"

    def get_state(self):
        """
        Extract state representation from the environment.
        CartPole state: [x, x_dot, theta, theta_dot]
        We discretize continuous values for experience matching.
        """
        obs = self.env.unwrapped.state
        if obs is None:
            return {
                'x_bin': 0,
                'theta_bin': 0,
                'x_dot_sign': 0,
                'theta_dot_sign': 0,
            }
        
        x, x_dot, theta, theta_dot = obs
        
        # Discretize position into bins
        x_bins = [-2.4, -1.2, -0.5, 0, 0.5, 1.2, 2.4]
        theta_bins = [-0.2095, -0.1, -0.05, 0, 0.05, 0.1, 0.2095]
        
        x_bin = self._get_bin(x, x_bins)
        theta_bin = self._get_bin(theta, theta_bins)
        
        # Record velocity direction (simplified)
        x_dot_sign = 1 if x_dot > 0.1 else (-1 if x_dot < -0.1 else 0)
        theta_dot_sign = 1 if theta_dot > 0.1 else (-1 if theta_dot < -0.1 else 0)
        
        return {
            'x_bin': x_bin,
            'theta_bin': theta_bin,
            'x_dot_sign': x_dot_sign,
            'theta_dot_sign': theta_dot_sign,
        }
    
    def _get_bin(self, value, bins):
        """Find which bin the value falls into."""
        for i, threshold in enumerate(bins):
            if value < threshold:
                return i
        return len(bins)

    def get_available_actions(self):
        """CartPole has 2 discrete actions."""
        return [0, 1]

    def get_action_path(self):
        return self.action_path.copy()

    def get_experience(self):
        """
        Create an experience record.
        Format: {id, reproduce_method, action_path, st, action, st1}
        """
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[cartPole_adaptor] In get_experience(), history not set")
        
        experience = {
            "id": f"{get_timestamp_ms()}_cartpole_{'-'.join(map(str, self.action_path))}",
            "reproduce_method": self.reproduce_method,
            "action_path": self.get_action_path(),
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience

    def is_valid_action(self, action):
        """Check if action is valid (0 or 1)."""
        return action in self.get_available_actions()

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

