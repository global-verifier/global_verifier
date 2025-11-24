import sys
import os
import random
import gymnasium as gym
from .base_env_adaptor import BaseEnvAdaptor
from .env_config import frozenlake_config
from utils import get_timestamp_ms

class FrozenLakeAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        seed = frozenlake_config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        self.env = gym.make(
            frozenlake_config['id'],
            desc=frozenlake_config['desc'],
            map_name=frozenlake_config['map_name'],
            is_slippery=frozenlake_config['is_slippery'],
            success_rate=frozenlake_config['success_rate'],
            reward_schedule=frozenlake_config['reward_schedule'],
            max_episode_steps=frozenlake_config['max_episode_steps']
        )
        self.reproduce_method = "action_path"

        # change for every exploration
        self.destination = None

        # history records
        self.action_path = []
        self.st = None
        self.prev_action = None
        self.st1 = None
        self.terminated = False
        self.reward = None

    def initialize_env(self):
        self.env.reset()
        # set destination
        self.destination = self._find_destination()
        # set new history
        self.st = None
        self.prev_action = None
        self.st1 = self.get_state()
        self.action_path = []
        self.terminated = False
        self.reward = None

    def step(self, action):
        # record history
        self.st = self.st1
        self.prev_action = action
        self.action_path.append(action)
        # make history
        step_metadata = self.env.step(action)
        self.consume_step(step_metadata)
        # observe history
        self.st1 = self.get_state()

    def consume_step(self, step_metadata):
        pos = step_metadata[0]
        self.reward = step_metadata[1]
        terminated = step_metadata[2]
        self.terminated = terminated
        truncated = step_metadata[3]

    def is_finished_state(self, state):
        """
        Check if the state is a terminal state (on H or G).
        Args:
            state: dict with 'cur_pos'
        Returns:
            True if on H (hole) or G (goal), False otherwise
        """
        cur_pos = state['cur_pos']
        assert cur_pos is not None and cur_pos == self.get_cur_pos()
        map_data = self._get_map()
        tile = map_data[cur_pos[0]][cur_pos[1]]
        assert tile is not None and tile == state['tile_type']
        is_finished = tile == 'H' or tile == 'G'
        assert is_finished == self.terminated, print(f"tile: {tile}, is_finished: {is_finished}, terminated: {self.terminated}")
        return is_finished

    def extract_reward_score(self):
        cur_pos = self.get_cur_pos()
        map_data = self._get_map()
        tile = map_data[cur_pos[0]][cur_pos[1]]
        if tile == 'G':
            if self.terminated and self.reward == 1:
                return 1
            else:
                raise ValueError(f"Reward problem: terminated: {self.terminated}, reward: {self.reward}")
        else:
            if self.reward == 1:
                raise ValueError(f"Reward cannot be 1 when not on the goal: terminated: {self.terminated}, reward: {self.reward}")
            return 0

    # Getters
    def get_env_description(self):
        map_list = self._get_map()
        map_str = '\n'.join([''.join(row) for row in map_list])
        return f"""-----
Init new environment:
[ENV] FrozenLake
[Destination]: {self.destination}
[Map]:
{map_str}
-----"""

    def get_state(self):
        state = {}
        cur_pos = self.get_cur_pos()
        state['cur_pos'] = cur_pos
        map_list = self._get_map()
        state['tile_type'] = map_list[cur_pos[0]][cur_pos[1]]
        # TODO: whether to put destination into the state
        return state

    def get_experience(self):
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[webshop_adaptor] In get_experience(), the history is not set, one of st0, a or st1 is None")
        experience = {
            "id": f"{get_timestamp_ms()}_{self.url_id}_{'-'.join(self.action_path)}",
            "reproduce_method": "action_path",
            "action_path": self.get_action_path(),  # Use copy() to avoid reference issue
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience

    def get_cur_pos(self):
        flat_pos = int(self.env.unwrapped.s)
        return (flat_pos // self.env.unwrapped.ncol, flat_pos % self.env.unwrapped.ncol)

    def get_available_actions(self):
        return [0, 1, 2, 3]

    def get_action_path(self):
        return self.action_path.copy()

    def get_experience(self):
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[webshop_adaptor] In get_experience(), the history is not set, one of st0, a or st1 is None")
        experience = {
            "id": f"{get_timestamp_ms()}_{self.destination}_{'-'.join(self.action_path)}",
            "reproduce_method": self.reproduce_method,
            "action_path": self.get_action_path(),  # Use copy() to avoid reference issue
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience

    def get_instruction(self):
        return f"Destination: {self.destination}; Map: {self._get_map()}"

    def is_valid_action(self, action):
        return action in self.get_available_actions()

    def reconstruct_state(self, exp):
        """Reconstruct the state from the experience."""
        assert exp['action'] == exp['action_path'][-1]
        assert len(exp['action_path']) > 1
        self.initialize_env()
        try:
            for i in range(len(exp['action_path']) - 1):
                action = exp['action_path'][i]
                self.step(action)
        except Exception as e:
            return False, e
        if self.get_state() != exp['st']:
            return False, f"Reconstructed state differs, expected: {exp['st']}, got: {self.get_state()}"
        return True, None

    # interal helper functions
    def _get_map(self):
        """
        Get the map as a list of lists of strings.
        Converts numpy array with byte strings to list of lists of regular strings.
        """ 
        desc = self.env.unwrapped.desc 
        map = []
        for row in desc:
            row_list = [cell.decode('utf-8') for cell in row]
            map.append(row_list)
        return map

    def _find_destination(self):
        """
        Find the goal position 'G' in the FrozenLake map.
        Returns the position as (row, col) tuple.
        Raises ValueError if there is not exactly one 'G' in the map.
        """
        map = self._get_map()
        goal_positions = []
        for row_idx, row in enumerate(map):
            for col_idx, cell in enumerate(row):
                if cell == 'G':
                    goal_positions.append((row_idx, col_idx))
        
        if len(goal_positions) == 0:
            raise ValueError("No goal 'G' found in the FrozenLake map")
        elif len(goal_positions) > 1:
            raise ValueError(f"Multiple goals 'G' found in the map at positions: {goal_positions}")
        return goal_positions[0]        
