import sys
import os
import random
import gymnasium as gym
from .base_env_adaptor import BaseEnvAdaptor
from .adopter_util import frozenlake_goal_positions, choose_format_full_prompt
from .env_config import frozenlake_config
from utils import get_timestamp_ms
import re
from .adaptor_prompt_factory import build_frozenlake_user_prompt, FROZENLAKE_SYSTEM_PROMPT

class FrozenLakeAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name, model_name, desc=None, goal_rewards=None):
        super().__init__(env_name, model_name)
        seed = frozenlake_config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        desc_to_use = desc if desc is not None else frozenlake_config.get('desc')
        self.env = gym.make(
            frozenlake_config['id'],
            desc=desc_to_use,
            map_name=frozenlake_config['map_name'],
            is_slippery=frozenlake_config['is_slippery'],
            success_rate=frozenlake_config['success_rate'],
            reward_schedule=frozenlake_config['reward_schedule'],
            max_episode_steps=frozenlake_config['max_episode_steps']
        )
        self.reproduce_method = "action_path"

        # change for every exploration
        self.destinations = []
        self.destination_label = None
        # optional custom rewards per goal coordinate, e.g. {(r, c): 0.5, (r2, c2): 1.0}
        if goal_rewards is None:
            goals = frozenlake_goal_positions(desc_to_use)
            if len(goals) != 1:
                raise ValueError(
                    f"goal_rewards is None, so desc must contain exactly one 'G', "
                    f"but found {len(goals)} goal(s): {goals}"
                )
            self.goal_rewards = {goals[0]: 1.0}
        else:
            self.goal_rewards = goal_rewards
            if not any(value == 1.0 for value in self.goal_rewards.values()):
                raise ValueError("goal_rewards must include at least one reward equal to 1.0")

        # history records
        self.action_path = []
        self.st = None
        self.prev_action = None
        self.st1 = None
        self.terminated = False
        self.reward = None

        self.format_full_prompt = choose_format_full_prompt(model_name)


    def initialize_env(self):
        self.env.reset()
        # set destinations (allow multiple goals)
        self.destinations = self._find_destinations()
        # label for logging/experience id
        self.destination_label = "|".join([f"{r}-{c}" for r, c in self.destinations])
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
        # self.reward = step_metadata[1]
        terminated = step_metadata[2]
        self.terminated = terminated
        truncated = step_metadata[3]
        # If landed on a goal, override reward by coordinate when provided
        cur_pos = self.get_cur_pos()
        map_data = self._get_map()
        tile = map_data[cur_pos[0]][cur_pos[1]]
        if tile == 'G':
            self.reward = self.goal_rewards.get(tuple(cur_pos))

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
            # allow custom goal rewards (e.g., 0.5 or 1.0)
            if self.terminated:
                assert self.reward is not None
                assert self.reward > 0
                return self.reward
            else:
                raise ValueError(f"Reward problem: terminated: {self.terminated}, reward: {self.reward}")
        else:
            if self.reward is None:
                return 0
            if self.reward > 0:
                raise ValueError(f"Reward cannot be 1 when not on the goal: terminated: {self.terminated}, reward: {self.reward}")
            return 0

    # Getters
    def get_env_description(self):
        map_list = self._get_map()
        map_str = '\n'.join([''.join(row) for row in map_list])
        return f"""-----
Init new environment:
[ENV] FrozenLake
[Destinations]: {self.destinations}
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
        if state['tile_type'] == 'G':
            # Attach final score when on a goal tile for downstream consumers
            state['score'] = self.extract_reward_score()

        return state

    def get_cur_pos(self):
        flat_pos = int(self.env.unwrapped.s)
        return [flat_pos // self.env.unwrapped.ncol, flat_pos % self.env.unwrapped.ncol]

    def get_available_actions(self):
        return [0, 1, 2, 3]

    def get_action_path(self):
        return self.action_path.copy()

    def get_experience(self):
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[frozenLake_adaptor] In get_experience(), the history is not set, one of st0, a or st1 is None")
        experience = {
            "id": f"{get_timestamp_ms()}_{self.destination_label}_{'-'.join(map(str, self.action_path))}",
            "reproduce_method": self.reproduce_method,
            "action_path": self.get_action_path(),  # Use copy() to avoid reference issue
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience

    def get_instruction(self):
        return f"Destinations: {self.destinations}; Map: {self._get_map()}"

    def is_valid_action(self, action):
        return action in self.get_available_actions()

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

    def _find_destinations(self):
        """
        Find all goal positions 'G' in the FrozenLake map.
        Returns the positions as (row, col) tuples.
        Raises ValueError if there is no 'G' in the map.
        """
        map = self._get_map()
        goal_positions = []
        for row_idx, row in enumerate(map):
            for col_idx, cell in enumerate(row):
                if cell == 'G':
                    goal_positions.append((row_idx, col_idx))

        if len(goal_positions) == 0:
            raise ValueError("No goal 'G' found in the FrozenLake map")
        # Ensure provided goal_rewards keys exactly match found goals (if provided)
        if self.goal_rewards:
            reward_keys = set(self.goal_rewards.keys())
            found_keys = set(goal_positions)
            if reward_keys != found_keys:
                raise ValueError(f"goal_rewards keys {reward_keys} do not match goal positions {found_keys}")
        return goal_positions

    def format_action(self, action):
        """
        Format the action from LLM output to valid action integer.
        Extracts exactly one digit (0-3) from the response.
        Raises error if zero or multiple valid actions found.
        """
        action = action.strip()
        # Find all numbers in the range 0-3
        matches = re.findall(r'[0-3]', action)
        if len(matches) == 0:
            raise ValueError(f"Could not extract valid action (0-3) from: {action}")
        elif len(matches) > 1:
            print(f"[PROBLEM] Multiple actions found ({matches}) in: {action}")
        return int(matches[0])

    # get action prompt
    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
        state = self.get_state()
        user_prompt = build_frozenlake_user_prompt(
            state=state,
            available_actions=self.get_available_actions(),
            destinations=self.destinations,
            goal_rewards=self.goal_rewards,
            map_rows=self.env.unwrapped.nrow,
            map_cols=self.env.unwrapped.ncol,
            retrieved_experiences=retrieved_experiences,
        )
        
        # Construct the prompt in Llama3 format
        prompt = self.format_full_prompt(FROZENLAKE_SYSTEM_PROMPT, user_prompt)
        return prompt
