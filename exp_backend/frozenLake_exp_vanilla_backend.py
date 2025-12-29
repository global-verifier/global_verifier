from .frozenLake_exp_backend import FrozenLakeExpBackend
from .backend_config import frozenlake_vanilla_config
from utils import log_flush
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from collections import deque
import copy
import json

class FrozenLakeExpVanillaBackend(FrozenLakeExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, log_dir=None):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        self.algorithm = frozenlake_vanilla_config["algorithm"]
        self.max_bfs_depth = frozenlake_vanilla_config.get("max_bfs_depth", 10)
        
        # set the retrieve experience
        log_flush(self.logIO, f"Using algorithm: {self.algorithm}")
        if self.algorithm == "sameSt_1Step":
            self.retrieve_experience = self.retrieve_experience_sameSt_1Step
        elif self.algorithm == "sameSt_bfs_maxScore":
            self.retrieve_experience = self.retrieve_experience_sameSt_bfs_maxScore
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm} is not supported.")

    def retrieve_experience_sameSt_1Step(self, state) -> list:
        results = []
        log_flush(self.logIO, f"Retrieving experience for state: {state}")
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences, results ids: {[exp['id'] for exp in results]}")        
        return results

    # ============ BFS max_score methods ============

    def _get_state_key(self, state: dict) -> str:
        """Generate a unique key for a state to track visited states."""
        # For FrozenLake, state is defined by position. 
        cur_pos = state.get('cur_pos')
        if cur_pos is None:
            return "unknown"
        return f"{cur_pos[0]}_{cur_pos[1]}"

    def _check_finished_and_get_score(self, state: dict):
        """
        Check if the state is a terminal state and return the score.
        """
        tile = state.get('tile_type')
        if tile == 'G':
            # Adaptor puts score in state if tile is G
            return True, state.get('score')
        elif tile == 'H':
            return True, 0.0
        return False, 0.0

    def _compute_max_score_bfs(self, exp: dict) -> float:
        """
        Use BFS to compute the maximum achievable score starting from this experience.
        """
        # Check if exp's st1 is already a finished state
        is_finished, score = self._check_finished_and_get_score(exp['st1'])
        if is_finished:
            return score
        
        max_score = None
        visited = set()
        queue = deque()
        
        # Add the original state (exp's st) to visited to prevent cycles back to start
        origin_key = self._get_state_key(exp['st'])
        visited.add(origin_key)
        
        start_state = exp['st1']
        start_key = self._get_state_key(start_state)
        
        queue.append((start_state, 0))
        visited.add(start_key)
        
        while queue:
            current_state, current_depth = queue.popleft()
            
            if current_depth >= self.max_bfs_depth:
                continue
            
            # Iterate all experiences to find transitions
            for exp_id in self.exp_store.keys():
                next_exp = self.exp_store[exp_id]
                
                # Check if this experience starts from current_state
                if not BaseEnvAdaptor.two_states_equal(current_state, next_exp['st']):
                    continue
                
                next_state = next_exp['st1']
                next_key = self._get_state_key(next_state)
                
                if next_key in visited:
                    continue
                
                is_finished, score = self._check_finished_and_get_score(next_state)
                if is_finished:
                    if max_score is None or score > max_score:
                        max_score = score
                    visited.add(next_key)
                    continue
                
                visited.add(next_key)
                queue.append((next_state, current_depth + 1))
        
        return max_score

    def retrieve_experience_sameSt_bfs_maxScore(self, state) -> list:
        """
        Retrieve experiences with max_score attribute computed via BFS.
        max_score is the maximum achievable score if following this experience.
        """
        results = []
        log_flush(self.logIO, f"[BFS] Retrieving experience for state: {state}")
        
        # Cache st1 max scores to avoid recalculating for the same start point
        st1_cache = {}

        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                exp_with_score = copy.deepcopy(exp)
                
                st1_key = self._get_state_key(exp['st1'])
                if st1_key in st1_cache:
                    max_score = st1_cache[st1_key]
                else:
                    max_score = self._compute_max_score_bfs(exp)
                    st1_cache[st1_key] = max_score
                
                exp_with_score['max_score'] = max_score
                results.append(exp_with_score)
        
        log_flush(self.logIO, f"[BFS] Retrieved {len(results)} experiences, ids: {[exp['id'] for exp in results]}")
        log_flush(self.logIO, f"[BFS] Max scores: {[(exp['id'], exp.get('max_score')) for exp in results]}")
        
        return results