from .webshop_exp_backend import WebshopExpBackend
from .backend_config import webshop_vanilla_config
from utils import log_flush
from env_adaptors.webshop_adaptor import WebshopAdaptor
from collections import deque
import copy

class WebshopExpVanillaBackend(WebshopExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, log_dir=None):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        self.algorithm = webshop_vanilla_config["algorithm"]
        self.max_bfs_depth = webshop_vanilla_config.get("max_bfs_depth", 10)
        
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
            if WebshopAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences, results ids: {[exp['id'] for exp in results]}")        
        return results

    # ============ BFS max_score methods ============
    
    def _is_search_page(self, state: dict) -> bool:
        """
        Check if the state is the search page (the starting point).
        
        The search page URL pattern: http://host/<session_id>
        - parts: ['http:', '', 'host', '<session_id>']
        - len(parts) == 4, meaning no further path after session_id
        
        This is where "back to search" leads to, and from here any search can be done.
        """
        url = state.get('url', '')
        parts = url.split('/')
        # Search page has exactly 4 parts: ['http:', '', 'host', '<session_id>']
        # Other pages have more parts like /item_page/, /search_results/, etc.
        if len(parts) == 4:
            return True
        return False

    def _get_state_key(self, state: dict) -> str:
        """Generate a unique key for a state to track visited states."""
        return state.get('url', '')

    def _compute_max_score_bfs(self, exp: dict) -> float:
        """
        Use BFS to compute the maximum achievable score starting from this experience.
        
        Rules:
        - Add exp's st to visited to prevent cycles back to start
        - Skip search pages to avoid cycles
        - Track visited states to avoid infinite loops
        - Respect max_bfs_depth to limit search depth
        """
        # Check if exp's st1 is already a finished state
        is_finished, score = WebshopAdaptor.check_finished_and_get_score(exp['st1'])
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
        
        # Skip if starting state is a search page
        if self._is_search_page(start_state):
            return None
        
        queue.append((start_state, 0))
        visited.add(start_key)
        
        while queue:
            current_state, current_depth = queue.popleft()
            
            if current_depth >= self.max_bfs_depth:
                continue
            
            for exp_id in self.exp_store.keys():
                next_exp = self.exp_store[exp_id]
                
                if not WebshopAdaptor.two_states_equal(current_state, next_exp['st']):
                    continue
                
                next_state = next_exp['st1']
                next_key = self._get_state_key(next_state)
                
                if next_key in visited:
                    continue
                
                if self._is_search_page(next_state):
                    continue
                
                is_finished, score = WebshopAdaptor.check_finished_and_get_score(next_state)
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
        
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if WebshopAdaptor.two_states_equal(state, exp['st']):
                exp_with_score = copy.deepcopy(exp)
                max_score = self._compute_max_score_bfs(exp)
                exp_with_score['max_score'] = max_score
                results.append(exp_with_score)
        
        log_flush(self.logIO, f"[BFS] Retrieved {len(results)} experiences, ids: {[exp['id'] for exp in results]}")
        log_flush(self.logIO, f"[BFS] Max scores: {[(exp['id'], exp.get('max_score')) for exp in results]}")
        
        return results
