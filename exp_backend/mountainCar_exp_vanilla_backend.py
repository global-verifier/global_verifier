from .mountainCar_exp_backend import MountainCarExpBackend
from .backend_config import mountaincar_vanilla_config
from utils import log_flush
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from collections import deque
import copy

class MountainCarExpVanillaBackend(MountainCarExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, log_dir=None):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        self.algorithm = mountaincar_vanilla_config["algorithm"]
        self.max_bfs_depth = mountaincar_vanilla_config.get("max_bfs_depth", 200)
        self.goal_position = mountaincar_vanilla_config.get("goal_position", 0.5)
        
        # Set the retrieve experience method
        log_flush(self.logIO, f"Using algorithm: {self.algorithm}")
        if self.algorithm == "sameSt_1Step":
            self.retrieve_experience = self.retrieve_experience_sameSt_1Step
        elif self.algorithm == "sameSt_bfs_reachable":
            self.retrieve_experience = self.retrieve_experience_sameSt_bfs_reachable
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm} is not supported.")

    def retrieve_experience_sameSt_1Step(self, state) -> list:
        """
        Retrieve experiences with exact state match.
        For fine-grained numerical states (position: 3 decimals, velocity: 4 decimals).
        """
        results = []
        log_flush(self.logIO, f"Retrieving experience for exact state: {state}")
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences, ids: {[exp['id'] for exp in results]}")        
        return results

    # ============ BFS Reachability Methods ============
    
    def _get_state_key(self, state: dict) -> str:
        """Generate a unique key for a state to track visited states."""
        return f"{state['position']}_{state['velocity']}"
    
    def _is_goal_state(self, state: dict) -> bool:
        """Check if state has reached the goal (position >= 0.5)."""
        return state.get('position', -1.2) >= self.goal_position
    
    def _compute_reachable_path_bfs(self, exp: dict) -> dict:
        """
        Use BFS to find if the goal is reachable from this experience.
        
        Returns:
            dict with:
            - 'reachable': bool - whether goal can be reached
            - 'path_length': int - number of steps to reach goal (None if not reachable)
            - 'path': list - sequence of (state, action) pairs to reach goal
        """
        # Check if exp's st1 is already a goal state
        if self._is_goal_state(exp['st1']):
            return {
                'reachable': True,
                'path_length': 1,
                'path': [(exp['st'], exp['action'], exp['st1'])]
            }
        
        visited = set()
        queue = deque()
        
        # Add the original state (exp's st) to visited to prevent cycles back
        origin_key = self._get_state_key(exp['st'])
        visited.add(origin_key)
        
        start_state = exp['st1']
        start_key = self._get_state_key(start_state)
        
        # Initial path: [(st, action, st1)]
        initial_path = [(exp['st'], exp['action'], exp['st1'])]
        
        queue.append((start_state, 0, initial_path))
        visited.add(start_key)
        
        while queue:
            current_state, current_depth, current_path = queue.popleft()
            
            if current_depth >= self.max_bfs_depth:
                continue
            
            # Find all experiences starting from current_state
            for exp_id, next_exp in self.exp_store.items():
                if not BaseEnvAdaptor.two_states_equal(current_state, next_exp['st']):
                    continue
                
                next_state = next_exp['st1']
                next_key = self._get_state_key(next_state)
                
                if next_key in visited:
                    continue
                
                # Build new path
                new_path = current_path + [(next_exp['st'], next_exp['action'], next_exp['st1'])]
                
                # Check if reached goal
                if self._is_goal_state(next_state):
                    return {
                        'reachable': True,
                        'path_length': len(new_path),
                        'path': new_path
                    }
                
                visited.add(next_key)
                queue.append((next_state, current_depth + 1, new_path))
        
        # Goal not reachable within max_bfs_depth
        return {
            'reachable': False,
            'path_length': None,
            'path': None
        }
    
    def retrieve_experience_sameSt_bfs_reachable(self, state) -> list:
        """
        Retrieve experiences with reachability info computed via BFS.
        
        Each returned experience includes:
        - 'reachable': bool - whether goal can be reached from this experience
        - 'path_length': int - steps to reach goal (None if not reachable)
        - 'path': list - the path to goal (for exploration/debugging)
        """
        results = []
        log_flush(self.logIO, f"[BFS] Retrieving experience for state: {state}")
        
        for exp_id, exp in self.exp_store.items():
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                exp_with_info = copy.deepcopy(exp)
                reachability = self._compute_reachable_path_bfs(exp)
                exp_with_info['reachable'] = reachability['reachable']
                exp_with_info['path_length'] = reachability['path_length']
                exp_with_info['path_to_goal'] = reachability['path']
                results.append(exp_with_info)
        
        # Sort by: reachable first, then by shortest path
        results.sort(key=lambda x: (not x['reachable'], x['path_length'] or float('inf')))
        
        # Log results
        log_flush(self.logIO, f"[BFS] Retrieved {len(results)} experiences")
        for exp in results:
            if exp['reachable']:
                log_flush(self.logIO, f"  ✅ {exp['id']}: reachable in {exp['path_length']} steps")
                # Log the action sequence
                if exp['path_to_goal']:
                    actions = [step[1] for step in exp['path_to_goal']]
                    log_flush(self.logIO, f"     Action sequence: {actions}")
            else:
                log_flush(self.logIO, f"  ❌ {exp['id']}: NOT reachable within {self.max_bfs_depth} steps")
        
        return results
    
    def explore_from_state(self, state) -> dict:
        """
        Explore all possible paths from a given state and report reachability.
        
        This is a utility method for analysis/debugging.
        
        Returns:
            dict with exploration results including all reachable paths.
        """
        log_flush(self.logIO, f"\n{'='*60}")
        log_flush(self.logIO, f"[EXPLORER] Starting exploration from state: {state}")
        log_flush(self.logIO, f"{'='*60}")
        
        experiences = self.retrieve_experience_sameSt_bfs_reachable(state)
        
        reachable_exps = [e for e in experiences if e['reachable']]
        unreachable_exps = [e for e in experiences if not e['reachable']]
        
        result = {
            'start_state': state,
            'total_experiences': len(experiences),
            'reachable_count': len(reachable_exps),
            'unreachable_count': len(unreachable_exps),
            'reachable_experiences': reachable_exps,
            'unreachable_experiences': unreachable_exps,
        }
        
        # Print summary
        log_flush(self.logIO, f"\n[EXPLORER] Summary:")
        log_flush(self.logIO, f"  Total experiences from this state: {result['total_experiences']}")
        log_flush(self.logIO, f"  Reachable to goal: {result['reachable_count']}")
        log_flush(self.logIO, f"  Not reachable: {result['unreachable_count']}")
        
        if reachable_exps:
            best = reachable_exps[0]  # Already sorted by path_length
            log_flush(self.logIO, f"\n[EXPLORER] Best path (shortest):")
            log_flush(self.logIO, f"  Experience ID: {best['id']}")
            log_flush(self.logIO, f"  Path length: {best['path_length']} steps")
            log_flush(self.logIO, f"  First action: {best['action']}")
            if best['path_to_goal']:
                actions = [step[1] for step in best['path_to_goal']]
                log_flush(self.logIO, f"  Full action sequence: {actions}")
                # Print state progression
                log_flush(self.logIO, f"  State progression:")
                for i, (st, action, st1) in enumerate(best['path_to_goal']):
                    action_name = ['LEFT', 'COAST', 'RIGHT'][action]
                    log_flush(self.logIO, f"    Step {i+1}: pos={st['position']:.3f}, vel={st['velocity']:.4f} --[{action_name}]--> pos={st1['position']:.3f}, vel={st1['velocity']:.4f}")
        
        log_flush(self.logIO, f"{'='*60}\n")
        
        return result

