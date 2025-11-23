from .base_exp_backend import BaseExpBackend
from env_adaptors.webshop_adaptor import WebshopAdaptor
from utils import log_flush
from utils import get_timestamp
import json

class WebshopExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)

    # Setters

    # Consultants

    def _has_conflict(self, e1, e2) -> bool:
        """
        Check if two experiences have conflict.
        - if same st, action
            - different st1
        # == actually works for dict, did not know that before
        """
        return WebshopAdaptor.has_conflict(e1, e2)

    def _are_same_exp(self, e1, e2) -> bool:
        """
        Check if two experiences are the same.
        """
        return WebshopAdaptor.are_same_exp(e1, e2)

    def get_most_optmized_path_exp_id(self, exp_id_group: set) -> str:
        """
        Given an iterable of experiences, return the ID whose action_path is shortest.
        """
        if not exp_id_group or len(exp_id_group) == 0:
            raise ValueError("exp_id_group is empty")

        best_id = None
        best_len = None
        for exp_id in exp_id_group:
            exp = self.exp_store[exp_id]
            action_path = exp.get("action_path")
            path_len = len(action_path)
            if best_len is None or path_len < best_len:
                best_id = exp_id
                best_len = path_len
        return best_id

    def resolve_experience_conflict(self, **kwargs):
        """
        Resolve the conflict between two experiences.
        
        Scenarios (9 cases based on 4 boolean flags):
        1. (T, T, T, T): Both reproducible - Non-deterministic behavior, keep both but log warning
        2. (T, T, T, F): e0 valid, e1 outdated - Deprecate e1
        3. (T, F, T, T): e0 outdated, e1 valid - Deprecate e0
        4. (T, F, T, F): Both outdated - Deprecate both
        5. (T, T, F, F): e0 valid, e1's st invalid - Deprecate e1
        6. (T, F, F, F): Both problematic - Deprecate both
        7. (F, F, T, T): e0's st invalid, e1 valid - Deprecate e0
        8. (F, F, T, F): Both problematic - Deprecate both
        9. (F, F, F, F): Both invalid - Deprecate both
        
        Decision logic:
        - If e0_st1_success and not e1_st1_success: deprecate e1
        - If e1_st1_success and not e0_st1_success: deprecate e0
        - If both st1 fail: deprecate both
        - If both st1 succeed: log warning (non-deterministic)
        """
        conflict_pair_id = kwargs['conflict_pair_id']
        examine_result = kwargs['examine_result']
        e0_st_success, e0_st1_success, e1_st_success, e1_st1_success = examine_result

        e0_id = conflict_pair_id[0]
        e1_id = conflict_pair_id[1]

        assert self._exp_exist(e0_id), f"Experience {e0_id} not found"
        assert self._exp_exist(e1_id), f"Experience {e1_id} not found"
        
        # If st False, depreciate
        if not e0_st_success:
            # 7, 8, 9
            self._deprecate_experience(e0_id)
        if not e1_st_success:
            # 5, 6, 9
            self._deprecate_experience(e1_id)
        
        # Set the id to None if depreciated
        if self._exp_is_depreciated(e0_id):
            log_flush(self.logIO, f"WARNING: Experience {e0_id} already in deprecated store")
            e0 = self.depreiciate_exp_store[e0_id]
        else:
            e0 = self.exp_store[e0_id]
        if self._exp_is_depreciated(e1_id):
            log_flush(self.logIO, f"WARNING: Experience {e1_id} already in deprecated store")
            e1 = self.depreiciate_exp_store[e1_id]
        else:
            e1 = self.exp_store[e1_id]

        # Assert conflict exists
        assert self._has_conflict(e0, e1)

        # Assert st1 can only be True if st is True
        if not e0_st_success and e0_st1_success:
            raise ValueError(f"Experience {e0_id} st is False but st1 is True")
        if not e1_st_success and e1_st1_success:
            raise ValueError(f"Experience {e1_id} st is False but st1 is True")
        
        # Cannot accept all True, has to depriate one
        if e0_st1_success and e1_st1_success:
            log_flush(self.logIO, f"WARNING: Non-deterministic behavior detected!")
            log_flush(self.logIO, f"  Both {e0_id} and {e1_id} are reproducible but lead to different st1")
            log_flush(self.logIO, f"  Cannot accept, report ERROR")
            raise ValueError(f"(T, T, T, T) All True, Non-deterministic behavior detected!")


        # Start checking st1
        if e0_st_success:
            if not e0_st1_success:
                self._deprecate_experience(e0_id)
        if e1_st_success:
            if not e1_st1_success:
                self._deprecate_experience(e1_id)

        log_flush(self.logIO, f"Conflict resolution completed for {e0_id} and {e1_id}")

    def _exp_exist(self, exp_id):
        return exp_id in self.exp_store or exp_id in self.depreiciate_exp_store

    def _exp_is_depreciated(self, exp_id):
        """Check if an experience is depreciated."""
        if exp_id in self.depreiciate_exp_store:
            assert exp_id not in self.exp_store
            return True
        return False
    
    def _deprecate_experience(self, exp_id):
        """Move an experience from exp_store to deprecated_exp_store."""
        if exp_id in self.exp_store and exp_id in self.depreiciate_exp_store:
            raise ValueError(f"Experience {exp_id} in both store and deprecated store")
        elif exp_id in self.depreiciate_exp_store:
            log_flush(self.logIO, f"  Experience {exp_id} already in deprecated store")
            return
        elif exp_id in self.exp_store:
            self.depreiciate_exp_store[exp_id] = self.exp_store[exp_id]
            self.depreiciate_exp_store[exp_id]['deprecated_at'] = get_timestamp()
            del self.exp_store[exp_id]
            log_flush(self.logIO, f"[DEPRECIATE] Experience {exp_id} moved to deprecated store")
            self.save_store()
        else:
            raise ValueError(f"Experience {exp_id} not found in in any store")
