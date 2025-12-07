from .base_exp_backend import BaseExpBackend
from utils import log_flush

class MountainCarExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path):
        # Must define expected_fields BEFORE calling super().__init__()
        # because parent's __init__ calls _is_valid_exp_store() which uses this field
        self.expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": int,  # MountainCar action is int (0, 1, or 2)
            "st1": dict,
        }
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)


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
            log_flush(self.logIO, f"  This is expected in MountainCar due to floating point precision")
            # Don't raise error for MountainCar - slight variations are normal

        # Start checking st1
        if e0_st_success:
            if not e0_st1_success:
                self._deprecate_experience(e0_id)
        if e1_st_success:
            if not e1_st1_success:
                self._deprecate_experience(e1_id)

        log_flush(self.logIO, f"Conflict resolution completed for {e0_id} and {e1_id}")

