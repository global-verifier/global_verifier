import json
import os
import itertools
from typing import Any, Dict, Iterable, List
from utils import get_timestamp
from .backend_config import base_backend_config
from utils import log_flush

class BaseExpBackend:
    def __init__(self, env_name: str, storage_path: str ="./storage/exp_store.json", depreiciate_exp_store_path: str ="./storage/depreiciate_exp_store.json") -> None:
        self.env_name = env_name
        self.storage_path = storage_path
        self.depreiciate_exp_store_path = depreiciate_exp_store_path

        # Add the logger (must be created before _load_store() since it uses logIO)
        log_dir = base_backend_config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.logIO = open(f'{log_dir}/exp_backendLog_{get_timestamp()}.log', 'a')
        
        self.exp_store = self._load_store(self.storage_path)
        self.depreiciate_exp_store = self._load_store(self.depreiciate_exp_store_path)
        if not self._is_valid_exp_store():
            raise ValueError(f"Invalid experience store, did not pass the validation.")

        # Check no overlap between stores
        self.check_stores_no_overlap()

    def _load_store(self, exp_storage_path):
        log_flush(self.logIO, f"########################################################")
        log_flush(self.logIO, f"Start loading {exp_storage_path} the store at {get_timestamp()}")
        if exp_storage_path is None:
            raise ValueError("storage_path is None, please provide a valid path.")

        # Start loading the store
        exp_store = {}
        try:
            storage_path = os.fspath(exp_storage_path)
        except TypeError as exc:
            raise TypeError("storage_path must be a path-like object or string.") from exc
        # Make sure point to a file a path for a new file to be created
        if os.path.isdir(storage_path):
            raise ValueError(f"Storage path '{storage_path}' is a directory; expected a file.")
        # If not exists, create a new file
        if not os.path.exists(storage_path):
            with open(storage_path, 'w') as file:
                json.dump({}, file)
            log_flush(self.logIO, f"Store not exists, creating a new one at {storage_path}, at {get_timestamp()}")
            return exp_store

        # Acutally exists, load the store
        with open(storage_path, 'r') as file:
            exp_store = json.load(file)
        log_flush(self.logIO, f"Successfully loaded the store, path: {storage_path}, size: {len(exp_store)}, at {get_timestamp()}")
        return exp_store

    def save_store(self) -> None:
        """
        Store the experience store to the storage path.
        """
        # Validate no overlap before saving
        self.check_stores_no_overlap()
        
        with open(self.storage_path, 'w') as file:
            json.dump(self.exp_store, file)
        log_flush(self.logIO, f"Save store, path: {self.storage_path}, size: {len(self.exp_store)}, at {get_timestamp()}")
        with open(self.depreiciate_exp_store_path, 'w') as file:
            json.dump(self.depreiciate_exp_store, file)
        log_flush(self.logIO, f"Save depreiciate store, path: {self.depreiciate_exp_store_path}, size: {len(self.depreiciate_exp_store)}, at {get_timestamp()}")

    def _loop_detect_exp_conflict(self):
        """
        Detect all the conflict pairs in the experience store.
        """
        log_flush(self.logIO, f"Loop detecting conflict pairs")
        conflict_pairs = []
        exp_ids = self.exp_store.keys()
        exp_id_combinations = list(itertools.combinations(exp_ids, 2))
        log_flush(self.logIO, f"Number of experience combinations: {len(exp_id_combinations)}")
        for i in range(len(exp_id_combinations)):
            exp_pair = exp_id_combinations[i]
            assert len(exp_pair) == 2
            log_flush(self.logIO, f"{i}/{len(exp_id_combinations)}: {exp_pair[0]} and {exp_pair[1]})")
            if self._has_conflict(self.exp_store[exp_pair[0]], self.exp_store[exp_pair[1]]):
                log_flush(self.logIO, f"Conflict pair detected: {exp_pair[0]} and {exp_pair[1]}")
                conflict_pairs.append(exp_pair)
        log_flush( self.logIO, f"Loop finish, num conflict pairs detected: {len(conflict_pairs)}")
        return conflict_pairs

    # Getters
    def get_exp_by_id(self, exp_id) -> dict:
        return self.exp_store[exp_id]

    # Validators
    def check_stores_no_overlap(self):
        """
        Check that exp_store and depreiciate_exp_store have no overlapping keys.
        
        Returns:
            True if no overlap, False otherwise
        
        Raises:
            ValueError if there are overlapping keys
        """
        exp_store_keys = set(self.exp_store.keys())
        depreiciate_store_keys = set(self.depreiciate_exp_store.keys())
        overlap = exp_store_keys & depreiciate_store_keys
        
        if overlap:
            error_msg = f"Found {len(overlap)} overlapping keys between exp_store and depreiciate_exp_store: {list(overlap)}"
            log_flush(self.logIO, f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        log_flush(self.logIO, f"Store validation passed: No overlap between exp_store ({len(exp_store_keys)} items) and depreiciate_exp_store ({len(depreiciate_store_keys)} items)")

    def get_redundant_experience_groups(self) -> list:
        """Get the redundant experience ids."""
        log_flush(self.logIO, f"Loop detecting redundant experiences")
        redundant_experience_groups = []
        exp_ids = list(self.exp_store.keys())
        grouped_exp_ids = set()
        for i in range(len(exp_ids)):
            cur_redundant_group = set()
            cur_id = exp_ids[i]
            if cur_id in grouped_exp_ids:
                continue
            grouped_exp_ids.add(cur_id)
            cur_redundant_group.add(cur_id)
            for j in range(i+1, len(exp_ids)):
                test_id = exp_ids[j]
                if test_id in grouped_exp_ids:
                    continue
                if self._are_same_exp(self.exp_store[cur_id], self.exp_store[test_id]):
                    cur_redundant_group.add(cur_id)
                    cur_redundant_group.add(test_id)
                    grouped_exp_ids.add(test_id)
            if len(cur_redundant_group) > 1:
                redundant_experience_groups.append(cur_redundant_group)
        log_flush(self.logIO, f"Loop finish, num redundant experience groups detected: {len(redundant_experience_groups)}")
        return redundant_experience_groups

    # To be implemented in the subclass
    def _is_valid_exp(self, exp) -> bool:
        """
        Check if an experience is valid.
        """
        if not isinstance(exp, dict):
            return False
        expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": str,
            "st1": dict,
        }
        for field, field_type in expected_fields.items():
            if field not in exp:
                return False
            if not isinstance(exp[field], field_type):
                return False
        return True


    def _is_valid_exp_store(self) -> bool:
        """
        Each experience should have
        - id: str
        - action_path: list[str]
        - st: dict
            - url: str
            - html_text: str
            # - action_status: dict
        - action: str
        - st1: dict
            - url: str
            - html_text: str
            # - action_status: dict
        """
        # Every experience should be valid
        for store in [self.exp_store, self.depreiciate_exp_store]:
            for exp_id in store.keys():
                exp = store[exp_id]
                # Id should match
                if exp_id != exp["id"]:
                    return False
                # Individual experience check
                if not self._is_valid_exp(exp):
                    return False
        return True

    def store_experience(self, exp):
        if not self._is_valid_exp(exp):
            raise ValueError(f"Invalid experience: {exp}")
        self.exp_store[exp["id"]] = exp
        self.save_store() # TODO: maybe should not save so frequent

    def _has_conflict(self, e1, e2) -> bool:
        raise NotImplementedError

    def retrieve_experience(self, state) -> list:
        raise NotImplementedError

    def resolve_experience_conflict(self, **kwargs):
        raise NotImplementedError

    def _are_same_exp(self, e1, e2) -> bool:
        raise NotImplementedError

    def get_most_optmized_path_exp(self, exp_group: set) -> str:
        raise NotImplementedError
