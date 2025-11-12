import json
from typing import Any, Dict, Iterable, List
import itertools

class BaseExpBackend:
    def __init__(self, env_name: str, storage_path=None) -> None:
        self.env_name = env_name
        self.storage_path = storage_path
        self.exp_store = self._load_store()
        if self.exp_store is not None and not self._is_valid_exp_store():
            raise ValueError(f"Invalid experience store, did not pass the validation.")

    def _load_store(self):
        exp_store = {}
        if self.storage_path is not None:
            with open(self.storage_path, 'r') as file:
                exp_store = json.load(file)
        return exp_store

    def store_backend(self) -> None:
        """
        Store the experience store to the storage path.
        """
        with open(self.storage_path, 'w') as file:
            json.dump(self.exp_store, file)

    def _loop_detect_exp_conflict(self):
        """
        Detect all the conflict pairs in the experience store.
        """
        conflict_pairs = []
        exp_ids = self.exp_store.keys()
        exp_id_combinations = list(itertools.combinations(exp_ids, 2))
        for exp_pair in exp_id_combinations:
            if self._has_conflict(self.exp_store[exp_pair[0]], self.exp_store[exp_pair[1]]):
               conflict_pairs.append(exp_pair)
        return conflict_pairs

    # To be implemented in the subclass
    def _is_valid_exp_store(self) -> bool:
        raise NotImplementedError

    def store_experience(self, experience: Any) -> None:
        raise NotImplementedError

    def _has_conflict(self, e1, e2) -> bool:
        raise NotImplementedError
