import json
import os
import itertools
from typing import Any, Dict, Iterable, List
from utils import get_timestamp
from .backend_config import base_backend_config
from utils import log_flush

class BaseExpBackend:
    def __init__(self, env_name: str, storage_path: str ="./storage/exp_store.json") -> None:
        self.env_name = env_name
        self.storage_path = storage_path
        
        # Add the logger (must be created before _load_store() since it uses logIO)
        log_dir = base_backend_config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.logIO = open(f'{log_dir}/exp_backendLog_{get_timestamp()}.log', 'a')
        
        self.exp_store = self._load_store()
        if not self._is_valid_exp_store():
            raise ValueError(f"Invalid experience store, did not pass the validation.")

    def _load_store(self):
        log_flush(self.logIO, f"########################################################")
        log_flush(self.logIO, f"Start loading the store at {get_timestamp()}")
        if self.storage_path is None:
            raise ValueError("storage_path is None, please provide a valid path.")

        # Start loading the store
        exp_store = {}
        try:
            storage_path = os.fspath(self.storage_path)
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
        with open(self.storage_path, 'w') as file:
            json.dump(self.exp_store, file)
        log_flush(self.logIO, f"Save store, path: {self.storage_path}, size: {len(self.exp_store)}, at {get_timestamp()}")

    def _loop_detect_exp_conflict(self):
        """
        Detect all the conflict pairs in the experience store.
        """
        log_flush(self.logIO, f"Loop detecting conflict pairs")
        conflict_pairs = []
        exp_ids = self.exp_store.keys()
        exp_id_combinations = list(itertools.combinations(exp_ids, 2))
        for exp_pair in exp_id_combinations:
            if self._has_conflict(self.exp_store[exp_pair[0]], self.exp_store[exp_pair[1]]):
                log_flush(self.logIO, f"Conflict pair detected: {exp_pair[0]['id']} and {exp_pair[1]['id']}")
                conflict_pairs.append(exp_pair)
        log_flush( self.logIO, f"Loop finish, num conflict pairs detected: {len(conflict_pairs)}")
        return conflict_pairs

    # To be implemented in the subclass
    def _is_valid_exp(self, exp) -> bool:
        raise NotImplementedError

    def _is_valid_exp_store(self) -> bool:
        raise NotImplementedError

    def store_experience(self, exp):
        raise NotImplementedError

    def _has_conflict(self, e1, e2) -> bool:
        raise NotImplementedError

    def retrieve_experience(self, state) -> list:
        raise NotImplementedError
