from .frozenLake_exp_backend import FrozenLakeExpBackend
from .backend_config import frozenlake_vanilla_config
from utils import log_flush
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
import json

class FrozenLakeExpVanillaBackend(FrozenLakeExpBackend):
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)
        self.algorithm = frozenlake_vanilla_config["algorithm"]
        
        # set the retrieve experience
        if self.algorithm == "sameSt_1Step":
            self.retrieve_experience = self.retrieve_experience_sameSt_1Step
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
