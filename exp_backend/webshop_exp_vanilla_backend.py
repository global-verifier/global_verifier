from .webshop_exp_backend import WebshopExpBackend
from .backend_config import webshop_vanilla_config
from utils import log_flush
import json

class WebshopExpVanillaBackend(WebshopExpBackend):
    def __init__(self, env_name, storage_path):
        super().__init__(env_name, storage_path)
        self.algorithm = webshop_vanilla_config["algorithm"]
        
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
            if self.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences, results ids: {[exp['id'] for exp in results]}")        
        return results
