from .webshop_exp_backend import WebshopExpBackend
import json

class WebshopExpVanillaBackend(WebshopExpBackend):
    def __init__(self, env_name, storage_path, algorithm: str):
        super().__init__(env_name, storage_path)
        self.algorithm = algorithm
        
        # set the retrieve experience
        if algorithm == "sameSt_1Step":
            self.retrieve_experience = self.retrieve_experience_sameSt_1Step
        raise NotImplementedError

    def retrieve_experience_sameSt_1Step(self, state) -> list:
        results = []
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if self.two_states_equal(state, exp['st']):
                results.append(exp)
        return results
