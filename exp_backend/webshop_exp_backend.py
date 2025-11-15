from .base_exp_backend import BaseExpBackend
import json

class WebshopExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path):
        super().__init__(env_name, storage_path)

    # Getters
    def _get_state_str(self, state) -> str:
        # Use sort_keys=True to ensure consistent ordering regardless of key insertion order
        return json.dumps(state, ensure_ascii=False, sort_keys=True)

    # Setters
    def store_experience(self, exp):
        if not self._is_valid_exp(exp):
            raise ValueError(f"Invalid experience: {exp}")
        self.exp_store[exp["id"]] = exp
        self.save_store() # TODO: maybe should not save so frequent

    # Consultants
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
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            # Id should match
            if exp_id != exp["id"]:
                return False
            # Individual experience check
            if not self._is_valid_exp(exp):
                return False
        return True

    def _has_conflict(self, e1, e2) -> bool:
        """
        Check if two experiences have conflict.
        - if same st, action
            - different st1
        # == actually works for dict, did not know that before
        """
        if e1['st'] == e2['st'] and e1['action'] == e2['action']:
            if e1['st1'] != e2['st1']:
                return True
        return False

    def two_states_equal(self, state1, state2) -> bool:
        return self._get_state_str(state1) == self._get_state_str(state2)
