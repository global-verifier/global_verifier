from .base_exp_backend import BaseExpBackend

class WebshopExpBackend(BaseExpBackend):
    def __init__(self, env_name, storage_path):
        super().__init__(env_name, storage_path)

    def _is_valid_exp_store(self) -> bool:
        """
        Each experience should have
        - id
        - action_path
        - st
            - url
            - html_text
            - action_status
        - action
        - st1
            - url
            - html_text
            - action_status
        """
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
