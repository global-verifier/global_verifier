import json

class BaseEnvAdaptor:
    def __init__(self, env_name):
        self.env_name = env_name

    # Consultants
    def check_action_valid(self, action, available_actions):
        raise NotImplementedError

    # Modifiers
    def initialize_env(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_action_prompt(self, instruction: str, state: dict, available_actions: list) -> str:
        raise NotImplementedError

    def reconstruct_state(self, exp):
        raise NotImplementedError

    def is_same_state(self, state1, state2):
        return state1 == state2

    def format_action(self, action):
        return action

    # Static
    @staticmethod
    def has_conflict(e1, e2) -> bool:
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

    @staticmethod
    def are_same_exp(e1, e2) -> bool:
        """
        Check if two experiences are the same.
        """
        return e1['st'] == e2['st'] and e1['action'] == e2['action'] and e1['st1'] == e2['st1']

    @staticmethod
    def get_state_str(state) -> str:
        # Use sort_keys=True to ensure consistent ordering regardless of key insertion order
        return json.dumps(state, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def two_states_equal(state1, state2) -> bool:
        return BaseEnvAdaptor.get_state_str(state1) == BaseEnvAdaptor.get_state_str(state2)
