from utils import load_explorer_model, load_adaptor
from config import explorer_settings

class Explorer:
    def __init__(self, model_name: str, env_name: str):
        self.explorer_model = load_explorer_model(model_name)
        self.max_steps = explorer_settings["max_steps"]
        self.adaptor = load_adaptor(env_name)
        self.instruction = None
        self.max_action_retries = explorer_settings["max_action_retries"]

    def explore(self):
        # rest the status
        self.adaptor.initialize_env() 
        # Get the instruction
        self.instruction = self.adaptor.get_instruction()
        is_episode_done = False
        for i in range(self.max_steps):
            if is_episode_done:
                break
            # get current state, t
            cur_state = self.adaptor.get_state()
            # get action
            action_status = self.adaptor.get_available_actions()
            if self.adaptor.is_finished_state(cur_state, action_status):
                is_episode_done = True
                break
            # TODO: get experience
            todo_action = self.explorer_model.get_next_action(self.instruction, cur_state, action_status)
            action_valid =False
            for j in range(self.max_action_retries):
                if self.adaptor.is_valid_action(action_status, todo_action):
                    action_valid = True
                    break
                # not valid, re-inference and get new action
                todo_action = self.explorer_model.get_next_action(self.instruction, cur_state, action_status)
            if not action_valid:
                raise ValueError(f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
            # get new state, t+1
            self.adaptor.step(todo_action)
            # TODO: store experience
        # check if the episode is done
        if not is_episode_done:
            raise ValueError(f"episode is not done after {self.max_steps} steps")
