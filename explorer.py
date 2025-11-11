from utils import load_explorer_model, load_adaptor
from config import explorer_settings
from utils import log_flush, get_timestamp

class Explorer:
    def __init__(self, model_name: str, env_name: str):
        self.explorer_model = load_explorer_model(model_name)
        self.max_steps = explorer_settings["max_steps"]
        self.adaptor = load_adaptor(env_name)
        self.instruction = None
        self.max_action_retries = explorer_settings["max_action_retries"]
        self.logIO = open(f'./log/explorerLog_{get_timestamp()}.txt', 'a')

    def explore(self):
        log_flush(self.logIO, f"########################################################")
        log_flush(self.logIO, f"Start exploring at {get_timestamp()}")
        # rest the status
        self.adaptor.initialize_env() 
        # Get the instruction
        self.instruction = self.adaptor.get_instruction()
        log_flush(self.logIO, self.adaptor.get_env_description())
        is_episode_done = False
        for i in range(self.max_steps):
            log_flush(self.logIO, f"--------------------------------------------------------")
            log_flush(self.logIO, f"Step {i}")
            if is_episode_done:
                log_flush(self.logIO, f"- Episode is done at step {i}")
                break
            # get current state, t
            cur_state = self.adaptor.get_state()
            log_flush(self.logIO, f"- Current state: {cur_state}")
            # get action
            action_status = self.adaptor.get_available_actions()
            log_flush(self.logIO, f"- Action status: {action_status}")
            if self.adaptor.is_finished_state(cur_state, action_status):
                log_flush(self.logIO, f"- Episode is done at step {i}")
                is_episode_done = True
                break
            # TODO: get experience
            todo_action = self.explorer_model.get_next_action(self.instruction, cur_state, action_status)
            log_flush(self.logIO, f"- Todo action: {todo_action}")
            action_valid =False
            for j in range(self.max_action_retries):
                if self.adaptor.is_valid_action(action_status, todo_action):
                    log_flush(self.logIO, f"- Action is valid after {j} retries")
                    action_valid = True
                    break
                # not valid, re-inference and get new action
                log_flush(self.logIO, f"- Action is not valid: {todo_action}")
                todo_action = self.explorer_model.get_next_action(self.instruction, cur_state, action_status)
                log_flush(self.logIO, f"- New todo action: {todo_action}")
            if not action_valid:
                raise ValueError(f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
            # get new state, t+1
            self.adaptor.step(todo_action)
            # TODO: store experience
        # check if the episode is done
        if not is_episode_done:
            raise ValueError(f"episode is not done after {self.max_steps} steps")
