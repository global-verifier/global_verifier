import os

from analyzer.explorer_run_analyzer import ExplorerRunAnalyzer
from plugin_loader import load_explorer_model, load_adaptor, load_exp_backend
from utils import log_flush, get_timestamp
from config import explorer_settings
import time

class Explorer:
    def __init__(self):
        # Add hyperparameter settings
        self.model_name = explorer_settings["model_name"]
        self.env_name = explorer_settings["env_name"]
        self.backend_env = explorer_settings["backend_env"]
        self.storage_path = explorer_settings["storage_path"]
        self.depreiciate_exp_store_path = explorer_settings["depreiciate_exp_store_path"]
        self.max_steps = explorer_settings["max_steps"]
        self.max_action_retries = explorer_settings["max_action_retries"]
        self.use_experience = explorer_settings.get("use_experience", True)  
        self.save_experience = explorer_settings.get("save_experience", True)
        
        # Add plug in
        self.explorer_model = load_explorer_model(self.model_name)
        self.adaptor = load_adaptor(self.env_name)
        self.exp_backend = load_exp_backend(self.backend_env, self.storage_path, self.depreiciate_exp_store_path)

        # Add the logger
        self.logIO = open(f'{explorer_settings["log_dir"]}/explorerLog_{get_timestamp()}.log', 'a')
        self.run_analyzer = ExplorerRunAnalyzer(explorer_settings["log_dir"])

        # Add explorer status
        self.in_process = False

    def get_next_action(self, retrieved_experiences: list = None) -> str:
        if retrieved_experiences is None:
            retrieved_experiences = []
        get_action_prompt = self.adaptor.get_action_prompt(retrieved_experiences)
        raw_action = self.explorer_model.get_next_action(get_action_prompt)
        formatted_action = self.adaptor.format_action(raw_action)
        return formatted_action

    def record_experience(self):
        """Record the current step's experience to the backend."""
        new_exp = self.adaptor.get_experience()
        self.exp_backend.store_experience(new_exp)
        log_flush(self.logIO, f"- Experience stored: {new_exp['id']}")

    def execute_step(self) -> bool:
        """
        Execute a single step in the exploration.
            
        Returns:
            True if the episode is done, False otherwise
        """

        # Get current state
        cur_state = self.adaptor.get_state()
        print(f"Current state: {cur_state}")
        log_flush(self.logIO, f"- Current state: {cur_state}")
        
        # Get action status/options
        print(f"Action status: {self.adaptor.get_available_actions()}")
        log_flush(self.logIO, f"- Action status: {self.adaptor.get_available_actions()}")
        
        # Check if finished
        if self.adaptor.is_finished_state(cur_state):
            return True
        
        # Retrieve experience
        retrieved_experiences = []
        if self.use_experience:
            retrieved_experiences = self.exp_backend.retrieve_experience(cur_state)
            print(f"Retrieved {len(retrieved_experiences)} experiences: {retrieved_experiences}")
            log_flush(self.logIO, f"- Retrieved experience: {retrieved_experiences}")
        else:
            log_flush(self.logIO, f"- Experience retrieval disabled (use_experience=False), using empty experience list")
        
        # Get and validate action (pass retrieved experiences to the prompt if enabled)
        todo_action = self.get_next_action(retrieved_experiences)
        log_flush(self.logIO, f"- Todo action: {todo_action}")
        action_valid = False
        
        for j in range(self.max_action_retries):
            if self.adaptor.is_valid_action(todo_action):
                log_flush(self.logIO, f"- Action is valid after {j} retries")
                action_valid = True
                break
            # Not valid, re-inference and get new action
            print(f"   - Action is not valid: {todo_action}")
            log_flush(self.logIO, f"- Action is not valid: {todo_action}")
            todo_action = self.get_next_action(retrieved_experiences)
            print(f"   - new todo action: {todo_action}")
            log_flush(self.logIO, f"- New todo action: {todo_action}")
        
        if not action_valid:
            raise ValueError(f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
        
        # Execute action
        self.adaptor.step(todo_action)
        print(f"Action '{todo_action}' is taken")
        
        # Store experience
        if self.save_experience:
            self.record_experience()
        else:
            log_flush(self.logIO, f"- Experience saving disabled (save_experience=False), skipping save")
        
        return False

    # Detect and resolve conflict pairs
    def _detect_experience_conflict(self):
        """Check if the current experience is conflict with the existing experiences."""
        conflict_pair_ids = self.exp_backend._loop_detect_exp_conflict()
        
        return conflict_pair_ids

    def solve_experience_conflict(self, conflict_pair_id):
        log_flush(self.logIO, f"-- Resolve Conflict pair ID: {conflict_pair_id} ---")
        e0 = self.exp_backend.get_exp_by_id(conflict_pair_id[0])
        e1 = self.exp_backend.get_exp_by_id(conflict_pair_id[1])
        # go to the st status for e0
        e0_st_success, error_message = self.adaptor.reconstruct_state(e0)
        e0_st1_success = False
        if not e0_st_success:
            log_flush(self.logIO, f"Error reconstructing st for e0: {error_message}")
        else:
            self.adaptor.step(e0['action'])
            e0_st1_success = self.adaptor.is_same_state(self.adaptor.get_state(), e0['st1'])
        # go to the st status for e1
        e1_st1_success = False
        e1_st_success, error_message = self.adaptor.reconstruct_state(e1)
        if not e1_st_success:
            log_flush(self.logIO, f"Error reconstructing st for e1: {error_message}")
        else:
            self.adaptor.step(e1['action'])
            e1_st1_success = self.adaptor.is_same_state(self.adaptor.get_state(), e1['st1'])
        # send result to backend, let it decide
        log_flush(self.logIO, f"- Result: e0_st_success: {e0_st_success}, e0_st1_success: {e0_st1_success}, e1_st_success: {e1_st_success}, e1_st1_success: {e1_st1_success}")
        print(f"- Result: e0_st_success: {e0_st_success}, e0_st1_success: {e0_st1_success}, e1_st_success: {e1_st_success}, e1_st1_success: {e1_st1_success}")
        self.exp_backend.resolve_experience_conflict(conflict_pair_id=conflict_pair_id, examine_result=(e0_st_success, e0_st1_success, e1_st_success, e1_st1_success))


    def resolve_all_experience_conflict(self):
        """Resolve the conflict pairs."""

        log_flush(self.logIO, f"---------------- Resolve Experience Conflict ----------------")
        conflict_pair_ids = self._detect_experience_conflict()
        for conflict_pair_id in conflict_pair_ids:
            while self.in_process:
                time.sleep(1)
            self.in_process = True
            self.solve_experience_conflict(conflict_pair_id)
            self.in_process = False
        log_flush(self.logIO, f"---------------- Finished Resolve Experience Conflict ----------------")

    def remove_redundant_experiences(self):
        """Remove redundant experiences."""
        log_flush(self.logIO, f"---------------- Remove Redundant Experiences ----------------")
        redundant_experience_groups = self.exp_backend.get_redundant_experience_groups()
        print(f"Amount of redundant experience groups: {len(redundant_experience_groups)}")
        print(f"Redundant experience groups: {redundant_experience_groups}")
        for group in redundant_experience_groups:
            best_exp_id = self.exp_backend.get_most_optmized_path_exp_id(group)
            group.remove(best_exp_id)
            for exp_id in group:
                self.exp_backend._deprecate_experience(exp_id)
        log_flush(self.logIO, f"---------------- Finished Remove Redundant Experiences ----------------")

    def refine_experience(self):
        """
        Consist of two steps:
        1. Remove redundant experiences
        2. Resolve conflict pairs
        """
        self.remove_redundant_experiences()
        self.resolve_all_experience_conflict()


    def explore(self):
        self.in_process = True
        print(f"Start exploring at {get_timestamp()}")
        log_flush(self.logIO, f"########################################################")
        log_flush(self.logIO, f"Start exploring at {get_timestamp()}")
        # Reset the status
        self.adaptor.initialize_env() 
        # Get the instruction
        log_flush(self.logIO, self.adaptor.get_env_description())
        
        is_episode_done = False
        step_count = 0
        
        for i in range(self.max_steps):
            print(f"Step {i}")
            log_flush(self.logIO, f"--------------------------------------------------------")
            log_flush(self.logIO, f"Step {i}")
            is_episode_done = self.execute_step()
            if is_episode_done:
                log_flush(self.logIO, f"- Episode is done at step {i}")
                step_count = i
                break
        
        # Get the final score and step count
        if not is_episode_done:
            # Episode didn't finish within max_steps
            log_flush(self.logIO, f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log_flush(self.logIO, f"- Episode is NOT done after {self.max_steps} steps")
            score = -1  # Mark failed episodes with -1 score
            step_count = self.max_steps
        else:
            # Episode finished successfully
            score = self.adaptor.extract_reward_score()

        log_flush(self.logIO, f"Insturction: {self.adaptor.get_instruction()}")
        log_flush(self.logIO, f"Step count: {step_count}")
        log_flush(self.logIO, f"Final score: {score}")
        log_flush(self.logIO, f"Action path: {self.adaptor.get_action_path()}")
        end_timestamp = get_timestamp()
        log_flush(self.logIO, f"End exploring at {end_timestamp}")
        log_flush(self.logIO, f"########################################################")

        # Record to CSV regardless of success or failure
        self.run_analyzer.record_run(
            timestamp=end_timestamp,
            model_name=self.model_name,
            env_name=self.env_name,
            instruction=self.adaptor.get_instruction(),
            action_path=self.adaptor.get_action_path(),
            step_count=step_count,
            final_score=score,
        )
        self.run_analyzer.save_to_csv()

        self.in_process = False
