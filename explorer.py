import os

from analyzer.explorer_run_analyzer import ExplorerRunAnalyzer
from plugin_loader import load_explorer_model, load_adaptor, load_exp_backend
from utils import log_flush, get_timestamp, get_timestamp_ms, is_success_trail, extract_exp_ids
from config import explorer_settings
import time

class Explorer:
    def __init__(
        self,
        start_timestep: int = 0,
        model_name: str = None,
        env_name: str = None,
        memory_env: str = None,
        max_steps: int = None,
        use_memory: bool = True,
        # use_experience: bool = None,
        # save_experience: bool = None,
        threshold: float = None,
        decay_rate: float = None,
        log_dir: str = None,
        backend_log_dir: str = None,
        storage_path: str = None,
        depreiciate_exp_store_path: str = None,
        desc=None,
        force=None,
        goal_rewards=None,
        enable_confirm_purchase=None,
        session=None,
        use_api=False,
        use_global_verifier=False,
        ):
        # Add plug in
        self.explorer_model = load_explorer_model(model_name or explorer_settings["model_name"], use_api=use_api)
        self.init_after_model(
            model_name=model_name,
            env_name=env_name,
            memory_env=memory_env,
            max_steps=max_steps,
            use_memory=use_memory,
            # use_experience=use_experience,
            # save_experience=save_experience,
            start_timestep=start_timestep,
            threshold=threshold,
            decay_rate=decay_rate,
            log_dir=log_dir,
            backend_log_dir=backend_log_dir,
            storage_path=storage_path,
            depreiciate_exp_store_path=depreiciate_exp_store_path,
            desc=desc,
            force=force,
            goal_rewards=goal_rewards,
            enable_confirm_purchase=enable_confirm_purchase,
            session=session,
            use_global_verifier=use_global_verifier,
        )

    def init_after_model(
        self,
        start_timestep=None,
        model_name=None,
        env_name=None,
        memory_env=None,
        max_steps=None,
        use_memory=None,
        threshold=None,
        decay_rate=None,
        log_dir=None,
        backend_log_dir=None,
        storage_path=None,
        depreiciate_exp_store_path=None,
        desc=None,
        force=None,
        goal_rewards=None,
        enable_confirm_purchase=None,
        session=None,
        use_global_verifier=False,
    ):
        """
        Finish initialization steps that do not require reloading the explorer_model.
        Separated for reuse when re-running init logic while keeping the same model.
        """
        # Update hyperparameters (prefer provided args, else config defaults or existing values)
        self.model_name = model_name or getattr(self, "model_name", None) or explorer_settings["model_name"]
        if env_name not in ["frozenlake", "mountaincar", "webshop"]:
            raise ValueError(f"Invalid environment name: {env_name}")
        self.env_name = env_name or getattr(self, "env_name", None) or explorer_settings["env_name"]
        self.process_memory_env(memory_env)
        self.max_steps = (
            self.max_steps if max_steps is None else max_steps
        ) if hasattr(self, "max_steps") else (max_steps if max_steps is not None else explorer_settings["max_steps"])

        self.log_dir = log_dir or getattr(self, "log_dir", None) or explorer_settings["log_dir"]
        self.backend_log_dir = (
            backend_log_dir
            or getattr(self, "backend_log_dir", None)
            or self.log_dir
        )
        self.storage_path = (
            self.storage_path if storage_path is None else storage_path
        ) if hasattr(self, "storage_path") else (storage_path if storage_path is not None else explorer_settings["storage_path"])
        self.depreiciate_exp_store_path = (
            self.depreiciate_exp_store_path if depreiciate_exp_store_path is None else depreiciate_exp_store_path
        ) if hasattr(self, "depreiciate_exp_store_path") else (depreiciate_exp_store_path if depreiciate_exp_store_path is not None else explorer_settings["depreiciate_exp_store_path"])
        self.max_action_retries = explorer_settings["max_action_retries"]
        self.start_timestep = (
            getattr(self, "start_timestep", start_timestep)
            if start_timestep is None
            else start_timestep
        )
        self.threshold = (
            getattr(self, "threshold", threshold)
            if threshold is None
            else threshold
        )
        self.decay_rate = (
            getattr(self, "decay_rate", decay_rate)
            if decay_rate is None
            else decay_rate
        )
        # Webshop-specific overrides
        self.enable_confirm_purchase = (
            enable_confirm_purchase
            if enable_confirm_purchase is not None
            else getattr(self, "enable_confirm_purchase", None)
        )
        self.session = (
            session
            if session is not None
            else getattr(self, "session", None)
        )
        self.use_global_verifier = use_global_verifier
        assert use_memory is not None, "use_memory must be provided"
        if use_memory:
            self.use_experience = True
            self.save_experience = True
        else:
            self.use_experience = False
            self.save_experience = False

        adaptor_kwargs = {}
        if "frozenlake" in self.env_name:
            adaptor_kwargs["desc"] = desc
            adaptor_kwargs["goal_rewards"] = goal_rewards
        if "mountaincar" in self.env_name:
            adaptor_kwargs["force"] = force
        if "webshop" in self.env_name:
            adaptor_kwargs["enable_confirm_purchase"] = self.enable_confirm_purchase
            adaptor_kwargs["session"] = self.session
        self.adaptor = load_adaptor(self.env_name, self.model_name, **adaptor_kwargs)
        # 传入 explorer_model 给 backend（voyager backend 需要用它生成总结）
        self.exp_backend = load_exp_backend(
            self.backend_env,
            self.storage_path,
            self.depreiciate_exp_store_path,
            self.explorer_model,
            log_dir=self.backend_log_dir,
            start_timestep=self.start_timestep,
            threshold=self.threshold,
            decay_rate=self.decay_rate,
        )

        # Add the logger
        os.makedirs(self.log_dir, exist_ok=True)
        self.logIO = open(f'{self.log_dir}/explorerLog_{get_timestamp()}.log', 'a')
        self.promptLogIO = open(f'{self.log_dir}/promptLog_{get_timestamp()}.log', 'a')
        self.statusLogIO = open(f'{self.log_dir}/statusLog_{get_timestamp()}.log', 'a')
        
        self.run_analyzer = ExplorerRunAnalyzer(self.log_dir)

        # Add the state recorders
        self.state_trace = None

        # Add explorer status
        self.used_exp_ids = set()
        self.in_process = False
        self.conflict_soultion = explorer_settings["conflict_soultion"]
        self.alpha = explorer_settings["alpha"]

    def process_memory_env(self, memory_env: str):
        if memory_env not in ["vanilla", "generative", "memorybank", "voyager"]:
            raise ValueError(f"Invalid memory environment: {memory_env}")
        self.backend_env = f"{self.env_name}-{memory_env}"

    def get_next_action(self, retrieved_experiences: list = None) -> str:
        if retrieved_experiences is None:
            retrieved_experiences = []
        get_action_prompt = self.adaptor.get_action_prompt(retrieved_experiences)
        log_flush(self.promptLogIO, f"Action prompt: [{get_timestamp()}] - {get_action_prompt}")
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
        self.state_trace.append(cur_state)
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
            log_flush(self.logIO, f"- Retrieved experience, len: {len(retrieved_experiences)}, exps: {retrieved_experiences}")
            self.used_exp_ids.update(extract_exp_ids(retrieved_experiences))
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
            # raise ValueError(f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
            log_flush(self.logIO, f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
            print(f"todo_action {todo_action} is not valid after {self.max_action_retries} retries")
            return False
        
        # Execute action
        self.adaptor.step(todo_action)
        print(f"Action '{todo_action}' is taken")
        
        # Store experience
        if self.save_experience:
            self.record_experience()
        else:
            log_flush(self.logIO, f"- Experience saving disabled (save_experience=False), skipping save")
        
        # For memory bank backend, step the memory bank
        self.exp_backend.step()
        
        return False

    # Detect and resolve conflict pairs
    def _detect_experience_conflict(self):
        """Check if the current experience is conflict with the existing experiences."""
        conflict_pair_ids = self.exp_backend._loop_detect_exp_conflict()
        return conflict_pair_ids

    def _detect_experience_redundancy(self):
        """Check if the current experience is conflict with the existing experiences."""
        redundant_experience_groups = self.exp_backend.get_redundant_experience_groups()
        return redundant_experience_groups

    def solve_experience_conflict(self, conflict_pair_id):
        log_flush(self.logIO, f"-- Resolve Conflict pair ID: {conflict_pair_id} ---")
        if self.exp_backend._exp_is_depreciated(conflict_pair_id[0]) or self.exp_backend._exp_is_depreciated(conflict_pair_id[1]):
            log_flush(self.logIO, f"One of the experiences is deprecated, skipping conflict resolution")
            return
        e0 = self.exp_backend.get_exp_by_id(conflict_pair_id[0])
        e1 = self.exp_backend.get_exp_by_id(conflict_pair_id[1])
        # go to the st status for e0
        e0_st_success, error_message = self.adaptor.reconstruct_st(e0)
        e0_st1_success = False
        if not e0_st_success:
            log_flush(self.logIO, f"Error reconstructing st for e0: {error_message}")
        else:
            self.adaptor.step(e0['action'])
            e0_st1_success = self.adaptor.is_same_state(self.adaptor.get_state(), e0['st1'])
        # go to the st status for e1
        e1_st1_success = False
        e1_st_success, error_message = self.adaptor.reconstruct_st(e1)
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
        print(f"Conflict pair len: {len(conflict_pair_ids)}, ids: {conflict_pair_ids}")
        log_flush(self.logIO, f"Conflict pair len: {len(conflict_pair_ids)}, ids: {conflict_pair_ids}")
        for conflict_pair_id in conflict_pair_ids:
            while self.in_process:
                time.sleep(1)
            self.in_process = True
            self.solve_experience_conflict(conflict_pair_id)
            self.in_process = False
        log_flush(self.logIO, f"---------------- Finished Resolve Experience Conflict ----------------")

    def remove_false_exp(self, exp_id):
        """
        Verify if an experience's st1 can be reproduced.
        If not, deprecate the experience.
        """
        log_flush(self.logIO, f"-- Verifying exp: {exp_id} ---")
        
        # Check if already deprecated
        if self.exp_backend._exp_is_depreciated(exp_id):
            log_flush(self.logIO, f"Experience {exp_id} already deprecated, skipping")
            return
        
        exp = self.exp_backend.get_exp_by_id(exp_id)
        
        # Reconstruct to st1 state directly
        st1_success, error_message = self.adaptor.reconstruct_st1(exp)
        
        if st1_success:
            log_flush(self.logIO, f"[VALID] {exp_id} - st1 reproduced successfully")
        else:
            log_flush(self.logIO, f"[INVALID] {exp_id} - {error_message}")
            log_flush(self.logIO, f"[DEPRECATE] {exp_id} - st1 cannot be reproduced")
            self.exp_backend._deprecate_experience(exp_id)

    def resolve_all_exp_conflict_st(self):
        """Resolve the conflict pairs."""

        log_flush(self.logIO, f"---------------- Resolve Experience Conflict ----------------")
        conflict_pair_ids = self._detect_experience_conflict()
        conflict_exp_ids = self.exp_backend.get_all_expIds_from_conflict_st(conflict_pair_ids)
        print(f"Conflict exp len: {len(conflict_exp_ids)}, ids: {conflict_exp_ids}")
        log_flush(self.logIO, f"Conflict exp len: {len(conflict_exp_ids)}, ids: {conflict_exp_ids}")
        for conflict_exp_id in conflict_exp_ids:
            while self.in_process:
                time.sleep(1)
            self.in_process = True
            self.remove_false_exp(conflict_exp_id)
            self.in_process = False
        log_flush(self.logIO, f"---------------- Finished Resolve Experience Conflict ----------------")

    def _mdp_reproduce_st(self, action_path, st, exp_id):
        """
        Try to reproduce st via action_path.
        Returns True if successful, False otherwise.
        """
        self.adaptor.initialize_env()
        try:
            for a in action_path:
                self.adaptor.step(a)
        except Exception as e:
            log_flush(self.logIO, f"[MDP] Failed to reproduce st for {exp_id}: {e}")
            return False
        
        if not self.adaptor.is_same_state(self.adaptor.get_state(), st):
            log_flush(self.logIO, f"[MDP] State mismatch for {exp_id}, skipping")
            return False
        return True

    def _mdp_collect_st1_distribution(self, action_path, action):
        """
        Run action alpha times to collect st1 distribution.
        Returns dict: state_str -> (st1, count)
        """
        st1_counts = {}
        for _ in range(self.alpha):
            self.adaptor.initialize_env()
            for a in action_path:
                self.adaptor.step(a)
            self.adaptor.step(action)
            st1 = self.adaptor.get_state()
            st1_key = self.adaptor.get_state_str(st1) if hasattr(self.adaptor, 'get_state_str') else str(st1)
            if st1_key not in st1_counts:
                st1_counts[st1_key] = (st1, 0)
            st1_counts[st1_key] = (st1_counts[st1_key][0], st1_counts[st1_key][1] + 1)
        return st1_counts

    def _mdp_store_experiences_with_probability(self, st, action, action_path, exp_id, st1_counts):
        """
        Create and store experiences with probability for each unique st1.
        """
        full_action_path = action_path + [action]
        for st1_key, (st1, count) in st1_counts.items():
            probability = count / self.alpha
            new_exp = {
                "id": f"{get_timestamp_ms()}_{exp_id}_prob{probability:.2f}",
                "reproduce_method": "action_path",
                "action_path": full_action_path,
                "st": st,
                "action": action,
                "st1": st1,
                "probability": probability,
            }
            self.exp_backend.store_experience(new_exp)
            log_flush(self.logIO, f"[MDP] Stored new exp with probability {probability:.2f}")

    def resolve_all_exp_conflict_mdp(self):
        """Resolve the conflict pairs using MDP distribution estimation."""
        log_flush(self.logIO, f"---------------- Resolve Experience Conflict ----------------")
        conflict_pair_ids = self._detect_experience_conflict()
        conflict_exp_ids = self.exp_backend.get_all_expIds_from_conflict_st(conflict_pair_ids)
        print(f"Conflict exp len: {len(conflict_exp_ids)}, ids: {conflict_exp_ids}")
        log_flush(self.logIO, f"Conflict exp len: {len(conflict_exp_ids)}, ids: {conflict_exp_ids}")
        to_get_distributions = self.exp_backend.get_unique_st_action_pairs(conflict_exp_ids)

        for (st, action, exp_id, action_path) in to_get_distributions:
            if not self._mdp_reproduce_st(action_path, st, exp_id):
                continue
            st1_counts = self._mdp_collect_st1_distribution(action_path, action)
            self._mdp_store_experiences_with_probability(st, action, action_path, exp_id, st1_counts)
        
        # Deprecate all original conflict experiences
        for exp_id in conflict_exp_ids:
            self.exp_backend._deprecate_experience(exp_id)
        log_flush(self.logIO, f"---------------- Finished Resolve Experience Conflict ----------------")

    def remove_redundant_experiences(self):
        """Remove redundant experiences."""
        log_flush(self.logIO, f"---------------- Remove Redundant Experiences ----------------")
        redundant_experience_groups = self._detect_experience_redundancy()
        print(f"Redundant experience len: {len(redundant_experience_groups)}, groups: {redundant_experience_groups}")
        log_flush(self.logIO, f"Redundant experience len: {len(redundant_experience_groups)}, groups: {redundant_experience_groups}")
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
        log_flush(self.logIO, f"[BEFORE] number of experiences: {len(self.exp_backend.exp_store)}")
        log_flush(self.logIO, f"[BEFORE] number of deprecated experiences: {len(self.exp_backend.depreiciate_exp_store)}")
        self.remove_redundant_experiences()
        if self.conflict_soultion == "conflict":
            self.resolve_all_experience_conflict()
        elif self.conflict_soultion == "st":
            self.resolve_all_exp_conflict_st()
        elif self.conflict_soultion == "mdp":
            self.resolve_all_exp_conflict_mdp()
        else:
            raise ValueError(f"Invalid conflict solution: {self.conflict_soultion}")
        log_flush(self.logIO, f"[AFTER] number of experiences: {len(self.exp_backend.exp_store)}")
        log_flush(self.logIO, f"[AFTER] number of deprecated experiences: {len(self.exp_backend.depreiciate_exp_store)}")

    def _reset_exploration_state(self):
        """Reset the exploration state for a new episode."""
        self.in_process = True
        self.state_trace = []
        self.used_exp_ids.clear()

    def explore(self):
        # Reset the exploration state
        self._reset_exploration_state()

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
            log_flush(self.logIO, f"Step {i} / {self.max_steps}")
            is_episode_done = self.execute_step()
            status = self.exp_backend.export_status()
            log_flush(self.statusLogIO, f"Step {i} export_status: {status}")
            
            if is_episode_done:
                log_flush(self.logIO, f"- Episode is done at step {i}")
                step_count = i
                break
        
        # Get the final score and step count
        if not is_episode_done:
            # Episode didn't finish within max_steps
            log_flush(self.logIO, f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log_flush(self.logIO, f"- Episode is NOT done after {self.max_steps} steps")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"- Episode is NOT done after {self.max_steps} steps")
            score = 0  # Mark failed episodes with -1 score
            step_count = self.max_steps
        else:
            # Episode finished successfully
            score = self.adaptor.extract_reward_score()
            if "memorybank" in self.backend_env and is_success_trail(score):
                # 成功的 trail，更新使用过的经验的时间戳
                exp_ids_list = list(self.used_exp_ids)
                self.exp_backend.finish_explore_trail(exp_ids=exp_ids_list)
                log_flush(self.logIO, f"- Success trail! Updated timestamps for {len(exp_ids_list)} experiences")

        log_flush(self.logIO, f"Insturction: {self.adaptor.get_instruction()}")
        log_flush(self.logIO, f"Step count: {step_count}")
        log_flush(self.logIO, f"Final score: {score}")
        log_flush(self.logIO, f"Action path: {self.adaptor.get_action_path()}")
        end_timestamp = get_timestamp()
        log_flush(self.logIO, f"End exploring at {end_timestamp}")
        log_flush(self.logIO, f"State trace: {self.state_trace}")
        log_flush(self.logIO, f"########################################################")

        print(f"Insturction: {self.adaptor.get_instruction()}")
        print(f"Step count: {step_count}")
        print(f"Final score: {score}")
        print(f"Action path: {self.adaptor.get_action_path()}")
        print(f"End exploring at {end_timestamp}")
        print(f"State trace: {self.state_trace}")
        print(f"########################################################")


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

        # Reset in_process before refining (to avoid deadlock)
        self.in_process = False

        # Refine experiences after each exploration
        if self.use_global_verifier:
            log_flush(self.logIO, f"[POST-EXPLORE] Running global verifier (refine_experience)")
            self.refine_experience()
        else:
            log_flush(self.logIO, f"[POST-EXPLORE] Running redundancy removal only")
            self.remove_redundant_experiences()
