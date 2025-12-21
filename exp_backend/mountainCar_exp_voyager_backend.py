from .mountainCar_exp_vanilla_backend import MountainCarExpVanillaBackend
from .voyager_mixin import VoyagerMixin
from utils import log_flush


# MountainCar 专用的 prompt 模板
MOUNTAINCAR_SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes MountainCar state transitions.
1) Summarize in 1-2 sentences focusing on position, velocity, and action effects.
2) Describe whether the car is gaining momentum or losing it, and whether the step is SAFE or reaches the goal.
3) Your response should be a single line of text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this MountainCar experience:
- Position: {position:.3f}, Velocity: {velocity:.4f}
- Action: {action_name} ({action})
- Next Position: {next_position:.3f}, Next Velocity: {next_velocity:.4f}
- Outcome Safety: {outcome_safety}

Action meanings: 0=push left, 1=no push (coast), 2=push right
Goal: reach position >= 0.5

Your summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class MountainCarExpVoyagerBackend(MountainCarExpVanillaBackend, VoyagerMixin):
    """
    MountainCar 带 Voyager 总结机制的 Backend
    
    简易版 Voyager：在存储经验时生成 LLM 总结，检索逻辑保持不变（包括 BFS reachability）。
    """
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, explorer_model=None):
        """
        Args:
            env_name: 环境名称
            storage_path: 经验存储路径
            depreiciate_exp_store_path: 废弃经验存储路径
            explorer_model: Explorer 模型实例，用于生成总结
        """
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)
        self.init_voyager(explorer_model=explorer_model)

    def store_experience(self, exp) -> None:
        """存储经验并生成总结"""
        # 生成总结
        self._voyager_store_mountaincar_exp(exp)
        # 调用父类存储
        super().store_experience(exp)
        log_flush(self.logIO, f"[Voyager] Experience {exp['id']} stored with summary")

    def _voyager_store_mountaincar_exp(self, exp: dict) -> None:
        """MountainCar 专用的总结生成"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', 1)
        goal_reached = st1.get('position', 0) >= 0.5
        outcome_safety = "SUCCESS (goal reached)" if goal_reached else "SAFE (episode continues)"
        
        action_names = ['push left', 'coast', 'push right']
        action_name = action_names[action] if 0 <= action <= 2 else 'unknown'
        
        prompt = MOUNTAINCAR_SUMMARY_PROMPT.format(
            position=st.get('position', 0),
            velocity=st.get('velocity', 0),
            action=action,
            action_name=action_name,
            next_position=st1.get('position', 0),
            next_velocity=st1.get('velocity', 0),
            outcome_safety=outcome_safety
        )
        
        summary = self.voyager_generate_summary(prompt)
        exp['voyager_summary'] = summary
