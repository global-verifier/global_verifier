from .frozenLake_exp_vanilla_backend import FrozenLakeExpVanillaBackend
from .voyager_mixin import VoyagerMixin
from utils import log_flush


# FrozenLake 专用的 prompt 模板
FROZENLAKE_SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes FrozenLake navigation experiences.
1) Summarize in 1-2 sentences focusing on the position, action taken, and result.
2) Mention if the agent reached the goal, fell into a hole, or moved safely.
3) Your response should be a single line of text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this FrozenLake experience:
- Current Position: {position}
- Action: {action_name} ({action})
- Next Position: {next_position}
- Action Path: {action_path}

Action meanings: 0=Left, 1=Down, 2=Right, 3=Up
Grid: S=Start, F=Frozen(safe), H=Hole(fail), G=Goal(win)

Your summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class FrozenLakeExpVoyagerBackend(FrozenLakeExpVanillaBackend, VoyagerMixin):
    """
    FrozenLake 带 Voyager 总结机制的 Backend
    
    简易版 Voyager：在存储经验时生成 LLM 总结，检索逻辑保持不变。
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
        self._voyager_store_frozenlake_exp(exp)
        # 调用父类存储
        super().store_experience(exp)
        log_flush(self.logIO, f"[Voyager] Experience {exp['id']} stored with summary")

    def _voyager_store_frozenlake_exp(self, exp: dict) -> None:
        """FrozenLake 专用的总结生成"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', 0)
        action_path = exp.get('action_path', [])
        
        action_names = ['Left', 'Down', 'Right', 'Up']
        action_name = action_names[action] if 0 <= action <= 3 else 'unknown'
        
        prompt = FROZENLAKE_SUMMARY_PROMPT.format(
            position=st.get('position', 0),
            action=action,
            action_name=action_name,
            next_position=st1.get('position', 0),
            action_path=action_path
        )
        
        summary = self.voyager_generate_summary(prompt)
        exp['voyager_summary'] = summary
