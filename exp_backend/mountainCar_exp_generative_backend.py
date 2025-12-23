from .mountainCar_exp_vanilla_backend import MountainCarExpVanillaBackend
from .generative_mixin import GenerativeMixin
from utils import log_flush


# MountainCar 专用的打分 prompt 模板
MOUNTAINCAR_SCORE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent that scores the relevance of past experiences for MountainCar tasks.
Score from 1-10 how helpful this past experience is for the current situation.
Consider: Does this experience help build momentum toward the goal (position >= 0.5)?
Only output a single number.<|eot_id|><|start_header_id|>user<|end_header_id|>

Current situation:
- Position: {query_position:.3f}
- Velocity: {query_velocity:.4f}

Past experience:
- From: position={exp_position:.3f}, velocity={exp_velocity:.4f}
- Action: {action_name} ({action})
- To: position={next_position:.3f}, velocity={next_velocity:.4f}

How relevant is this experience for reaching the goal?
Score (1-10):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class MountainCarExpGenerativeBackend(MountainCarExpVanillaBackend, GenerativeMixin):
    """
    MountainCar 带 Generative 打分排序机制的 Backend
    
    简易版 Generative：检索时用 LLM 对候选经验打分排序，存储和 BFS reachability 逻辑不变。
    """
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, explorer_model=None, log_dir=None):
        """
        Args:
            env_name: 环境名称
            storage_path: 经验存储路径
            depreiciate_exp_store_path: 废弃经验存储路径
            explorer_model: Explorer 模型实例，用于打分
        """
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        self.init_generative(explorer_model=explorer_model)
        
        # 保存原始检索方法
        self._base_retrieve = self.retrieve_experience
        self.retrieve_experience = self.retrieve_experience_with_scoring

    def _build_score_prompt(self, exp: dict, query_state: dict) -> str:
        """构建 MountainCar 的打分 prompt"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', 1)
        
        action_names = ['push left', 'coast', 'push right']
        action_name = action_names[action] if 0 <= action <= 2 else 'unknown'
        
        return MOUNTAINCAR_SCORE_PROMPT.format(
            query_position=query_state.get('position', 0),
            query_velocity=query_state.get('velocity', 0),
            exp_position=st.get('position', 0),
            exp_velocity=st.get('velocity', 0),
            action=action,
            action_name=action_name,
            next_position=st1.get('position', 0),
            next_velocity=st1.get('velocity', 0)
        )

    def retrieve_experience_with_scoring(self, state) -> list:
        """检索经验并用 LLM 打分排序"""
        log_flush(self.logIO, f"[Generative] Retrieving for state: {state}")
        
        # 先用原始方法检索（包含 BFS reachability 等逻辑）
        raw_results = self._base_retrieve(state)
        
        if not raw_results:
            return []
        
        # 用 LLM 打分排序
        ranked_results = self.generative_rank_experiences(
            experiences=raw_results,
            query_state=state,
            build_score_prompt_func=self._build_score_prompt
        )
        
        log_flush(self.logIO, f"[Generative] Retrieved {len(ranked_results)} experiences after scoring")
        return ranked_results

    def retrieve_experience_no_scoring(self, state) -> list:
        """不带打分的检索（用于对比）"""
        return self._base_retrieve(state)

