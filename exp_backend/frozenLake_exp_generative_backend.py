from .frozenLake_exp_vanilla_backend import FrozenLakeExpVanillaBackend
from .generative_mixin import GenerativeMixin
from utils import log_flush


# FrozenLake 专用的打分 prompt 模板
FROZENLAKE_SCORE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent that scores the relevance of past experiences for navigation tasks.
Score from 1-10 how helpful this past experience is for the current situation.
Only output a single number.<|eot_id|><|start_header_id|>user<|end_header_id|>

Current situation:
- Position: {query_position}

Past experience:
- From Position: {exp_position}
- Action: {action_name} ({action})
- To Position: {next_position}

How relevant is this experience for the current situation?
Score (1-10):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class FrozenLakeExpGenerativeBackend(FrozenLakeExpVanillaBackend, GenerativeMixin):
    """
    FrozenLake 带 Generative 打分排序机制的 Backend
    
    简易版 Generative：检索时用 LLM 对候选经验打分排序，存储逻辑不变。
    """
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, explorer_model=None):
        """
        Args:
            env_name: 环境名称
            storage_path: 经验存储路径
            depreiciate_exp_store_path: 废弃经验存储路径
            explorer_model: Explorer 模型实例，用于打分
        """
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)
        self.init_generative(explorer_model=explorer_model)
        
        # 保存原始检索方法
        self._base_retrieve = self.retrieve_experience
        self.retrieve_experience = self.retrieve_experience_with_scoring

    def _build_score_prompt(self, exp: dict, query_state: dict) -> str:
        """构建 FrozenLake 的打分 prompt"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', 0)
        
        action_names = ['Left', 'Down', 'Right', 'Up']
        action_name = action_names[action] if 0 <= action <= 3 else 'unknown'
        
        return FROZENLAKE_SCORE_PROMPT.format(
            query_position=query_state.get('position', 0),
            exp_position=st.get('position', 0),
            action=action,
            action_name=action_name,
            next_position=st1.get('position', 0)
        )

    def retrieve_experience_with_scoring(self, state) -> list:
        """检索经验并用 LLM 打分排序"""
        log_flush(self.logIO, f"[Generative] Retrieving for state: {state}")
        
        # 先用原始方法检索
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

