from .webshop_exp_vanilla_backend import WebshopExpVanillaBackend
from .generative_mixin import GenerativeMixin
from utils import log_flush


# Webshop 专用的打分 prompt 模板
WEBSHOP_SCORE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent that scores the relevance of past shopping experiences for the current task.
Score from 1-10 how helpful this past experience is for the current situation.
Consider: Does this action lead to finding or purchasing the target product?
Only output a single number.<|eot_id|><|start_header_id|>user<|end_header_id|>

Current situation:
- URL: {query_url}

Past experience:
- From URL: {exp_url}
- Action: {action}
- To URL: {next_url}

How relevant is this experience for the current shopping task?
Score (1-10):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class WebshopExpGenerativeBackend(WebshopExpVanillaBackend, GenerativeMixin):
    """
    Webshop 带 Generative 打分排序机制的 Backend
    
    简易版 Generative：检索时用 LLM 对候选经验打分排序，存储和 BFS max_score 逻辑不变。
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
        """构建 Webshop 的打分 prompt"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', '')
        
        return WEBSHOP_SCORE_PROMPT.format(
            query_url=query_state.get('url', 'unknown'),
            exp_url=st.get('url', 'unknown'),
            action=action,
            next_url=st1.get('url', 'unknown')
        )

    def retrieve_experience_with_scoring(self, state) -> list:
        """检索经验并用 LLM 打分排序"""
        log_flush(self.logIO, f"[Generative] Retrieving for state: {state}")
        
        # 先用原始方法检索（包含 BFS max_score 等逻辑）
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

