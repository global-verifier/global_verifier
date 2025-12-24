from .webshop_exp_vanilla_backend import WebshopExpVanillaBackend
from .voyager_mixin import VoyagerMixin
from utils import log_flush


# Webshop 专用的 prompt 模板
WEBSHOP_SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes web shopping navigation experiences.
1) Summarize in 1-2 sentences focusing on the page type, action taken, and result.
2) Note whether this step reached a done/checkout page (finished) or is still navigating (safe/ongoing).
3) Highlight key information like product names, search queries, or purchase decisions.
4) Your response should be a single line of text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this web shopping experience:
- Current URL: {url}
- Action taken: {action}
- Next URL: {next_url}
- Action path so far: {action_path}
- Outcome Status: {outcome_status}

Your summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class WebshopExpVoyagerBackend(WebshopExpVanillaBackend, VoyagerMixin):
    """
    Webshop 带 Voyager 总结机制的 Backend
    
    简易版 Voyager：在存储经验时生成 LLM 总结，检索逻辑保持不变（包括 BFS max_score）。
    """
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, explorer_model=None, log_dir=None):
        """
        Args:
            env_name: 环境名称
            storage_path: 经验存储路径
            depreiciate_exp_store_path: 废弃经验存储路径
            explorer_model: Explorer 模型实例，用于生成总结
        """
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=log_dir)
        self.init_voyager(explorer_model=explorer_model)

    def store_experience(self, exp) -> None:
        """存储经验并生成总结"""
        # 生成总结
        self._voyager_store_webshop_exp(exp)
        # 调用父类存储
        super().store_experience(exp)
        log_flush(self.logIO, f"[Voyager] Experience {exp['id']} stored with summary")

    def _voyager_store_webshop_exp(self, exp: dict) -> None:
        """Webshop 专用的总结生成"""
        st = exp.get('st', {})
        st1 = exp.get('st1', {})
        action = exp.get('action', '')
        action_path = exp.get('action_path', [])
        
        next_url = st1.get('url', 'unknown')
        outcome_status = "finished (done/checkout page)" if 'done' in next_url else "safe/ongoing"
        
        prompt = WEBSHOP_SUMMARY_PROMPT.format(
            url=st.get('url', 'unknown'),
            action=action,
            next_url=next_url,
            action_path=action_path,
            outcome_status=outcome_status
        )
        
        summary = self.voyager_generate_summary(prompt)
        exp['voyager_summary'] = summary
