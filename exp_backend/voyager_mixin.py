"""
Voyager Mixin - 存储时生成 LLM 总结的通用模块

简易版 Voyager：在存储经验时，调用 LLM 生成一条总结 (summary)，
其他检索逻辑保持不变。

使用方式：传入 explorer_model，通过 explorer_model.get_next_action(prompt) 调用 LLM
"""
from utils import log_flush


class VoyagerMixin:
    """
    Voyager 总结机制 Mixin
    
    使用方式：让你的 Backend 类继承这个 Mixin，然后调用 init_voyager()
    
    Example:
        class MyBackend(SomeExpBackend, VoyagerMixin):
            def __init__(self, ..., explorer_model=None):
                super().__init__(...)
                self.init_voyager(explorer_model)
    """
    
    def init_voyager(self, explorer_model=None) -> None:
        """
        初始化 Voyager 参数
        
        Args:
            explorer_model: Explorer 模型实例，通过 get_next_action(prompt) 调用
        """
        self.voyager_model = explorer_model
        
        if self.voyager_model is None:
            log_flush(self.logIO, f"[Voyager] WARNING: No explorer_model provided, summary will be placeholder")
        else:
            log_flush(self.logIO, f"[Voyager] Initialized with explorer_model")

    def voyager_generate_summary(self, prompt: str) -> str:
        """
        调用 LLM 生成总结
        
        Args:
            prompt: 完整的 prompt（包含 system 和 user 部分）
            
        Returns:
            生成的总结字符串
        """
        if self.voyager_model is None:
            return "No explorer_model available for summary generation."
        
        try:
            summary = self.voyager_model.get_next_action(prompt)
            # 清理可能的多余空白
            summary = summary.strip()
            log_flush(self.logIO, f"[Voyager] Generated summary: {summary[:100]}...")
            return summary
        except Exception as e:
            log_flush(self.logIO, f"[Voyager] ERROR generating summary: {e}")
            return f"Summary generation failed: {str(e)}"

    def voyager_get_summary(self, exp_id: str) -> str:
        """获取经验的总结"""
        if exp_id in self.exp_store:
            return self.exp_store[exp_id].get('voyager_summary', '')
        return ''
