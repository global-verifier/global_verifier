"""
Generative Mixin - 检索时用 LLM 对经验打分排序的通用模块

简易版 Generative：在检索经验时，调用 LLM 对候选经验打分，
然后根据分数重新排序返回。存储逻辑保持不变。

使用方式：传入 explorer_model，通过 explorer_model.get_next_action(prompt) 调用 LLM
"""
import re
import copy
from utils import log_flush


class GenerativeMixin:
    """
    Generative 打分排序机制 Mixin
    
    使用方式：让你的 Backend 类继承这个 Mixin，然后调用 init_generative()
    
    Example:
        class MyBackend(SomeExpBackend, GenerativeMixin):
            def __init__(self, ..., explorer_model=None):
                super().__init__(...)
                self.init_generative(explorer_model)
    """
    
    def init_generative(self, explorer_model=None) -> None:
        """
        初始化 Generative 参数
        
        Args:
            explorer_model: Explorer 模型实例，通过 get_next_action(prompt) 调用
        """
        self.generative_model = explorer_model
        
        if self.generative_model is None:
            log_flush(self.logIO, f"[Generative] WARNING: No explorer_model provided, scoring will be skipped")
        else:
            log_flush(self.logIO, f"[Generative] Initialized with explorer_model")

    def generative_score_experience(self, exp: dict, query_state: dict, score_prompt: str) -> int:
        """
        调用 LLM 对单个经验打分
        
        Args:
            exp: 经验字典
            query_state: 当前查询状态
            score_prompt: 完整的打分 prompt
            
        Returns:
            分数 (int)，解析失败返回 0
        """
        if self.generative_model is None:
            return 0
        
        try:
            response = self.generative_model.get_next_action(score_prompt)
            # 从响应中提取数字分数
            match = re.search(r'\d+', response)
            score = int(match.group()) if match else 0
            # 限制分数范围在 1-10
            score = max(1, min(10, score))
            log_flush(self.logIO, f"[Generative] Scored exp {exp.get('id', 'unknown')}: {score}")
            return score
        except Exception as e:
            log_flush(self.logIO, f"[Generative] ERROR scoring exp: {e}")
            return 0

    def generative_rank_experiences(
        self, 
        experiences: list, 
        query_state: dict,
        build_score_prompt_func,
        topk: int = None
    ) -> list:
        """
        对经验列表进行 LLM 打分并排序
        
        Args:
            experiences: 原始经验列表
            query_state: 当前查询状态
            build_score_prompt_func: 构建打分 prompt 的函数，签名: (exp, query_state) -> str
            topk: 返回前 k 个，None 则返回全部
            
        Returns:
            排序后的经验列表（带 'generative_score' 字段）
        """
        if not experiences:
            return []
        
        if self.generative_model is None:
            log_flush(self.logIO, f"[Generative] No model, returning original order")
            return experiences[:topk] if topk else experiences
        
        log_flush(self.logIO, f"[Generative] Scoring {len(experiences)} experiences...")
        
        scored_experiences = []
        for exp in experiences:
            prompt = build_score_prompt_func(exp, query_state)
            score = self.generative_score_experience(exp, query_state, prompt)
            
            exp_copy = copy.deepcopy(exp)
            exp_copy['generative_score'] = score
            scored_experiences.append(exp_copy)
        
        # 按分数降序排序
        scored_experiences.sort(key=lambda x: x['generative_score'], reverse=True)
        
        # 返回 topk
        result = scored_experiences[:topk] if topk else scored_experiences
        
        log_flush(self.logIO, f"[Generative] Ranked {len(result)} experiences, scores: {[e['generative_score'] for e in result]}")
        return result

