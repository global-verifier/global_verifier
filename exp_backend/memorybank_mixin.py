"""
Memory Bank Mixin - 带时间戳遗忘机制的通用模块
"""
import math
import copy
from utils import log_flush
from .backend_config import memorybank_config


class MemoryBankMixin:
    """
    Memory Bank 遗忘机制 Mixin
    
    使用方式：让你的 Backend 类继承这个 Mixin，然后调用 init_memorybank()
    
    Example:
        class MyBackend(SomeExpBackend, MemoryBankMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.init_memorybank()
    """
    
    def init_memorybank(
        self, 
        threshold: float = None, 
        decay_rate: float = None
    ) -> None:
        """
        初始化 Memory Bank 参数
        
        Args:
            threshold: 遗忘阈值，低于此值的记忆被过滤 (默认从 config 读取)
            decay_rate: 衰减速率，越大记忆保持越久 (默认从 config 读取)
        """
        self.mb_threshold: float = threshold if threshold is not None else memorybank_config["threshold"]
        self.mb_decay_rate: float = decay_rate if decay_rate is not None else memorybank_config["decay_rate"]
        self.mb_current_timestep: int = 0
        self.mb_exp_timestamps: dict[str, int] = {}
        
        log_flush(self.logIO, f"[MemoryBank] Initialized with threshold={threshold}, decay_rate={decay_rate}")

    def _forgetting_function(self, time_interval: int) -> float:
        """
        遗忘曲线：exp(-time_interval / decay_rate)
        
        - time_interval = 0 时，返回 1.0（完全记得）
        - time_interval 越大，返回值越小（越容易被遗忘）
        """
        return math.exp(-time_interval / self.mb_decay_rate)

    def mb_store_experience(self, exp) -> None:
        """
        存储经验并记录时间戳
        
        注意：需要先调用父类的 store_experience()
        """
        self.mb_exp_timestamps[exp["id"]] = self.mb_current_timestep
        self.mb_current_timestep += 1
        log_flush(self.logIO, f"[MemoryBank] Stored exp {exp['id']} at timestep {self.mb_current_timestep - 1}")

    def mb_filter_by_forgetting(self, experiences: list) -> list:
        """
        对经验列表应用遗忘过滤
        
        Args:
            experiences: 原始经验列表
            
        Returns:
            过滤后的经验列表，按保留度降序排列，每个经验附加 'retention' 和 'timestep' 字段
        """
        results = []
        
        for exp in experiences:
            exp_id = exp["id"]
            exp_timestep = self.mb_exp_timestamps.get(exp_id, 0)
            time_interval = self.mb_current_timestep - exp_timestep
            retention = self._forgetting_function(time_interval)
            
            if retention >= self.mb_threshold:
                exp_copy = copy.deepcopy(exp)
                exp_copy['retention'] = retention
                exp_copy['timestep'] = exp_timestep
                results.append(exp_copy)
                log_flush(self.logIO, f"  [RETAIN] exp {exp_id}, retention={retention:.3f}, timestep={exp_timestep}")
            else:
                log_flush(self.logIO, f"  [FORGET] exp {exp_id}, retention={retention:.3f}, timestep={exp_timestep}")
        
        # 按保留度排序（记得越清楚的排前面）
        results.sort(key=lambda x: x['retention'], reverse=True)
        
        return results

    def mb_tick(self) -> None:
        """时间流逝一步（每次 step 后调用）"""
        self.mb_current_timestep += 1

    def mb_reset_time(self) -> None:
        """重置时间戳（新 episode 开始时调用）"""
        self.mb_current_timestep = 0
        self.mb_exp_timestamps.clear()
        log_flush(self.logIO, f"[MemoryBank] Time reset")

    def mb_set_params(self, threshold: float = None, decay_rate: float = None) -> None:
        """调整遗忘参数"""
        if threshold is not None:
            self.mb_threshold = threshold
        if decay_rate is not None:
            self.mb_decay_rate = decay_rate
        log_flush(self.logIO, f"[MemoryBank] Updated params: threshold={self.mb_threshold}, decay_rate={self.mb_decay_rate}")

    def mb_get_stats(self) -> dict:
        """获取 Memory Bank 统计信息"""
        return {
            "current_timestep": self.mb_current_timestep,
            "total_tracked": len(self.mb_exp_timestamps),
            "threshold": self.mb_threshold,
            "decay_rate": self.mb_decay_rate,
        }

    def mb_cleanup_forgotten(self) -> list[str]:
        """
        清理被遗忘的经验：删除所有 retention < threshold 的经验
        
        Returns:
            被删除的经验 ID 列表
        """
        forgotten_ids = []
        
        # 遍历所有经验，找出被遗忘的
        for exp_id in list(self.exp_store.keys()):
            exp_timestep = self.mb_exp_timestamps.get(exp_id, 0)
            time_interval = self.mb_current_timestep - exp_timestep
            retention = self._forgetting_function(time_interval)
            
            if retention < self.mb_threshold:
                forgotten_ids.append(exp_id)
        
        # 废弃这些经验
        for exp_id in forgotten_ids:
            log_flush(self.logIO, f"[MemoryBank] Cleanup: deprecating forgotten exp {exp_id}")
            self._deprecate_experience(exp_id)
            # 同时清理时间戳记录
            if exp_id in self.mb_exp_timestamps:
                del self.mb_exp_timestamps[exp_id]
        
        log_flush(self.logIO, f"[MemoryBank] Cleanup done: {len(forgotten_ids)} experiences deprecated")
        return forgotten_ids
