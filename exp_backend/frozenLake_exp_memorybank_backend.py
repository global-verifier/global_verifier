from typing import List
from .frozenLake_exp_vanilla_backend import FrozenLakeExpVanillaBackend
from .memorybank_mixin import MemoryBankMixin
from utils import log_flush
from env_adaptors.base_env_adaptor import BaseEnvAdaptor


class FrozenLakeExpMemoryBankBackend(FrozenLakeExpVanillaBackend, MemoryBankMixin):
    """
    FrozenLake 带时间戳遗忘机制的 Memory Bank Backend
    继承 VanillaBackend 以复用其检索算法
    """
    def __init__(self, env_name, storage_path, depreiciate_exp_store_path, **kwargs):
        super().__init__(env_name, storage_path, depreiciate_exp_store_path, log_dir=kwargs.get("log_dir"))
        self.init_memorybank(
            threshold=kwargs.get("threshold"),
            decay_rate=kwargs.get("decay_rate"),
            start_timestep=kwargs.get("start_timestep", 0),
        )
        
        # 保存原始检索方法，然后包装
        self._base_retrieve = self.retrieve_experience
        self.retrieve_experience = self.retrieve_experience_with_forgetting

    def store_experience(self, exp) -> None:
        """存储经验并记录时间戳"""
        self.mb_store_experience(exp)
        super().store_experience(exp)

    def retrieve_experience_with_forgetting(self, state) -> list:
        """检索经验，应用遗忘机制过滤"""
        log_flush(self.logIO, f"[MemoryBank] Retrieving for state: {state}, timestep: {self.mb_current_timestep}")
        
        # 用 vanilla 的方法检索
        raw_results = self._base_retrieve(state)
        
        # 应用遗忘过滤
        results = self.mb_filter_by_forgetting(raw_results)
        
        log_flush(self.logIO, f"[MemoryBank] Retrieved {len(results)}/{len(raw_results)} after forgetting filter")
        return results

    def retrieve_experience_no_forgetting(self, state) -> list:
        """不带遗忘的检索（用于对比）"""
        return self._base_retrieve(state)

    def step(self):
        self.mb_cleanup_forgotten()
        self.mb_tick()
        log_flush(self.logIO, f"[MemoryBank] Status: {self.export_status()}")

    def export_status(self):
        return {"mb_current_timestep": self.mb_current_timestep}

    def finish_explore_trail(self, exp_ids: List[str]) -> None:
        """完成一次成功的探索轨迹，调用 mixin 的方法"""
        self.mb_finish_explore_trail(exp_ids)
