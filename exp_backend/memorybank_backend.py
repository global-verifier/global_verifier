"""
MemoryBank Backend Implementation

This implements the MemoryBank strategy from GMemory, which uses a forgetting
mechanism to manage memory over time. Experiences decay based on how long ago
they occurred, with only the most recent and relevant experiences retained.

Reference: GMemory/mas/memory/mas_memory/memorybank.py
"""

import math
import copy
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from .memory_enhanced_backend import MemoryEnhancedBackend, ExperienceMessage
from utils import log_flush


@dataclass
class ExperienceForgetter:
    """
    Manages forgetting mechanism for experiences based on time.
    
    Experiences are assigned importance scores based on recency,
    with older experiences gradually "forgotten" (filtered out).
    """
    experience_time_pairs: List[Tuple[dict, int]] = field(default_factory=list)
    threshold: float = 0.3  # Minimum importance to retain
    current_time: int = 0
    
    def add_experience(self, exp: dict, time_stamp: int) -> None:
        """Add an experience with timestamp."""
        self.experience_time_pairs.append((exp, time_stamp))
        self.current_time = max(self.current_time, time_stamp)
    
    def manage_memory(self) -> List[Tuple[dict, int]]:
        """
        Filter experiences based on forgetting function.
        
        Returns:
            List of (experience, timestamp) pairs that pass threshold
        """
        if len(self.experience_time_pairs) == 0:
            return []
        
        max_time = self.experience_time_pairs[-1][1]
        
        # Apply forgetting function
        filtered_pairs = []
        for exp, time_stamp in self.experience_time_pairs:
            time_interval = max_time - time_stamp
            importance = self._forgetting_function(time_interval)
            
            if importance >= self.threshold:
                filtered_pairs.append((exp, time_stamp))
        
        self.experience_time_pairs = filtered_pairs
        return copy.deepcopy(filtered_pairs)
    
    def _forgetting_function(self, time_interval: float, scale: float = 1.0) -> float:
        """
        Exponential decay function for memory importance.
        
        Args:
            time_interval: Time since experience occurred
            scale: Scaling factor for decay rate
            
        Returns:
            Importance score between 0 and 1
        """
        return math.exp(-time_interval / 5.0 * scale)
    
    def clear(self) -> None:
        """Clear all memories."""
        self.current_time = 0
        self.experience_time_pairs = []
    
    def get_active_experiences(self) -> List[dict]:
        """Get all experiences above threshold."""
        pairs = self.manage_memory()
        return [exp for exp, _ in pairs]


class MemoryBankBackend(MemoryEnhancedBackend):
    """
    Experience backend with MemoryBank strategy.
    
    Features:
    - Time-based forgetting mechanism
    - Exponential decay of older experiences
    - Automatic memory management to prevent overflow
    
    This backend is suitable for environments where recent experiences
    are more valuable than older ones, and where experience relevance
    naturally decays over time.
    """
    
    def __init__(
        self,
        env_name: str,
        storage_path: str,
        depreiciate_exp_store_path: str,
        embedding_func=None,
        memory_persist_dir: str = None,
        forgetting_threshold: float = 0.3,
        llm_model=None
    ):
        """
        Initialize MemoryBank backend.
        
        Args:
            env_name: Environment name
            storage_path: Path to experience store JSON
            depreiciate_exp_store_path: Path to deprecated store
            embedding_func: Embedding function for vector DB
            memory_persist_dir: Vector DB directory
            forgetting_threshold: Minimum importance to retain (0-1)
            llm_model: LLM for summarization (optional)
        """
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir
        )
        
        self.forgetter = ExperienceForgetter(threshold=forgetting_threshold)
        self.llm_model = llm_model
        self.time_counter = 0
        
        log_flush(
            self.logIO,
            f"MemoryBank backend initialized with threshold={forgetting_threshold}"
        )
    
    def store_experience(self, exp: dict):
        """
        Store experience with timestamp for forgetting mechanism.
        
        Args:
            exp: Experience with label
        """
        # Add to forgetter with timestamp
        self.forgetter.add_experience(exp, self.time_counter)
        self.time_counter += 1
        
        # Store in parent (JSON + vector DB)
        super().store_experience(exp)
        
        log_flush(self.logIO, f"Stored experience {exp['id']} at time {self.time_counter-1}")
    
    def retrieve_similar_experiences(
        self,
        query_state: dict,
        successful_topk: int = 3,
        failed_topk: int = 1,
        use_forgetting: bool = True,
        **kwargs
    ) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve similar experiences with forgetting mechanism applied.
        
        Args:
            query_state: State to query
            successful_topk: Number of successful experiences
            failed_topk: Number of failed experiences
            use_forgetting: Whether to apply forgetting filter
            
        Returns:
            (successful_exps, failed_exps) filtered by recency
        """
        # Get similar experiences from vector DB
        successful_exps, failed_exps = super().retrieve_similar_experiences(
            query_state,
            successful_topk * 2 if use_forgetting else successful_topk,
            failed_topk * 2 if use_forgetting else failed_topk,
            **kwargs
        )
        
        # Apply forgetting mechanism if enabled
        if use_forgetting:
            # Get active experience IDs
            active_pairs = self.forgetter.manage_memory()
            active_ids = {exp['id'] for exp, _ in active_pairs}
            
            # Filter retrieved experiences
            successful_exps = [
                exp for exp in successful_exps 
                if exp['id'] in active_ids
            ][:successful_topk]
            
            failed_exps = [
                exp for exp in failed_exps 
                if exp['id'] in active_ids
            ][:failed_topk]
            
            log_flush(
                self.logIO,
                f"After forgetting filter: {len(successful_exps)} successful, "
                f"{len(failed_exps)} failed (from {len(active_ids)} active)"
            )
        
        return successful_exps, failed_exps
    
    def get_memory_statistics(self) -> Dict:
        """Get statistics about memory state."""
        active_pairs = self.forgetter.manage_memory()
        
        total_exps = len(self.exp_store)
        active_exps = len(active_pairs)
        forgotten_exps = total_exps - active_exps
        
        # Calculate average importance
        if active_pairs:
            max_time = self.time_counter - 1
            avg_importance = sum(
                self.forgetter._forgetting_function(max_time - t)
                for _, t in active_pairs
            ) / len(active_pairs)
        else:
            avg_importance = 0.0
        
        stats = {
            'total_experiences': total_exps,
            'active_experiences': active_exps,
            'forgotten_experiences': forgotten_exps,
            'current_time': self.time_counter,
            'average_importance': avg_importance,
            'threshold': self.forgetter.threshold
        }
        
        return stats
    
    def cleanup_forgotten(self):
        """
        Remove experiences that fall below threshold from storage.
        
        WARNING: This permanently removes forgotten experiences.
        Use with caution - consider deprecating instead.
        """
        active_pairs = self.forgetter.manage_memory()
        active_ids = {exp['id'] for exp, _ in active_pairs}
        
        # Find forgotten experiences
        forgotten_ids = [
            exp_id for exp_id in self.exp_store.keys()
            if exp_id not in active_ids
        ]
        
        log_flush(
            self.logIO,
            f"Cleaning up {len(forgotten_ids)} forgotten experiences"
        )
        
        # Deprecate (not delete) forgotten experiences
        for exp_id in forgotten_ids:
            self._deprecate_experience(exp_id)
        
        log_flush(self.logIO, f"Cleanup completed")
