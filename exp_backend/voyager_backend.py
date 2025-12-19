"""
Voyager Backend Implementation

This implements the Voyager strategy from GMemory, which uses straightforward
vector similarity-based retrieval without additional mechanisms. This serves
as a clean baseline for comparison with more complex strategies.

Reference: GMemory/mas/memory/mas_memory/voyager.py
"""

from typing import List, Tuple, Optional, Callable

from .memory_enhanced_backend import MemoryEnhancedBackend, ExperienceMessage
from utils import log_flush


class VoyagerBackend(MemoryEnhancedBackend):
    """
    Experience backend with Voyager (baseline similarity) strategy.
    
    Features:
    - Pure vector similarity-based retrieval
    - Optional LLM-based summarization of experiences
    - Simple and efficient baseline approach
    
    This backend is suitable as a baseline for comparison, or for
    environments where simple similarity matching is sufficient.
    """
    
    # Default prompt templates for summarization
    SYSTEM_INSTRUCTION = """You are an expert at summarizing experiences in reinforcement learning environments.
Given a task trajectory, provide a concise summary that captures:
1. The initial state/goal
2. The key actions taken
3. The outcome achieved

Keep the summary brief (2-3 sentences) but informative."""
    
    USER_INSTRUCTION_TEMPLATE = """Summarize the following task trajectory:

{task_trajectory}

Provide a concise summary:"""
    
    def __init__(
        self,
        env_name: str,
        storage_path: str,
        depreiciate_exp_store_path: str,
        embedding_func=None,
        memory_persist_dir: str = None,
        llm_model: Optional[Callable] = None,
        use_summarization: bool = False,
        system_instruction: Optional[str] = None,
        user_instruction_template: Optional[str] = None
    ):
        """
        Initialize Voyager backend.
        
        Args:
            env_name: Environment name
            storage_path: Path to experience store JSON
            depreiciate_exp_store_path: Path to deprecated store
            embedding_func: Embedding function for vector DB
            memory_persist_dir: Vector DB directory
            llm_model: LLM for summarization (optional)
            use_summarization: Whether to use LLM for summarization
            system_instruction: Custom system instruction
            user_instruction_template: Custom user instruction template
        """
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir
        )
        
        self.llm_model = llm_model
        self.use_summarization = use_summarization
        self.system_instruction = system_instruction or self.SYSTEM_INSTRUCTION
        self.user_instruction_template = user_instruction_template or self.USER_INSTRUCTION_TEMPLATE
        
        if use_summarization and llm_model is None:
            log_flush(
                self.logIO,
                "WARNING: Summarization enabled but no LLM model provided. "
                "Falling back to default formatting."
            )
        
        log_flush(
            self.logIO,
            f"Voyager backend initialized (summarization: {use_summarization})"
        )
    
    def _summarize_experience(self, exp: dict) -> str:
        """
        Generate summary of an experience using LLM.
        
        Args:
            exp: Experience dictionary
            
        Returns:
            Summary string
        """
        if not self.use_summarization or self.llm_model is None:
            # Use default formatting
            exp_msg = ExperienceMessage(exp)
            return exp_msg.task_main
        
        try:
            # Format trajectory
            exp_msg = ExperienceMessage(exp)
            trajectory = exp_msg.task_description + exp_msg.task_trajectory
            
            # Create prompt
            user_prompt = self.user_instruction_template.format(
                task_trajectory=trajectory
            )
            
            # Call LLM
            messages = [
                {'role': 'system', 'content': self.system_instruction},
                {'role': 'user', 'content': user_prompt}
            ]
            
            summary = self.llm_model(messages, temperature=0.1)
            
            log_flush(
                self.logIO,
                f"Generated summary for experience {exp['id']}"
            )
            
            return summary
            
        except Exception as e:
            log_flush(
                self.logIO,
                f"ERROR summarizing experience {exp['id']}: {str(e)}"
            )
            # Fall back to default
            exp_msg = ExperienceMessage(exp)
            return exp_msg.task_main
    
    def store_experience(self, exp: dict):
        """
        Store experience with optional summarization.
        
        Args:
            exp: Experience with label
        """
        # Generate summary if enabled
        if self.use_summarization:
            try:
                summary = self._summarize_experience(exp)
                # Store summary in a custom field
                exp['_summary'] = summary
            except Exception as e:
                log_flush(
                    self.logIO,
                    f"WARNING: Failed to generate summary: {str(e)}"
                )
        
        # Store in parent
        super().store_experience(exp)
    
    def retrieve_similar_experiences(
        self,
        query_state: dict,
        successful_topk: int = 3,
        failed_topk: int = 1,
        **kwargs
    ) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve similar experiences using pure vector similarity.
        
        This is the baseline approach - no re-ranking, no filtering,
        just straightforward similarity search.
        
        Args:
            query_state: State to query
            successful_topk: Number of successful experiences
            failed_topk: Number of failed experiences
            
        Returns:
            (successful_exps, failed_exps) sorted by similarity
        """
        # Use parent's implementation (pure vector similarity)
        successful_exps, failed_exps = super().retrieve_similar_experiences(
            query_state,
            successful_topk,
            failed_topk,
            **kwargs
        )
        
        log_flush(
            self.logIO,
            f"Voyager retrieval: {len(successful_exps)} successful, "
            f"{len(failed_exps)} failed (pure similarity)"
        )
        
        return successful_exps, failed_exps
    
    def get_all_successful_experiences(self) -> List[dict]:
        """
        Get all successful experiences without similarity filtering.
        
        Returns:
            List of all successful experiences
        """
        return [
            exp for exp in self.exp_store.values()
            if exp.get('label') == True
        ]
    
    def get_all_failed_experiences(self) -> List[dict]:
        """
        Get all failed experiences without similarity filtering.
        
        Returns:
            List of all failed experiences
        """
        return [
            exp for exp in self.exp_store.values()
            if exp.get('label') == False
        ]
    
    def get_statistics(self) -> dict:
        """
        Get basic statistics about stored experiences.
        
        Returns:
            Dictionary with counts and ratios
        """
        total = len(self.exp_store)
        successful = len(self.get_all_successful_experiences())
        failed = len(self.get_all_failed_experiences())
        unlabeled = total - successful - failed
        deprecated = len(self.depreiciate_exp_store)
        
        return {
            'total_experiences': total,
            'successful_experiences': successful,
            'failed_experiences': failed,
            'unlabeled_experiences': unlabeled,
            'deprecated_experiences': deprecated,
            'success_rate': successful / total if total > 0 else 0.0,
            'using_summarization': self.use_summarization
        }
