"""
Generative Backend Implementation

This implements the Generative strategy from GMemory, which uses LLM-based
scoring to rank and select the most relevant experiences. Retrieved experiences
are re-ranked based on their importance to the current query using language
model reasoning.

Reference: GMemory/mas/memory/mas_memory/generative.py
"""

import re
from typing import List, Tuple, Dict, Optional, Callable

from .memory_enhanced_backend import MemoryEnhancedBackend, ExperienceMessage
from utils import log_flush


class GenerativeBackend(MemoryEnhancedBackend):
    """
    Experience backend with Generative (LLM-based scoring) strategy.
    
    Features:
    - LLM-based relevance scoring for retrieved experiences
    - Re-ranking based on semantic importance
    - Contextual selection of most helpful experiences
    
    This backend is suitable for environments where experiences need to be
    evaluated in context, and where simple similarity may not capture
    true relevance.
    """
    
    # Default prompt templates
    SYSTEM_PROMPT = """You are an expert at evaluating the relevance of past experiences to current scenarios.
Given a past experience trajectory and a current query scenario, rate how useful this experience would be.
Provide a score from 0-10, where:
- 0-2: Not relevant at all
- 3-4: Slightly relevant
- 5-6: Moderately relevant
- 7-8: Very relevant
- 9-10: Extremely relevant

Focus on:
1. Similarity of the starting states
2. Whether the actions taken are applicable
3. Whether the outcome is useful to learn from

Respond with ONLY a single number between 0-10."""
    
    USER_PROMPT_TEMPLATE = """Past Experience:
{trajectory}

Current Query Scenario:
{query_scenario}

How relevant is this past experience to the current scenario? Provide a score (0-10):"""
    
    def __init__(
        self,
        env_name: str,
        storage_path: str,
        depreiciate_exp_store_path: str,
        embedding_func=None,
        memory_persist_dir: str = None,
        llm_model: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None
    ):
        """
        Initialize Generative backend.
        
        Args:
            env_name: Environment name
            storage_path: Path to experience store JSON
            depreiciate_exp_store_path: Path to deprecated store
            embedding_func: Embedding function for vector DB
            memory_persist_dir: Vector DB directory
            llm_model: LLM callable for scoring (required for scoring)
            system_prompt: Custom system prompt for LLM
            user_prompt_template: Custom user prompt template
        """
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir
        )
        
        self.llm_model = llm_model
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.USER_PROMPT_TEMPLATE
        
        if llm_model is None:
            log_flush(
                self.logIO,
                "WARNING: No LLM model provided. Scoring will not be available. "
                "Backend will fall back to similarity-only retrieval."
            )
        
        log_flush(self.logIO, "Generative backend initialized with LLM scoring")
    
    def _score_experience_relevance(
        self,
        exp: dict,
        query_scenario: str
    ) -> float:
        """
        Score an experience's relevance to a query using LLM.
        
        Args:
            exp: Experience dictionary
            query_scenario: Query state/scenario description
            
        Returns:
            Relevance score (0-10), or 0 if scoring fails
        """
        if self.llm_model is None:
            log_flush(self.logIO, "WARNING: Cannot score - no LLM model provided")
            return 5.0  # Default neutral score
        
        try:
            # Format experience as trajectory
            exp_msg = ExperienceMessage(exp)
            trajectory = exp_msg.task_description + exp_msg.task_trajectory
            
            # Create prompt
            user_prompt = self.user_prompt_template.format(
                trajectory=trajectory,
                query_scenario=query_scenario
            )
            
            # Call LLM (assuming it takes list of Message objects)
            # Adapt this based on your actual LLM interface
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = self.llm_model(messages, temperature=0.1)
            
            # Extract score from response
            score_match = re.search(r'\b([0-9]|10)\b', response)
            if score_match:
                score = float(score_match.group(1))
                log_flush(
                    self.logIO,
                    f"Scored experience {exp['id']}: {score}/10"
                )
                return score
            else:
                log_flush(
                    self.logIO,
                    f"WARNING: Could not parse score from LLM response: {response}"
                )
                return 5.0  # Default if parsing fails
                
        except Exception as e:
            log_flush(
                self.logIO,
                f"ERROR scoring experience {exp['id']}: {str(e)}"
            )
            return 0.0
    
    def retrieve_similar_experiences(
        self,
        query_state: dict,
        successful_topk: int = 3,
        failed_topk: int = 1,
        use_llm_scoring: bool = True,
        score_multiplier: int = 2,
        **kwargs
    ) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve and re-rank experiences using LLM scoring.
        
        Args:
            query_state: State to query
            successful_topk: Number of successful experiences to return
            failed_topk: Number of failed experiences to return
            use_llm_scoring: Whether to use LLM for re-ranking
            score_multiplier: Retrieve this many times topk for scoring pool
            
        Returns:
            (successful_exps, failed_exps) sorted by relevance score
        """
        query_scenario = ExperienceMessage._state_to_str(query_state)
        
        # Retrieve more candidates than needed for re-ranking
        if use_llm_scoring and self.llm_model is not None:
            retrieve_success = successful_topk * score_multiplier
            retrieve_fail = failed_topk * score_multiplier
        else:
            retrieve_success = successful_topk
            retrieve_fail = failed_topk
        
        # Get candidates from vector DB
        successful_exps, failed_exps = super().retrieve_similar_experiences(
            query_state,
            retrieve_success,
            retrieve_fail,
            **kwargs
        )
        
        # Re-rank using LLM if available
        if use_llm_scoring and self.llm_model is not None:
            log_flush(
                self.logIO,
                f"Re-ranking {len(successful_exps)} successful experiences with LLM"
            )
            
            # Score each successful experience
            scored_exps = []
            for exp in successful_exps:
                score = self._score_experience_relevance(exp, query_scenario)
                scored_exps.append((score, exp))
            
            # Sort by score (descending) and take top-k
            scored_exps.sort(key=lambda x: x[0], reverse=True)
            successful_exps = [exp for _, exp in scored_exps[:successful_topk]]
            
            log_flush(
                self.logIO,
                f"Selected top {len(successful_exps)} after LLM scoring"
            )
            
            # Failed experiences typically don't need scoring (fewer of them)
            failed_exps = failed_exps[:failed_topk]
        
        return successful_exps, failed_exps
    
    def batch_score_experiences(
        self,
        experiences: List[dict],
        query_scenario: str
    ) -> List[Tuple[float, dict]]:
        """
        Score multiple experiences in batch.
        
        Args:
            experiences: List of experience dicts
            query_scenario: Query description
            
        Returns:
            List of (score, experience) tuples, sorted by score
        """
        scored = []
        
        for exp in experiences:
            score = self._score_experience_relevance(exp, query_scenario)
            scored.append((score, exp))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        log_flush(
            self.logIO,
            f"Batch scored {len(experiences)} experiences. "
            f"Top score: {scored[0][0] if scored else 0:.1f}, "
            f"Avg: {sum(s for s, _ in scored)/len(scored) if scored else 0:.1f}"
        )
        
        return scored
    
    def analyze_scoring_distribution(self) -> Dict:
        """
        Analyze the distribution of relevance scores.
        
        Useful for tuning thresholds and understanding what the LLM
        considers relevant.
        
        Returns:
            Dictionary with statistics about score distribution
        """
        if self.llm_model is None:
            return {'error': 'No LLM model available'}
        
        # Sample some experiences
        sample_ids = list(self.exp_store.keys())[:20]
        
        # Use a generic query
        query = "generic query scenario"
        
        scores = []
        for exp_id in sample_ids:
            exp = self.exp_store[exp_id]
            score = self._score_experience_relevance(exp, query)
            scores.append(score)
        
        if not scores:
            return {'error': 'No scores collected'}
        
        return {
            'sample_size': len(scores),
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'median': sorted(scores)[len(scores)//2]
        }
