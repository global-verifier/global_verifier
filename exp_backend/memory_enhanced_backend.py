"""
Memory-Enhanced Backend Base Class

This module provides a bridge between global_verifier's experience backend
and GMemory's memory management systems (MemoryBank, Generative, Voyager).

The MemoryEnhancedBackend extends BaseExpBackend with vector-based memory
retrieval capabilities while maintaining the original conflict resolution
and experience validation features.
"""

import os
from typing import List, Tuple, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base_exp_backend import BaseExpBackend
from utils import log_flush, get_timestamp


class ExperienceMessage:
    """
    Adapter class to bridge global_verifier's experience format
    with GMemory's MASMessage format.
    """
    def __init__(self, exp: dict):
        self.exp_id = exp['id']
        self.action_path = exp['action_path']
        self.st = exp['st']
        self.action = exp['action']
        self.st1 = exp['st1']
        
        # Create unified format for memory storage
        self.task_main = self._format_task_main(exp)
        self.task_description = self._format_task_description(exp)
        self.task_trajectory = self._format_task_trajectory(exp)
        self.label = exp.get('label', None)  # True for successful, False for failed
        
    def _format_task_main(self, exp: dict) -> str:
        """Create a summary of the experience for embedding."""
        return f"Experience {exp['id']}: Action path length {len(exp['action_path'])}"
    
    def _format_task_description(self, exp: dict) -> str:
        """Format the initial state and context."""
        return f"Start state: {self._state_to_str(exp['st'])}\nAction: {exp['action']}"
    
    def _format_task_trajectory(self, exp: dict) -> str:
        """Format the action path and result."""
        path_str = " -> ".join([str(a) for a in exp['action_path']])
        return f"\nPath: {path_str}\nResult state: {self._state_to_str(exp['st1'])}"
    
    @staticmethod
    def _state_to_str(state: dict) -> str:
        """Convert state dict to string representation."""
        if isinstance(state, dict):
            return str(state)
        return str(state)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return {
            'exp_id': self.exp_id,
            'action_path': self.action_path,
            'st': self.st,
            'action': self.action,
            'st1': self.st1,
            'task_main': self.task_main,
            'task_description': self.task_description,
            'task_trajectory': self.task_trajectory,
            'label': self.label
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperienceMessage':
        """Reconstruct from metadata."""
        exp = {
            'id': data['exp_id'],
            'action_path': data['action_path'],
            'st': data['st'],
            'action': data['action'],
            'st1': data['st1'],
            'label': data.get('label')
        }
        msg = cls(exp)
        msg.task_main = data.get('task_main', msg.task_main)
        msg.task_description = data.get('task_description', msg.task_description)
        msg.task_trajectory = data.get('task_trajectory', msg.task_trajectory)
        return msg


class MemoryEnhancedBackend(BaseExpBackend):
    """
    Base class for memory-enhanced experience backends.
    
    This extends BaseExpBackend with vector-based memory retrieval,
    allowing experiences to be retrieved based on semantic similarity
    rather than just exact state matching.
    """
    
    def __init__(
        self, 
        env_name: str, 
        storage_path: str,
        depreiciate_exp_store_path: str,
        embedding_func=None,
        memory_persist_dir: str = None
    ):
        """
        Initialize memory-enhanced backend.
        
        Args:
            env_name: Environment name
            storage_path: Path to experience store JSON
            depreiciate_exp_store_path: Path to deprecated experience store JSON
            embedding_func: Embedding function for vector database (optional)
            memory_persist_dir: Directory for vector database persistence (optional)
        """
        super().__init__(env_name, storage_path, depreiciate_exp_store_path)
        
        # Set up embedding function
        if embedding_func is None:
            # Use default embedding from langchain
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_func = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embedding_func = embedding_func
        
        # Set up vector database
        if memory_persist_dir is None:
            memory_persist_dir = os.path.join(
                os.path.dirname(storage_path), 
                f"memory_db_{env_name}"
            )
        self.memory_persist_dir = memory_persist_dir
        os.makedirs(self.memory_persist_dir, exist_ok=True)
        
        self.vector_memory = Chroma(
            embedding_function=self.embedding_func,
            persist_directory=self.memory_persist_dir
        )
        
        log_flush(self.logIO, f"Memory-enhanced backend initialized with vector DB at {memory_persist_dir}")
        
        # Sync existing experiences to vector database
        self._sync_to_vector_db()
    
    def _sync_to_vector_db(self):
        """Sync existing experiences from JSON store to vector database."""
        log_flush(self.logIO, f"Syncing {len(self.exp_store)} experiences to vector database...")
        
        for exp_id, exp in self.exp_store.items():
            if exp.get('label') is not None:  # Only sync labeled experiences
                try:
                    self._add_to_vector_db(exp)
                except Exception as e:
                    log_flush(self.logIO, f"WARNING: Failed to sync exp {exp_id}: {str(e)}")
        
        log_flush(self.logIO, f"Vector database sync completed")
    
    def _add_to_vector_db(self, exp: dict):
        """Add a single experience to vector database."""
        exp_msg = ExperienceMessage(exp)
        
        # Create document for vector storage
        doc = Document(
            page_content=exp_msg.task_main,
            metadata=exp_msg.to_dict()
        )
        
        # Check if already exists (avoid duplicates)
        existing = self.vector_memory.get(where={"exp_id": exp['id']})
        if existing and existing['ids']:
            log_flush(self.logIO, f"Experience {exp['id']} already in vector DB, skipping")
            return
        
        self.vector_memory.add_documents([doc])
        log_flush(self.logIO, f"Added experience {exp['id']} to vector DB")
    
    def store_experience(self, exp: dict):
        """
        Override to also store in vector database.
        
        Args:
            exp: Experience dictionary with label (True/False)
        """
        # Store in JSON (parent method)
        super().store_experience(exp)
        
        # Store in vector DB if labeled
        if exp.get('label') is not None:
            try:
                self._add_to_vector_db(exp)
            except Exception as e:
                log_flush(self.logIO, f"WARNING: Failed to add exp to vector DB: {str(e)}")
    
    def retrieve_similar_experiences(
        self,
        query_state: dict,
        successful_topk: int = 3,
        failed_topk: int = 1,
        **kwargs
    ) -> Tuple[List[dict], List[dict]]:
        """
        Retrieve experiences similar to the query state using vector search.
        
        Args:
            query_state: The state to query for similar experiences
            successful_topk: Number of successful experiences to retrieve
            failed_topk: Number of failed experiences to retrieve
            
        Returns:
            Tuple of (successful_experiences, failed_experiences)
        """
        # Format query
        query_text = ExperienceMessage._state_to_str(query_state)
        
        successful_exps = []
        failed_exps = []
        
        # Retrieve successful experiences
        if successful_topk > 0:
            try:
                docs = self.vector_memory.similarity_search_with_score(
                    query=query_text,
                    k=successful_topk,
                    filter={'label': True}
                )
                successful_exps = [
                    self._reconstruct_exp_from_metadata(doc[0].metadata)
                    for doc in docs
                ]
            except Exception as e:
                log_flush(self.logIO, f"WARNING: Failed to retrieve successful experiences: {str(e)}")
        
        # Retrieve failed experiences
        if failed_topk > 0:
            try:
                docs = self.vector_memory.similarity_search_with_score(
                    query=query_text,
                    k=failed_topk,
                    filter={'label': False}
                )
                failed_exps = [
                    self._reconstruct_exp_from_metadata(doc[0].metadata)
                    for doc in docs
                ]
            except Exception as e:
                log_flush(self.logIO, f"WARNING: Failed to retrieve failed experiences: {str(e)}")
        
        log_flush(
            self.logIO, 
            f"Retrieved {len(successful_exps)} successful and {len(failed_exps)} failed experiences"
        )
        
        return successful_exps, failed_exps
    
    def _reconstruct_exp_from_metadata(self, metadata: dict) -> dict:
        """Reconstruct experience from vector DB metadata."""
        return {
            'id': metadata['exp_id'],
            'action_path': metadata['action_path'],
            'st': metadata['st'],
            'action': metadata['action'],
            'st1': metadata['st1'],
            'label': metadata.get('label')
        }
    
    def _deprecate_experience(self, exp_id: str):
        """
        Override to also remove from vector database.
        
        Args:
            exp_id: Experience ID to deprecate
        """
        # Remove from vector DB
        try:
            results = self.vector_memory.get(where={"exp_id": exp_id})
            if results and results['ids']:
                self.vector_memory.delete(ids=results['ids'])
                log_flush(self.logIO, f"Removed experience {exp_id} from vector DB")
        except Exception as e:
            log_flush(self.logIO, f"WARNING: Failed to remove exp from vector DB: {str(e)}")
        
        # Call parent to handle JSON store
        super()._deprecate_experience(exp_id)
