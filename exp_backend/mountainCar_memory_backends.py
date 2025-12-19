"""
MountainCar Memory-Enhanced Backends

This module provides MountainCar-specific implementations of the three
memory strategies: MemoryBank, Generative, and Voyager.
"""

from .memorybank_backend import MemoryBankBackend
from .generative_backend import GenerativeBackend
from .voyager_backend import VoyagerBackend
from env_adaptors.base_env_adaptor import BaseEnvAdaptor
from utils import log_flush


class MountainCarMemoryBankBackend(MemoryBankBackend):
    """MountainCar backend with MemoryBank strategy."""
    
    def __init__(
        self,
        env_name: str = "MountainCar",
        storage_path: str = "./storage/mountainCar_memorybank_exp_store.json",
        depreiciate_exp_store_path: str = "./storage/mountainCar_memorybank_deprecated_exp_store.json",
        embedding_func=None,
        memory_persist_dir: str = None,
        forgetting_threshold: float = 0.3,
        llm_model=None
    ):
        # Define MountainCar-specific expected fields
        self.expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": int,  # MountainCar uses integer actions (0, 1, 2)
            "st1": dict,
        }
        
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir,
            forgetting_threshold,
            llm_model
        )
    
    def retrieve_experience(self, state) -> list:
        """Retrieve experiences using MemoryBank strategy with forgetting."""
        try:
            successful_exps, failed_exps = self.retrieve_similar_experiences(
                query_state=state, successful_topk=3, failed_topk=1
            )
            results = successful_exps + failed_exps
            if results:
                log_flush(self.logIO, f"Retrieved {len(results)} experiences via similarity search")
                return results
        except Exception as e:
            log_flush(self.logIO, f"Similarity search failed: {e}")
        
        results = []
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences via exact match")
        return results


class MountainCarGenerativeBackend(GenerativeBackend):
    """MountainCar backend with Generative (LLM scoring) strategy."""
    
    def __init__(
        self,
        env_name: str = "MountainCar",
        storage_path: str = "./storage/mountainCar_generative_exp_store.json",
        depreiciate_exp_store_path: str = "./storage/mountainCar_generative_deprecated_exp_store.json",
        embedding_func=None,
        memory_persist_dir: str = None,
        llm_model=None
    ):
        # Define MountainCar-specific expected fields
        self.expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": int,
            "st1": dict,
        }
        
        # Custom prompts for MountainCar
        system_prompt = """You are an expert at the MountainCar task.
Given a past experience and a current state, rate how useful this experience would be.
Consider:
1. Car position and velocity similarity
2. Whether momentum-building strategies were used
3. Whether the goal was reached
4. The efficiency of the path taken

Provide a score from 0-10."""
        
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir,
            llm_model,
            system_prompt=system_prompt
        )
    
    def retrieve_experience(self, state) -> list:
        """Retrieve experiences using LLM-based scoring."""
        try:
            successful_exps, failed_exps = self.retrieve_similar_experiences(
                query_state=state, successful_topk=3, failed_topk=1
            )
            results = successful_exps + failed_exps
            if results:
                log_flush(self.logIO, f"Retrieved {len(results)} experiences via LLM scoring")
                return results
        except Exception as e:
            log_flush(self.logIO, f"LLM scoring failed: {e}")
        
        results = []
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences via exact match")
        return results


class MountainCarVoyagerBackend(VoyagerBackend):
    """MountainCar backend with Voyager (baseline) strategy."""
    
    def __init__(
        self,
        env_name: str = "MountainCar",
        storage_path: str = "./storage/mountainCar_voyager_exp_store.json",
        depreiciate_exp_store_path: str = "./storage/mountainCar_voyager_deprecated_exp_store.json",
        embedding_func=None,
        memory_persist_dir: str = None,
        llm_model=None,
        use_summarization: bool = False
    ):
        # Define MountainCar-specific expected fields
        self.expected_fields = {
            "id": str,
            "action_path": list,
            "st": dict,
            "action": int,
            "st1": dict,
        }
        
        super().__init__(
            env_name,
            storage_path,
            depreiciate_exp_store_path,
            embedding_func,
            memory_persist_dir,
            llm_model,
            use_summarization
        )
    
    def retrieve_experience(self, state) -> list:
        """Retrieve experiences using pure vector similarity."""
        try:
            successful_exps, failed_exps = self.retrieve_similar_experiences(
                query_state=state, successful_topk=3, failed_topk=1
            )
            results = successful_exps + failed_exps
            if results:
                log_flush(self.logIO, f"Retrieved {len(results)} experiences via vector similarity")
                return results
        except Exception as e:
            log_flush(self.logIO, f"Vector similarity failed: {e}")
        
        results = []
        for exp_id in self.exp_store.keys():
            exp = self.exp_store[exp_id]
            if BaseEnvAdaptor.two_states_equal(state, exp['st']):
                results.append(exp)
        log_flush(self.logIO, f"Retrieved {len(results)} experiences via exact match")
        return results
