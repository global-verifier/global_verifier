"""
Memory Backend Demo

This script demonstrates how to use the three memory strategies
(MemoryBank, Generative, Voyager) with the global_verifier backend.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp_backend.backend_factory import create_backend


def simple_llm_mock(messages, temperature=0.1):
    """
    Mock LLM for demonstration purposes.
    
    In production, replace this with your actual LLM implementation
    (e.g., OpenAI API, local model, etc.)
    """
    # Extract the user message
    user_msg = ""
    for msg in messages:
        if msg.get('role') == 'user':
            user_msg = msg.get('content', '')
    
    # Simple heuristic scoring based on keywords
    if 'successful' in user_msg.lower() or 'success' in user_msg.lower():
        return "8"  # High score for successful experiences
    elif 'failed' in user_msg.lower() or 'fail' in user_msg.lower():
        return "3"  # Low score for failed experiences
    else:
        return "5"  # Neutral score


def demo_memorybank():
    """Demonstrate MemoryBank strategy."""
    print("\n" + "="*80)
    print("DEMO: MemoryBank Strategy (Forgetting Mechanism)")
    print("="*80)
    
    # Create backend
    backend = create_backend(
        env_name="CartPole",
        memory_strategy="memorybank",
        forgetting_threshold=0.3
    )
    
    # Store some experiences with timestamps
    print("\n1. Storing experiences over time...")
    experiences = [
        {
            "id": f"exp_mb_{i}",
            "action_path": [0, 1] * i,
            "st": {"position": 0.0, "velocity": 0.0},
            "action": i % 2,
            "st1": {"position": 0.1 * i, "velocity": 0.05 * i},
            "label": True if i % 2 == 0 else False
        }
        for i in range(10)
    ]
    
    for exp in experiences:
        backend.store_experience(exp)
        print(f"  Stored: {exp['id']}")
    
    # Check memory statistics
    print("\n2. Memory statistics:")
    stats = backend.get_memory_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Retrieve similar experiences
    print("\n3. Retrieving similar experiences:")
    query_state = {"position": 0.0, "velocity": 0.0}
    successful, failed = backend.retrieve_similar_experiences(
        query_state,
        successful_topk=3,
        failed_topk=1,
        use_forgetting=True
    )
    
    print(f"  Retrieved {len(successful)} successful experiences")
    print(f"  Retrieved {len(failed)} failed experiences")
    
    # Close
    backend.logIO.close()
    print("\n✓ MemoryBank demo completed")


def demo_generative():
    """Demonstrate Generative strategy."""
    print("\n" + "="*80)
    print("DEMO: Generative Strategy (LLM Scoring)")
    print("="*80)
    
    # Create backend with mock LLM
    backend = create_backend(
        env_name="MountainCar",
        memory_strategy="generative",
        llm_model=simple_llm_mock
    )
    
    # Store some experiences
    print("\n1. Storing labeled experiences...")
    experiences = [
        {
            "id": f"exp_gen_{i}",
            "action_path": [0, 1, 2] * i,
            "st": {"position": -0.5, "velocity": 0.0},
            "action": i % 3,
            "st1": {"position": -0.4 + 0.1*i, "velocity": 0.01*i},
            "label": True if i > 5 else False
        }
        for i in range(10)
    ]
    
    for exp in experiences:
        backend.store_experience(exp)
        print(f"  Stored: {exp['id']} (label: {exp['label']})")
    
    # Retrieve with LLM scoring
    print("\n2. Retrieving with LLM scoring:")
    query_state = {"position": -0.5, "velocity": 0.0}
    
    # With LLM scoring
    print("  With LLM scoring enabled:")
    successful, failed = backend.retrieve_similar_experiences(
        query_state,
        successful_topk=3,
        failed_topk=1,
        use_llm_scoring=True,
        score_multiplier=2
    )
    print(f"    Retrieved {len(successful)} successful experiences")
    for exp in successful:
        print(f"      - {exp['id']}")
    
    # Without LLM scoring (for comparison)
    print("  Without LLM scoring (similarity only):")
    successful, failed = backend.retrieve_similar_experiences(
        query_state,
        successful_topk=3,
        failed_topk=1,
        use_llm_scoring=False
    )
    print(f"    Retrieved {len(successful)} successful experiences")
    for exp in successful:
        print(f"      - {exp['id']}")
    
    # Close
    backend.logIO.close()
    print("\n✓ Generative demo completed")


def demo_voyager():
    """Demonstrate Voyager strategy."""
    print("\n" + "="*80)
    print("DEMO: Voyager Strategy (Baseline Similarity)")
    print("="*80)
    
    # Create backend
    backend = create_backend(
        env_name="Webshop",
        memory_strategy="voyager",
        use_summarization=False
    )
    
    # Store some experiences
    print("\n1. Storing diverse experiences...")
    experiences = [
        {
            "id": f"exp_voy_{i}",
            "action_path": [f"search_{i}", f"click_{i}", f"buy_{i}"],
            "st": {"url": f"shop.com/page{i}", "html_text": f"page {i} content"},
            "action": f"search_product_{i}",
            "st1": {"url": f"shop.com/result{i}", "html_text": f"result {i}"},
            "label": True if i % 3 == 0 else False
        }
        for i in range(15)
    ]
    
    for exp in experiences:
        backend.store_experience(exp)
        print(f"  Stored: {exp['id']}")
    
    # Get statistics
    print("\n2. Backend statistics:")
    stats = backend.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Simple retrieval
    print("\n3. Retrieving similar experiences:")
    query_state = {"url": "shop.com/page5", "html_text": "page 5 content"}
    successful, failed = backend.retrieve_similar_experiences(
        query_state,
        successful_topk=5,
        failed_topk=2
    )
    
    print(f"  Retrieved {len(successful)} successful experiences:")
    for exp in successful:
        print(f"    - {exp['id']}")
    
    print(f"  Retrieved {len(failed)} failed experiences:")
    for exp in failed:
        print(f"    - {exp['id']}")
    
    # Close
    backend.logIO.close()
    print("\n✓ Voyager demo completed")


def demo_comparison():
    """Compare all three strategies side-by-side."""
    print("\n" + "="*80)
    print("DEMO: Strategy Comparison")
    print("="*80)
    
    # Create all three backends
    backends = {
        "MemoryBank": create_backend("CartPole", "memorybank"),
        "Generative": create_backend("CartPole", "generative", llm_model=simple_llm_mock),
        "Voyager": create_backend("CartPole", "voyager")
    }
    
    # Store same experiences in all backends
    print("\n1. Storing 20 experiences in each backend...")
    experiences = [
        {
            "id": f"exp_comp_{i}",
            "action_path": [0, 1] * (i % 5),
            "st": {"position": 0.0 + 0.1*i, "velocity": 0.0 + 0.01*i},
            "action": i % 2,
            "st1": {"position": 0.1 + 0.1*i, "velocity": 0.01 + 0.01*i},
            "label": True if i % 3 != 0 else False
        }
        for i in range(20)
    ]
    
    for name, backend in backends.items():
        for exp in experiences:
            backend.store_experience(exp)
        print(f"  {name}: Stored {len(experiences)} experiences")
    
    # Query all backends
    print("\n2. Querying all backends with same state:")
    query_state = {"position": 0.5, "velocity": 0.05}
    
    results = {}
    for name, backend in backends.items():
        successful, failed = backend.retrieve_similar_experiences(
            query_state,
            successful_topk=3,
            failed_topk=1
        )
        results[name] = {
            "successful": [exp['id'] for exp in successful],
            "failed": [exp['id'] for exp in failed]
        }
    
    # Display results
    print("\n3. Retrieval results:")
    for name, result in results.items():
        print(f"\n  {name}:")
        print(f"    Successful: {result['successful']}")
        print(f"    Failed: {result['failed']}")
    
    # Clean up
    for backend in backends.values():
        backend.logIO.close()
    
    print("\n✓ Comparison demo completed")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("Memory-Enhanced Backend Demonstrations")
    print("="*80)
    
    try:
        demo_memorybank()
        demo_generative()
        demo_voyager()
        demo_comparison()
        
        print("\n" + "="*80)
        print("All demos completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
