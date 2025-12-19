"""
Strategy Comparison Tool

This script compares the three memory strategies (MemoryBank, Generative, Voyager)
on metrics like retrieval accuracy, speed, and memory usage.
"""

import sys
import os
import time
import json
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp_backend.backend_factory import create_backend


def generate_test_experiences(env_name: str, count: int) -> List[Dict]:
    """Generate test experiences for benchmarking."""
    experiences = []
    
    if env_name.lower() == "cartpole":
        for i in range(count):
            experiences.append({
                "id": f"test_exp_{i}",
                "action_path": [0, 1] * (i % 10),
                "st": {
                    "cart_position": -1.0 + 2.0 * (i / count),
                    "cart_velocity": -0.5 + 1.0 * (i / count),
                    "pole_angle": -0.2 + 0.4 * (i / count),
                    "pole_velocity": -1.0 + 2.0 * (i / count)
                },
                "action": i % 2,
                "st1": {
                    "cart_position": -1.0 + 2.0 * ((i+1) / count),
                    "cart_velocity": -0.5 + 1.0 * ((i+1) / count),
                    "pole_angle": -0.2 + 0.4 * ((i+1) / count),
                    "pole_velocity": -1.0 + 2.0 * ((i+1) / count)
                },
                "label": True if i % 3 != 0 else False
            })
    
    return experiences


def benchmark_storage(backend, experiences: List[Dict]) -> Dict:
    """Benchmark storage performance."""
    start_time = time.time()
    
    for exp in experiences:
        backend.store_experience(exp)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "total_time": total_time,
        "avg_time_per_exp": total_time / len(experiences),
        "experiences_per_second": len(experiences) / total_time
    }


def benchmark_retrieval(backend, query_states: List[Dict], topk: int = 5) -> Dict:
    """Benchmark retrieval performance."""
    retrieval_times = []
    results = []
    
    for query_state in query_states:
        start_time = time.time()
        
        successful, failed = backend.retrieve_similar_experiences(
            query_state,
            successful_topk=topk,
            failed_topk=1
        )
        
        end_time = time.time()
        retrieval_times.append(end_time - start_time)
        results.append((successful, failed))
    
    return {
        "total_queries": len(query_states),
        "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
        "min_retrieval_time": min(retrieval_times),
        "max_retrieval_time": max(retrieval_times),
        "queries_per_second": len(query_states) / sum(retrieval_times)
    }


def compare_strategies(
    env_name: str = "CartPole",
    num_experiences: int = 100,
    num_queries: int = 20
):
    """
    Compare all three memory strategies.
    
    Args:
        env_name: Environment to test
        num_experiences: Number of experiences to store
        num_queries: Number of retrieval queries to run
    """
    print("\n" + "="*80)
    print(f"Strategy Comparison: {env_name}")
    print(f"Experiences: {num_experiences}, Queries: {num_queries}")
    print("="*80)
    
    strategies = ["memorybank", "generative", "voyager"]
    results = {}
    
    # Generate test data
    print("\nGenerating test data...")
    experiences = generate_test_experiences(env_name, num_experiences)
    
    # Query states (sample from experiences)
    query_states = [exp['st'] for exp in experiences[::max(1, num_experiences//num_queries)]][:num_queries]
    
    # Test each strategy
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy.upper()}")
        print('='*80)
        
        try:
            # Create backend
            backend = create_backend(
                env_name=env_name,
                memory_strategy=strategy,
                storage_path=f"./storage/benchmark_{env_name}_{strategy}_exp_store.json",
                depreiciate_exp_store_path=f"./storage/benchmark_{env_name}_{strategy}_deprecated.json"
            )
            
            # Benchmark storage
            print("\n1. Storage Performance:")
            storage_metrics = benchmark_storage(backend, experiences)
            for key, value in storage_metrics.items():
                if 'time' in key:
                    print(f"  {key}: {value:.4f} seconds")
                else:
                    print(f"  {key}: {value:.2f}")
            
            # Benchmark retrieval
            print("\n2. Retrieval Performance:")
            retrieval_metrics = benchmark_retrieval(backend, query_states)
            for key, value in retrieval_metrics.items():
                if 'time' in key:
                    print(f"  {key}: {value:.4f} seconds")
                else:
                    print(f"  {key}: {value:.2f}")
            
            # Get strategy-specific metrics
            print("\n3. Strategy-Specific Metrics:")
            
            if strategy == "memorybank":
                stats = backend.get_memory_statistics()
                print(f"  Active experiences: {stats['active_experiences']}")
                print(f"  Forgotten experiences: {stats['forgotten_experiences']}")
                print(f"  Average importance: {stats['average_importance']:.3f}")
            
            elif strategy == "voyager":
                stats = backend.get_statistics()
                print(f"  Success rate: {stats['success_rate']:.2%}")
                print(f"  Successful experiences: {stats['successful_experiences']}")
                print(f"  Failed experiences: {stats['failed_experiences']}")
            
            # Store results
            results[strategy] = {
                "storage": storage_metrics,
                "retrieval": retrieval_metrics
            }
            
            # Clean up
            backend.logIO.close()
            
            print(f"\n✓ {strategy.upper()} completed")
            
        except Exception as e:
            print(f"\n✗ Error testing {strategy}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[strategy] = {"error": str(e)}
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\nStorage Performance (experiences/second):")
    for strategy in strategies:
        if strategy in results and "storage" in results[strategy]:
            eps = results[strategy]["storage"]["experiences_per_second"]
            print(f"  {strategy:12s}: {eps:8.2f} exp/s")
    
    print("\nRetrieval Performance (queries/second):")
    for strategy in strategies:
        if strategy in results and "retrieval" in results[strategy]:
            qps = results[strategy]["retrieval"]["queries_per_second"]
            print(f"  {strategy:12s}: {qps:8.2f} queries/s")
    
    print("\nAverage Retrieval Time:")
    for strategy in strategies:
        if strategy in results and "retrieval" in results[strategy]:
            avg_time = results[strategy]["retrieval"]["avg_retrieval_time"]
            print(f"  {strategy:12s}: {avg_time:8.4f} seconds")
    
    # Save results to file
    output_file = f"./storage/benchmark_results_{env_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


def main():
    """Run comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare memory strategies")
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole",
        choices=["CartPole", "MountainCar", "Webshop"],
        help="Environment to test"
    )
    parser.add_argument(
        "--num-experiences",
        type=int,
        default=100,
        help="Number of experiences to store"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of retrieval queries"
    )
    
    args = parser.parse_args()
    
    try:
        results = compare_strategies(
            env_name=args.env,
            num_experiences=args.num_experiences,
            num_queries=args.num_queries
        )
        
        print("\n" + "="*80)
        print("Comparison completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
