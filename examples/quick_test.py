"""
Quick Test Script

A simple script to verify that the memory-enhanced backends are working correctly.
Run this to make sure everything is installed and configured properly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from exp_backend import (
            create_backend,
            MemoryBankBackend,
            GenerativeBackend,
            VoyagerBackend
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_backend_creation():
    """Test creating backends for each strategy."""
    print("\nTesting backend creation...")
    
    from exp_backend import create_backend
    
    strategies = ["none", "memorybank", "generative", "voyager"]
    
    for strategy in strategies:
        try:
            backend = create_backend(
                env_name="CartPole",
                memory_strategy=strategy,
                storage_path=f"./storage/test_{strategy}_exp_store.json",
                depreiciate_exp_store_path=f"./storage/test_{strategy}_deprecated.json"
            )
            print(f"✓ Created {strategy} backend: {backend.__class__.__name__}")
            backend.logIO.close()
        except Exception as e:
            print(f"✗ Failed to create {strategy} backend: {e}")
            return False
    
    return True


def test_basic_operations():
    """Test basic store and retrieve operations."""
    print("\nTesting basic operations...")
    
    from exp_backend import create_backend
    
    # Create a simple backend
    backend = create_backend(
        env_name="CartPole",
        memory_strategy="voyager",
        storage_path="./storage/test_operations_exp_store.json",
        depreiciate_exp_store_path="./storage/test_operations_deprecated.json"
    )
    
    try:
        # Store some experiences
        for i in range(5):
            exp = {
                "id": f"test_exp_{i}",
                "action_path": [0, 1] * i,
                "st": {"position": 0.0 + i*0.1, "velocity": 0.0},
                "action": i % 2,
                "st1": {"position": 0.1 + i*0.1, "velocity": 0.01},
                "label": True if i % 2 == 0 else False
            }
            backend.store_experience(exp)
        
        print("✓ Stored 5 experiences")
        
        # Retrieve experiences
        query_state = {"position": 0.2, "velocity": 0.0}
        successful, failed = backend.retrieve_similar_experiences(
            query_state,
            successful_topk=2,
            failed_topk=1
        )
        
        print(f"✓ Retrieved {len(successful)} successful and {len(failed)} failed experiences")
        
        # Get an experience by ID
        exp = backend.get_exp_by_id("test_exp_0")
        print(f"✓ Retrieved experience by ID: {exp['id']}")
        
        backend.logIO.close()
        return True
        
    except Exception as e:
        print(f"✗ Operation failed: {e}")
        import traceback
        traceback.print_exc()
        backend.logIO.close()
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")
    
    required = [
        ('langchain_chroma', 'Chroma'),
        ('langchain.docstore.document', 'Document'),
        ('chromadb', None)
    ]
    
    all_ok = True
    for module_name, class_name in required:
        try:
            if class_name:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"✓ {module_name}.{class_name}")
            else:
                __import__(module_name)
                print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} not found - install with: pip install {module_name.split('.')[0]}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("="*80)
    print("Quick Test for Memory-Enhanced Backends")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("Backend Creation", test_backend_creation()))
    results.append(("Basic Operations", test_basic_operations()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All tests passed! The memory-enhanced backends are ready to use.")
        print("\nNext steps:")
        print("  1. Run full demo: python examples/memory_backend_demo.py")
        print("  2. Compare strategies: python examples/compare_strategies.py")
        print("  3. Read the documentation: MEMORY_INTEGRATION_CN.md")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  1. Install missing dependencies: pip install langchain-chroma chromadb")
        print("  2. Make sure you're in the correct directory")
        print("  3. Check that storage/ directory exists and is writable")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
