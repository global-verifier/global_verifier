# Examples for Memory-Enhanced Backends

This directory contains example scripts demonstrating how to use the memory-enhanced backends.

## Quick Start

### 1. Verify Installation

First, make sure everything is installed correctly:

```bash
python quick_test.py
```

This will check:
- ✓ Required dependencies are installed
- ✓ Modules can be imported
- ✓ Backends can be created
- ✓ Basic operations work

### 2. Run the Demo

See all three strategies in action:

```bash
python memory_backend_demo.py
```

This demonstrates:
- MemoryBank strategy with forgetting mechanism
- Generative strategy with LLM scoring
- Voyager strategy with baseline retrieval
- Side-by-side comparison of all three

### 3. Compare Performance

Benchmark the strategies:

```bash
# Default: CartPole, 100 experiences, 20 queries
python compare_strategies.py

# Custom parameters
python compare_strategies.py --env MountainCar --num-experiences 500 --num-queries 50
```

## Example Scripts

### quick_test.py

**Purpose**: Verify installation and basic functionality

**Usage**:
```bash
python quick_test.py
```

**What it tests**:
- Dependencies (langchain, chromadb, etc.)
- Module imports
- Backend creation for all strategies
- Basic store and retrieve operations

**Expected output**:
```
✓ Dependencies: PASS
✓ Imports: PASS
✓ Backend Creation: PASS
✓ Basic Operations: PASS
```

### memory_backend_demo.py

**Purpose**: Demonstrate each strategy with detailed examples

**Usage**:
```bash
python memory_backend_demo.py
```

**What it shows**:

1. **MemoryBank Demo**
   - Storing experiences with timestamps
   - Memory statistics (active/forgotten)
   - Retrieval with forgetting mechanism

2. **Generative Demo**
   - LLM-based scoring
   - Comparison with/without LLM
   - Re-ranking of candidates

3. **Voyager Demo**
   - Simple similarity-based retrieval
   - Statistics tracking
   - Baseline performance

4. **Comparison Demo**
   - Same experiences in all backends
   - Same query to all backends
   - Compare retrieval results

### compare_strategies.py

**Purpose**: Benchmark and compare strategy performance

**Usage**:
```bash
python compare_strategies.py [options]

Options:
  --env ENV                 Environment (CartPole, MountainCar, Webshop)
  --num-experiences N       Number of experiences to store
  --num-queries N          Number of retrieval queries
```

**Examples**:
```bash
# Quick test
python compare_strategies.py --num-experiences 50 --num-queries 10

# Full benchmark
python compare_strategies.py --num-experiences 1000 --num-queries 100

# Test MountainCar
python compare_strategies.py --env MountainCar --num-experiences 200
```

**Output**:
- Storage performance (experiences/second)
- Retrieval performance (queries/second)
- Average retrieval time
- Strategy-specific metrics
- Results saved to `storage/benchmark_results_*.json`

## Creating Your Own Examples

### Basic Template

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp_backend import create_backend

# Create backend
backend = create_backend(
    env_name="CartPole",
    memory_strategy="memorybank"
)

# Store experiences
for i in range(10):
    exp = {
        "id": f"exp_{i}",
        "action_path": [0, 1],
        "st": {"position": 0.0},
        "action": 0,
        "st1": {"position": 0.1},
        "label": True
    }
    backend.store_experience(exp)

# Retrieve experiences
query_state = {"position": 0.0}
successful, failed = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3
)

print(f"Found {len(successful)} similar experiences")

# Cleanup
backend.logIO.close()
```

### Using with LLM

```python
from exp_backend import create_backend

# Define your LLM interface
def my_llm(messages, temperature=0.1):
    # Call your LLM here
    # messages: [{"role": "system"/"user", "content": "..."}]
    # return: string response
    pass

# Create backend with LLM
backend = create_backend(
    env_name="MountainCar",
    memory_strategy="generative",
    llm_model=my_llm
)

# Use as normal
# LLM will be called for scoring during retrieval
```

### Comparing Multiple Environments

```python
from exp_backend import create_backend

environments = ["CartPole", "MountainCar", "Webshop"]

for env in environments:
    backend = create_backend(env, "voyager")
    
    # Your evaluation code here
    
    stats = backend.get_statistics()
    print(f"{env}: {stats['total_experiences']} experiences")
    
    backend.logIO.close()
```

## Troubleshooting

### ImportError: No module named 'langchain_chroma'

```bash
pip install langchain-chroma chromadb
```

### ImportError: No module named 'sentence_transformers'

```bash
pip install sentence-transformers
```

### PermissionError: Cannot write to storage/

```bash
mkdir -p storage
chmod 755 storage
```

### ChromaDB errors

Delete the vector database and let it rebuild:

```bash
rm -rf storage/memory_db_*
python quick_test.py  # Will rebuild from JSON
```

### Slow LLM scoring

Use a smaller candidate pool:

```python
backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,
    score_multiplier=1  # Instead of default 2
)
```

Or disable LLM scoring:

```python
backend.retrieve_similar_experiences(
    query_state,
    use_llm_scoring=False
)
```

## Additional Resources

- **Detailed Documentation**: `../exp_backend/MEMORY_BACKEND_README.md`
- **Chinese Guide**: `../MEMORY_INTEGRATION_CN.md`
- **Integration Summary**: `../INTEGRATION_SUMMARY.md`

## Contributing

To add new examples:

1. Create a new `.py` file in this directory
2. Follow the template structure above
3. Add documentation comments
4. Update this README with a description

## License

Same as the global_verifier project.
