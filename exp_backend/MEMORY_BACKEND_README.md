# Memory-Enhanced Backends for Global Verifier

This directory contains memory-enhanced backends that integrate GMemory's memory management strategies into the global_verifier experience backend system.

## Overview

Three memory strategies from GMemory have been integrated:

1. **MemoryBank** - Uses a forgetting mechanism where older experiences decay over time
2. **Generative** - Uses LLM-based scoring to rank experience relevance
3. **Voyager** - Baseline vector similarity-based retrieval

Each strategy is available for all supported environments (CartPole, MountainCar, Webshop).

## Architecture

```
exp_backend/
├── memory_enhanced_backend.py       # Base class for memory-enhanced backends
├── memorybank_backend.py            # MemoryBank strategy implementation
├── generative_backend.py            # Generative strategy implementation
├── voyager_backend.py               # Voyager strategy implementation
├── cartPole_memory_backends.py      # CartPole-specific implementations
├── mountainCar_memory_backends.py   # MountainCar-specific implementations
├── webshop_memory_backends.py       # Webshop-specific implementations
└── backend_factory.py               # Factory for easy backend creation
```

## Quick Start

### Basic Usage

```python
from exp_backend.backend_factory import create_backend

# Create a MemoryBank backend for CartPole
backend = create_backend(
    env_name="CartPole",
    memory_strategy="memorybank",
    llm_model=my_llm_model  # Optional, for summarization
)

# Store an experience (with label for memory)
experience = {
    "id": "exp_001",
    "action_path": [0, 1, 0, 1],
    "st": {"cart_position": 0.0, "cart_velocity": 0.0},
    "action": 1,
    "st1": {"cart_position": 0.1, "cart_velocity": 0.05},
    "label": True  # True = successful, False = failed
}
backend.store_experience(experience)

# Retrieve similar experiences
query_state = {"cart_position": 0.0, "cart_velocity": 0.0}
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,
    failed_topk=1
)
```

## Memory Strategies Comparison

### 1. MemoryBank Strategy

**Best for**: Environments where recent experiences are more valuable.

**Features**:
- Exponential decay of older experiences
- Automatic memory management
- Configurable forgetting threshold

**Example**:
```python
backend = create_backend(
    env_name="CartPole",
    memory_strategy="memorybank",
    forgetting_threshold=0.3  # 0-1, higher = more aggressive forgetting
)

# Check memory statistics
stats = backend.get_memory_statistics()
print(f"Active experiences: {stats['active_experiences']}")
print(f"Forgotten experiences: {stats['forgotten_experiences']}")

# Clean up forgotten experiences
backend.cleanup_forgotten()
```

### 2. Generative Strategy

**Best for**: Environments where context matters and similarity isn't enough.

**Features**:
- LLM-based relevance scoring
- Context-aware experience selection
- Custom scoring prompts

**Example**:
```python
# Requires an LLM model
def my_llm_model(messages, temperature=0.1):
    # Your LLM implementation
    # messages: list of {"role": "system"/"user", "content": "..."}
    # returns: string response
    pass

backend = create_backend(
    env_name="MountainCar",
    memory_strategy="generative",
    llm_model=my_llm_model
)

# Retrieve with LLM scoring
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,
    use_llm_scoring=True,
    score_multiplier=2  # Retrieve 2x candidates for scoring
)

# Analyze scoring distribution
dist = backend.analyze_scoring_distribution()
print(f"Mean score: {dist['mean']:.2f}")
```

### 3. Voyager Strategy (Baseline)

**Best for**: Simple environments or as a baseline for comparison.

**Features**:
- Pure vector similarity search
- Fast and efficient
- Optional LLM summarization

**Example**:
```python
backend = create_backend(
    env_name="Webshop",
    memory_strategy="voyager",
    use_summarization=False  # Set True to use LLM summarization
)

# Simple retrieval (no extra mechanisms)
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=5
)

# Get statistics
stats = backend.get_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Comparison with Original Backend

| Feature | Original Backend | Memory-Enhanced Backend |
|---------|-----------------|-------------------------|
| Storage Format | JSON only | JSON + Vector DB |
| Retrieval Method | Exact state match | Semantic similarity |
| Experience Ranking | None | Strategy-dependent |
| Memory Management | Manual | Automatic (MemoryBank) |
| LLM Integration | None | Optional (Generative) |
| Conflict Resolution | ✓ | ✓ (inherited) |

## Environment-Specific Features

### CartPole

- Action type: Integer (0 or 1)
- State includes: cart position/velocity, pole angle/angular velocity
- Custom prompts for LLM scoring

### MountainCar

- Action type: Integer (0, 1, or 2)
- State includes: car position and velocity
- Focuses on momentum-building strategies

### Webshop

- Action type: String (navigation commands)
- State includes: URL, HTML text
- Optimized for shopping task patterns

## Advanced Usage

### Custom Embedding Function

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

def custom_embedding(texts):
    return model.encode(texts)

backend = create_backend(
    env_name="CartPole",
    memory_strategy="voyager",
    embedding_func=custom_embedding
)
```

### Custom Memory Persistence Directory

```python
backend = create_backend(
    env_name="MountainCar",
    memory_strategy="memorybank",
    memory_persist_dir="./custom_vector_db/mountaincar"
)
```

### Batch Operations

```python
# Store multiple experiences
experiences = [exp1, exp2, exp3, ...]
for exp in experiences:
    backend.store_experience(exp)

# Batch score experiences (Generative only)
query = "some query state"
scored = backend.batch_score_experiences(experiences, query)
for score, exp in scored[:5]:  # Top 5
    print(f"Experience {exp['id']}: score={score:.1f}")
```

## Performance Considerations

### Memory Usage

- **Vector DB**: Stores embeddings in ChromaDB (disk-based by default)
- **JSON Store**: Same as original backend
- **Typical overhead**: ~100-500MB for 10k experiences

### Speed

- **Retrieval**: O(log n) with vector index
- **Storage**: O(1) with some embedding computation
- **LLM Scoring**: O(k) where k = number of candidates (can be slow)

### Recommendations

- **MemoryBank**: Best balance of speed and intelligence
- **Generative**: Use when accuracy > speed
- **Voyager**: Use when speed > accuracy

## Troubleshooting

### Issue: "No module named 'langchain_chroma'"

```bash
pip install langchain-chroma chromadb
```

### Issue: "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### Issue: Vector DB corrupted

```python
# Delete and rebuild
import shutil
shutil.rmtree("./storage/memory_db_CartPole")

# Recreate backend (will auto-sync from JSON)
backend = create_backend("CartPole", "memorybank")
```

### Issue: LLM scoring is slow

```python
# Use smaller candidate pool
backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,
    score_multiplier=1  # Only retrieve exactly what's needed
)

# Or disable LLM scoring
backend.retrieve_similar_experiences(
    query_state,
    use_llm_scoring=False  # Fall back to similarity only
)
```

## Testing

### Run Unit Tests

```bash
cd global_verifier
python -m pytest exp_backend/test_memory_backends.py
```

### Benchmark Performance

```python
from exp_backend.benchmark import benchmark_all_strategies

results = benchmark_all_strategies(
    env_name="CartPole",
    num_experiences=1000,
    num_queries=100
)

print(results)
```

## Migration Guide

### From Original Backend

```python
# Old code
from exp_backend.cartPole_exp_backend import CartPoleExpBackend
backend = CartPoleExpBackend("CartPole", "./storage/exp_store.json", "./storage/deprecated.json")

# New code (equivalent)
from exp_backend.backend_factory import create_backend
backend = create_backend("CartPole", "none")  # "none" = original behavior

# Or with memory enhancement
backend = create_backend("CartPole", "memorybank")
```

### API Compatibility

All memory-enhanced backends maintain compatibility with the original `BaseExpBackend` API:

- `store_experience(exp)` - ✓ Compatible
- `get_exp_by_id(exp_id)` - ✓ Compatible
- `_deprecate_experience(exp_id)` - ✓ Compatible
- `resolve_experience_conflict(...)` - ✓ Compatible

Additional methods:
- `retrieve_similar_experiences(query_state, ...)` - New
- Strategy-specific methods (see individual strategy docs)

## References

- **GMemory Paper**: [Link to paper if available]
- **MemoryBank**: `GMemory/mas/memory/mas_memory/memorybank.py`
- **Generative**: `GMemory/mas/memory/mas_memory/generative.py`
- **Voyager**: `GMemory/mas/memory/mas_memory/voyager.py`

## Contributing

To add a new environment:

1. Create `{env_name}_memory_backends.py`
2. Implement three classes inheriting from each strategy
3. Define `expected_fields` for your environment
4. Add to `backend_factory.py`
5. Update this README

## License

Same as global_verifier project.
