# Inter-Task KV Reuse Implementation (Sim-LLM)

This document describes the implementation of Inter-Task KV Reuse based on the Sim-LLM paper.

## Overview

The Inter-Task KV Reuse optimization allows reusing the Key-Value (KV) cache from the **last transformer layer** of similar previous requests, significantly reducing inference latency for similar queries.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SimLLMInference                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  LlamaForCausalLMWithKVReuse                ││
│  │  ┌───────────────────────────────────────────────────────┐  ││
│  │  │              LlamaModelWithKVReuse                     │  ││
│  │  │  ┌─────────────────────────────────────────────────┐  │  ││
│  │  │  │           InterTaskKVManager                     │  │  ││
│  │  │  │  ┌───────────────┐  ┌─────────────────────────┐ │  │  ││
│  │  │  │  │   LSHIndex    │  │   TaskCacheEntry        │ │  │  ││
│  │  │  │  │ (Hash Bucket) │  │ (embedding, KV, hash)   │ │  │  ││
│  │  │  │  └───────────────┘  └─────────────────────────┘ │  │  ││
│  │  │  └─────────────────────────────────────────────────┘  │  ││
│  │  └───────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. LSHIndex (`models/inter_task_kv_manager.py`)

Implements Locality-Sensitive Hashing using Random Projection for fast similarity search.

```python
class LSHIndex:
    def __init__(self, embedding_dim: int, num_hyperplanes: int = 16, seed: int = 42):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_hyperplanes: Number of random hyperplanes for hashing
            seed: Random seed for reproducibility
        """
```

**Key Methods:**
- `compute_hash(embedding)`: Compute LSH hash for a single embedding
- `compute_hash_batch(embeddings)`: Compute hashes for multiple embeddings

### 2. TaskCacheEntry (`models/inter_task_kv_manager.py`)

Represents a cached task with its embedding and KV pairs.

```python
class TaskCacheEntry:
    def __init__(
        self,
        task_id: str,           # Unique identifier
        task_embedding: Tensor, # Mean-pooled embedding
        lsh_hash: str,          # LSH hash for bucketing
        top_layer_kv: Tuple,    # (key, value) from last layer
        timestamp: int          # For LRU eviction
    ):
```

### 3. InterTaskKVManager (`models/inter_task_kv_manager.py`)

Global KV Manager implementing LSH-based similarity search and LRU eviction.

```python
class InterTaskKVManager:
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.8,
        num_hyperplanes: int = 16,
        device: str = 'cuda'
    ):
```

**Key Methods:**
- `search_similar_task(query_embedding)`: Search for similar cached task
- `add_task(task_id, task_embedding, top_layer_kv)`: Add new task to cache
- `get_statistics()`: Get cache hit/miss statistics
- `reset()`: Clear the cache

### 4. LlamaModelWithKVReuse (`models/modeling_llama_inter_task_kv.py`)

Modified LlamaModel that supports KV cache reuse from similar tasks.

**Key Features:**
- Intercepts forward pass to check for KV reuse opportunity
- Skips layers 0 to L-2 when cache hit occurs
- Saves KV from last layer on cache miss

### 5. LlamaForCausalLMWithKVReuse (`models/modeling_llama_inter_task_kv.py`)

Causal LM wrapper with KV reuse support.

**Key Methods:**
- `set_kv_manager(kv_manager)`: Connect KV manager
- `generate_with_kv_reuse(input_ids, task_id, **kwargs)`: Generate with KV reuse

### 6. SimLLMInference (`models/sim_llm_inference.py`)

High-level inference wrapper providing a clean API.

```python
class SimLLMInference:
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.8,
        num_hyperplanes: int = 16,
        device: str = 'cuda',
        **model_kwargs
    ) -> 'SimLLMInference':
```

**Key Methods:**
- `generate(prompt, max_new_tokens, enable_kv_reuse)`: Generate response
- `batch_generate(prompts, max_new_tokens)`: Batch generation
- `get_statistics()`: Get inference statistics
- `reset_cache()`: Clear KV cache

## Inference Flow

### Cache Miss (Standard Flow + Save)

```
Input → Embed → [Layer 0] → [Layer 1] → ... → [Layer L-1] → Output
                                                    ↓
                                              Save KV to Cache
```

### Cache Hit (Reuse Flow)

```
Input → Embed → [Skip Layer 0-L-2] → [Layer L-1 with Cached KV] → Output
                                            ↑
                                    Retrieve KV from Cache
```

## Usage Examples

### Basic Usage

```python
from models import SimLLMInference

# Create inference wrapper
inference = SimLLMInference.from_pretrained(
    model_path="meta-llama/Llama-2-7b-hf",
    max_cache_size=100,
    similarity_threshold=0.8
)

# Generate responses
response1 = inference.generate("What is machine learning?")
response2 = inference.generate("Explain machine learning.")  # May reuse KV

# Check statistics
print(inference.get_statistics())
```

### Advanced Usage

```python
from models import (
    InterTaskKVManager,
    LlamaForCausalLMWithKVReuse
)
from transformers import AutoTokenizer

# Load model and tokenizer
model = LlamaForCausalLMWithKVReuse.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
).cuda()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create KV manager
kv_manager = InterTaskKVManager(
    embedding_dim=model.config.hidden_size,
    max_cache_size=100,
    similarity_threshold=0.8
)

# Connect KV manager to model
model.set_kv_manager(kv_manager)

# Generate with KV reuse
inputs = tokenizer("What is AI?", return_tensors="pt").to("cuda")
outputs = model.generate_with_kv_reuse(
    inputs["input_ids"],
    task_id="query_001",
    max_new_tokens=50
)
```

### Batch Processing

```python
prompts = [
    "What is machine learning?",
    "Explain deep learning.",
    "What is neural network?",
]

responses = inference.batch_generate(
    prompts,
    max_new_tokens=100,
    enable_kv_reuse=True
)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 4096 | Dimension of task embeddings |
| `max_cache_size` | 100 | Maximum number of cached tasks |
| `similarity_threshold` | 0.8 | Cosine similarity threshold for cache hit |
| `num_hyperplanes` | 16 | Number of hyperplanes for LSH |
| `device` | 'cuda' | Device for tensor storage |

## Performance Considerations

### Memory Usage

- Each cached task stores:
  - Task embedding: `(embedding_dim,)` float32
  - LSH hash: string
  - KV pairs: `(batch, num_heads, seq_len, head_dim)` × 2

### Tuning Tips

1. **Similarity Threshold**
   - Higher (0.9): More strict matching, fewer cache hits, higher accuracy
   - Lower (0.7): More lenient matching, more cache hits, potential quality loss

2. **Cache Size**
   - Larger cache: More potential hits, higher memory usage
   - Smaller cache: Less memory, more evictions

3. **Number of Hyperplanes**
   - More hyperplanes: Finer-grained buckets, fewer false positives
   - Fewer hyperplanes: Coarser buckets, more candidates to verify

## API Reference

### SimLLMInference

```python
class SimLLMInference:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs) -> 'SimLLMInference'
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        task_id: Optional[str] = None,
        enable_kv_reuse: bool = True,
        **generate_kwargs
    ) -> str
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        enable_kv_reuse: bool = True,
        **generate_kwargs
    ) -> List[str]
    
    def get_statistics(self) -> Dict
    def reset_cache(self) -> None
    def set_similarity_threshold(self, threshold: float) -> None
```

### InterTaskKVManager

```python
class InterTaskKVManager:
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.8,
        num_hyperplanes: int = 16,
        device: str = 'cuda'
    )
    
    def search_similar_task(
        self,
        query_embedding: torch.Tensor
    ) -> Optional[TaskCacheEntry]
    
    def add_task(
        self,
        task_id: str,
        task_embedding: torch.Tensor,
        top_layer_kv: Tuple[torch.Tensor, torch.Tensor]
    ) -> None
    
    def get_statistics(self) -> Dict
    def reset(self) -> None
    def generate_task_id(self, input_text: str) -> str
```

## File Structure

```
models/
├── __init__.py                      # Module exports
├── inter_task_kv_manager.py         # KV Manager with LSH
├── modeling_llama_inter_task_kv.py  # Modified Llama model
└── sim_llm_inference.py             # High-level inference wrapper

examples/
└── sim_llm_demo.py                  # Demo script
```

## Running the Demo

```bash
# Basic demo
python examples/sim_llm_demo.py --demo_mode basic

# Similarity threshold demo
python examples/sim_llm_demo.py --demo_mode similarity

# Performance benchmark
python examples/sim_llm_demo.py --demo_mode benchmark

# Custom configuration
python examples/sim_llm_demo.py \
    --model_path /path/to/model \
    --max_cache_size 200 \
    --similarity_threshold 0.85 \
    --demo_mode basic
```

## Limitations

1. **Prefill Phase Only**: Currently optimizes only the prefill phase, not autoregressive decoding
2. **Single Batch**: Designed for batch_size=1 scenarios
3. **Last Layer Only**: Reuses KV from only the last transformer layer
4. **Embedding-Based Similarity**: Uses mean-pooled embeddings which may not capture all semantic nuances

## Future Improvements

1. Multi-layer KV reuse
2. Batch processing optimization
3. More sophisticated similarity metrics
4. Adaptive threshold tuning
5. Distributed cache support

## References

- Sim-LLM Paper: Inter-Task KV Reuse for Efficient LLM Inference
- HuggingFace Transformers: https://github.com/huggingface/transformers
- Locality-Sensitive Hashing: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
