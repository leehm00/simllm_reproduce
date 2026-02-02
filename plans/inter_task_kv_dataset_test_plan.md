# Inter-Task KV Dataset Test Script Plan

## Overview

Create a new test script `inter_task_kv_dataset_test.py` that tests the Inter-Task KV Reuse functionality using the same dataset format and testing methodology as `llama3_args.py`.

## Key Differences from Current `inter_task_kv_test.py`

| Aspect | Current `inter_task_kv_test.py` | New Script |
|--------|--------------------------------|------------|
| Data Source | Hardcoded test prompts | JSON dataset file |
| Test Types | Cache functionality test | TTFT + F1 + Cache tests |
| Prompt Format | Simple prompts | QA format with context/question/answer |
| Metrics | Cache hit rate, latency | TTFT, F1 score, cache statistics |

## Script Structure

```
inter_task_kv_dataset_test.py
├── parse_args()                    # Command-line argument parsing
├── garbage_collection()            # Memory cleanup
├── start_model()                   # Load model with KV manager
├── stop_model()                    # Release model resources
├── clean_cache()                   # Clean cache directory
├── run_inference_ttft()            # TTFT latency test with KV reuse
├── run_inference_f1()              # F1 score test with KV reuse
├── main()                          # Main entry point
└── Dataset iteration loop          # Process samples from JSON
```

## Command-Line Arguments

Based on `llama3_args.py` pattern, with Inter-Task KV specific parameters:

```python
parser.add_argument("--ttft_test", type=bool, default=False)
parser.add_argument("--f1_test", type=bool, default=True)
parser.add_argument("--similarity_threshold", type=float, default=0.7)
parser.add_argument("--max_cache_size", type=int, default=100)
parser.add_argument("--num_hyperplanes", type=int, default=16)
parser.add_argument("--fromdataset", type=str, default='...')
parser.add_argument("--max_count", type=int, default=100)
parser.add_argument("--model_path", type=str, default='...')
```

## Dataset Format

Expected JSON format - same as used in `llama3_args.py`:

```json
[
  {
    "context": "passage text...",
    "question": "question text",
    "answer": "expected answer"
  },
  ...
]
```

## Prompt Template

Following `llama3_args.py` pattern:

```python
prefix_prompt = "You will be asked a question after reading several passages..."
query_prompt = "\n\nAnswer the question directly based on the given passages..."

input_prompt = prefix_prompt + doc_prompt + query_prompt + question_prompt

messages = [
    {"role": "system", "content": "You are an assistant..."},
    {"role": "user", "content": input_prompt},
]

input_ids = tokenizer.apply_chat_template(messages, ...)
```

## Test Flow

### TTFT Test Flow

```
For each sample in dataset:
    1. Build prompt from context + question
    2. Apply chat template
    3. Run inference with KV reuse enabled
       - Check for cache hit
       - If hit: use cached KV, measure TTFT
       - If miss: full prefill, save KV, measure TTFT
    4. Record TTFT latency
    5. Garbage collection
    6. Print running statistics
```

### F1 Test Flow

```
For each sample in dataset:
    1. Build prompt from context + question
    2. Apply chat template
    3. Run inference with KV reuse enabled
       - Generate max_new_tokens=10
    4. Decode response
    5. Compute F1 score against ground truth
    6. Record latency and F1
    7. Garbage collection
    8. Print running statistics
```

## Key Implementation Details

### 1. Model Loading

```python
def start_model(model_path, similarity_threshold, max_cache_size, num_hyperplanes):
    model = LlamaForCausalLMWithKVReuse.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    kv_manager = InterTaskKVManager(
        embedding_dim=model.config.hidden_size,
        max_cache_size=max_cache_size,
        similarity_threshold=similarity_threshold,
        num_hyperplanes=num_hyperplanes,
        device=str(device)
    )
    
    model.set_kv_manager(kv_manager)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, kv_manager
```

### 2. Inference with KV Reuse

```python
def run_inference_with_kv_reuse(input_ids, attention_mask, model, tokenizer, task_id, kv_manager):
    # Compute embedding for similarity search
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        task_embedding = kv_manager._compute_task_embedding(inputs_embeds)
    
    # Search for similar task
    matched_entry = kv_manager.search_similar_task(task_embedding, task_id=task_id)
    
    if matched_entry is not None:
        # Cache HIT: Use layer skipping
        # ... layer skipping logic from inter_task_kv_test.py
    else:
        # Cache MISS: Full inference with auto-save
        model.model.current_task_id = task_id
        outputs = model.generate(...)
        model.model.current_task_id = None
    
    return latency, response, cache_hit
```

### 3. Statistics Tracking

Track and report:
- TTFT latencies - separate for cache hit vs miss
- F1 scores - separate for cache hit vs miss
- Cache hit rate
- Average speedup from cache hits

## Output Format

Following `llama3_args.py` output style:

```
TTFTs = [0.123, 0.045, 0.134, ...]
F1s = [0.85, 0.92, 0.78, ...]
TTFT_with_cache_mean = 0.045
TTFT_with_full_prefill_mean = 0.128
Average_f1_full = 0.82
Average_f1_blend = 0.80
Cache_hit_rate = 0.35
```

## File Location

New file: `inter_task_kv_dataset_test.py` in project root

## Dependencies

- `models.modeling_llama_inter_task_kv.LlamaForCausalLMWithKVReuse`
- `models.inter_task_kv_manager.InterTaskKVManager`
- `utils.load_datasets`
- `utils.compute_f1`
- `transformers.AutoTokenizer`
- `transformers.cache_utils.DynamicCache` - optional for transformers 4.36+

## Implementation Checklist

- [ ] Create file with imports and environment setup
- [ ] Implement `parse_args()` with all parameters
- [ ] Implement `start_model()` with KV manager setup
- [ ] Implement `run_inference_ttft()` with KV reuse
- [ ] Implement `run_inference_f1()` with KV reuse
- [ ] Implement main loop for TTFT test
- [ ] Implement main loop for F1 test
- [ ] Add statistics tracking and reporting
- [ ] Add cache statistics output
- [ ] Test with sample dataset
