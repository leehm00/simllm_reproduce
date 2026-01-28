✅ 成功加载Inter-Task KV Reuse模型代码！
Using device: cuda

================================================================================
INTER-TASK KV REUSE TEST
================================================================================
Model: /mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct
Similarity Threshold: 0.7
Max Cache Size: 100
Num Hyperplanes: 16
================================================================================


============================================================
Loading model...
Model path: /mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct
Similarity threshold: 0.7
Max cache size: 100
Num hyperplanes: 16
============================================================

Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:01,  2.22it/s]Loading checkpoint shards:  50%|█████     | 2/4 [00:00<00:00,  2.27it/s]Loading checkpoint shards:  75%|███████▌  | 3/4 [00:01<00:00,  2.30it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  3.09it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  2.73it/s]
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=6
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=5
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=9
[LlamaModel] KV Reuse enabled
✅ Model loaded successfully!
Hidden size: 4096
Num layers: 32
Pad token: <|eot_id|> (id=128009)

================================================================================
RUNNING DIAGNOSTIC TEST FIRST
================================================================================

================================================================================
DIAGNOSTIC RUN
================================================================================
Initial cache size: 0

--- Diagnostic prompt 1/3: '什么是人工智能' ---

======================================================================
Running inference for task: diag_0
Input length: 6 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=diag_0, query_hash=110001101101..., query_norm=0.2505
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = diag_0
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=6, task_id=diag_0, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: （ B��������
⏱️ Latency: 1.077s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 1
Cache hits: 0
Cache misses: 1
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


--- Diagnostic prompt 2/3: '人工智能是什么' ---

======================================================================
Running inference for task: diag_1
Input length: 5 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=diag_1, query_hash=100011101100..., query_norm=0.2725
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = diag_1
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=5, task_id=diag_1, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: ？ and the States States States States States States States
⏱️ Latency: 0.365s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 2
Cache hits: 0
Cache misses: 2
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


--- Diagnostic prompt 3/3: '解释人工智能的含义' ---

======================================================================
Running inference for task: diag_2
Input length: 9 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=diag_2, query_hash=100001101000..., query_norm=0.2247
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = diag_2
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=9, task_id=diag_2, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=6
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=5
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: 
        in the States States States States States States
⏱️ Latency: 0.365s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 3
Cache hits: 0
Cache misses: 3
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


Final cache size: 0

Diagnostic Summary:
  Prompts processed: 3
  Cache hits: 0
  Cache misses: 3
  Successful saves: 0
  Failed saves: 3
  Cache size increase: 0

❌ CRITICAL ASSERTION FAILED: All 3 cache misses failed to save!
This indicates a fundamental problem with the auto-save logic.
Debug info:
  diag_0: cache_hit=False, add_success=False
  diag_1: cache_hit=False, add_success=False
  diag_2: cache_hit=False, add_success=False

⚠️ Diagnostic test failed! Check the logs above for details.
[KVManager] Cache reset

================================================================================
CACHE FUNCTIONALITY TEST
Testing with short similar sentences
================================================================================

============================================================
Testing Group 1: AI questions (Chinese) - Primary Test
============================================================

======================================================================
Running inference for task: group0_prompt0
Input length: 6 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=group0_prompt0, query_hash=110001101101..., query_norm=0.2505
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = group0_prompt0
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=6, task_id=group0_prompt0, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: （ B��������
⏱️ Latency: 0.365s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 1
Cache hits: 0
Cache misses: 1
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


======================================================================
Running inference for task: group0_prompt1
Input length: 5 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=group0_prompt1, query_hash=100011101100..., query_norm=0.2725
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = group0_prompt1
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=5, task_id=group0_prompt1, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: ？ and the States States States States States States States
⏱️ Latency: 0.365s
📊 Cache Hit: False
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=9
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=6
[LlamaModel] ⚠️ Auto-save failed: Could not extract valid last_layer_kv. next_cache type=tuple, seq_len=6

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 2
Cache hits: 0
Cache misses: 2
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


======================================================================
Running inference for task: group0_prompt2
Input length: 9 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=group0_prompt2, query_hash=100001101000..., query_norm=0.2247
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = group0_prompt2
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=9, task_id=group0_prompt2, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer: 
        in the States States States States States States
⏱️ Latency: 0.364s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 3
Cache hits: 0
Cache misses: 3
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


============================================================
Testing Group 2: ML questions (English)
============================================================

======================================================================
Running inference for task: group1_prompt0
Input length: 6 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=group1_prompt0, query_hash=111110110101..., query_norm=0.2085
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = group1_prompt0
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=6, task_id=group1_prompt0, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer:  Machine in the States States States States States States States
⏱️ Latency: 0.366s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 4
Cache hits: 0
Cache misses: 4
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


======================================================================
Running inference for task: group1_prompt1
Input length: 6 tokens
======================================================================

======================================================================
[KVManager] REQUEST: task_id=group1_prompt1, query_hash=011100110001..., query_norm=0.2123
[KVManager] Cache size: 0, Buckets: 0
[KVManager] Bucket candidates: [], Total cached entries: 0
[KVManager] ❌ Cache MISS - No candidates available
======================================================================

📝 Cache MISS, running full inference (KV will be auto-saved)...
[Cache MISS] Set model.model.current_task_id = group1_prompt1
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Auto-save check: seq_len=6, task_id=group1_prompt1, use_cache=True
[LlamaModel] next_cache type: tuple
[LlamaModel] next_cache length: 32
[LlamaModel] next_cache[0] type: NoneType
[LlamaModel] _extract_last_layer_kv: tuple detected, len=32
[LlamaModel] _extract_last_layer_kv: next_cache[0] is None
[LlamaModel] _extract_last_layer_kv: next_cache[-1] is None
[LlamaModel] _extract_last_layer_kv: FAILED to extract any valid KV
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 0 present_kv is None (layer_outputs len=2)
[LlamaModel] Layer 31 present_kv is None (layer_outputs len=2)
[Auto-Save] ⚠️ KV was not auto-saved (may already exist or save failed)

🤖 Generated Answer:  What more:// Angeles States States States States States States
⏱️ Latency: 0.367s
📊 Cache Hit: False

======================================================================
[KVManager] CACHE STATE DUMP
======================================================================
Cache size: 0
Number of buckets: 0
Total queries: 5
Cache hits: 0
Cache misses: 5
Hit rate: 0.00%

Buckets:

Top 5 entries:
======================================================================

======================================================================


================================================================================
TEST RESULTS SUMMARY
================================================================================

Prompt                         Cache Hit    Saved    Latency   
------------------------------------------------------------
什么是人工智能                        ❌ MISS       ❌        0.365s
人工智能是什么                        ❌ MISS       ❌        0.365s
解释人工智能的含义                      ❌ MISS       ❌        0.364s
What is machine learning?      ❌ MISS       ❌        0.366s
Explain machine learning.      ❌ MISS       ❌        0.367s

------------------------------------------------------------
Total Queries: 5
Cache Hits: 0
Cache Misses: 5
Hit Rate: 0.00%

KV Manager Statistics:
  Cache Size: 0
  Num Buckets: 0
  Total Queries: 5
  Cache Hits: 0
  Cache Misses: 5
  Hit Rate: 0.00%

================================================================================
TEST COMPLETE
================================================================================
模型已释放，显存已清理
/mnt/sda1/homie_cache does not exist.
