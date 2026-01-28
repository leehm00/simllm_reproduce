**Role:** AI System Engineer
**Task:** Implement the "Inter-Task KV Reuse" logic from the Sim-LLM paper using HuggingFace `transformers.models.llama`.
**Context:** We want to optimize inference by reusing the Key-Value (KV) cache of the **last transformer layer** from satisfyingly similar previous requests.

**Core Logic Requirements:**

You need to modify or wrap the `LlamaForCausalLM` (and `LlamaModel`) inference flow to support the following three stages:

#### 1. Global KV Manager (The Storage)

Implement a class `KVManager` that persists data from processed requests.

* **Data Structure:** It must store:
* `task_embedding`: The mean-pooled embedding of the prompt tokens.
* `lsh_hash`: A locality-sensitive hash of the embedding (use Random Projection or SimHash).
* `top_layer_kv`: The Key and Value tensors **only from the last layer** (Layer ) of the model.


* **Eviction:** Implement a simple LRU (Least Recently Used) policy when cache is full.

#### 2. Similarity Search (The Retrieval)

Before the model's `forward()` pass, intercept the input `input_ids`:

1. **Compute Embedding:** Get the embedding of the current input .
2. 
**LSH Bucketing:** Compute hash  and retrieve candidate tasks from `KVManager` sharing the same bucket .


3. **Verification:** Calculate **Cosine Similarity** between  and candidate embeddings.
4. 
**Match Condition:** If similarity > **0.8** (strict threshold), declare a "Cache Hit".



#### 3. Inference Execution Flow (The Reuse Logic)

Modify the inference logic based on the Match Condition:

* 
**Scenario A: Cache Hit (Reuse Logic)** 


* **Retrieve:** Get the `top_layer_kv` from the matched old task.
* **Bypass:** Do **NOT** compute the full forward pass for layers  to .
* **Inject:** Construct a `past_key_values` object where:
* Layers  to  are `None` (or empty/dummy tensors).
* Layer  (Last Layer) contains the retrieved `top_layer_kv`.


* **Execute:** Run the model's `forward` pass but ensure it effectively only processes the last layer using the injected KV, generating the next token directly.


* **Scenario B: Cache Miss (Standard Logic & Save)**
* **Execute:** Run standard `LlamaModel` forward pass.
* **Save:** After the prefill phase, extract the KV pairs from the **last layer** and the prompt's embedding.
* **Update:** Push `(embedding, hash, top_layer_kv)` into `KVManager`.



**Implementation Constraints:**

* Base your code on `transformers.models.llama.modeling_llama`.
* Focus on the **Prefill** phase optimization.
* Ensure tensor shapes for `past_key_values` match `(batch, num_heads, seq_len, head_dim)`.
* Keep the LSH implementation simple (e.g., using `numpy` for random projection).

