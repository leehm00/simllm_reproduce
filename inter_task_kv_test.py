#!/usr/bin/env python3
"""
Inter-Task KV Reuse Test Script
Based on llama_lsh_args.py pattern
Tests the KV cache reuse functionality with detailed logging
"""

import argparse
import gc
import torch
import numpy as np
import time
import sys
import os
from transformers import AutoTokenizer
from pathlib import Path

# Try to import DynamicCache for transformers 4.36+
try:
    from transformers.cache_utils import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None

import shutil

# 在main函数前定义解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser(description="Inter-Task KV Reuse Test Script")
    parser.add_argument("--ttft_test", type=bool, default=False, help="Enable TTFT latency test")
    parser.add_argument("--f1_test", type=bool, default=False, help="Enable F1 score test")
    parser.add_argument("--cache_test", type=bool, default=True, help="Enable cache hit/miss test")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, 
                       help="Cosine similarity threshold for cache hit (0-1)")
    parser.add_argument("--max_cache_size", type=int, default=100, 
                       help="Maximum number of cached tasks")
    parser.add_argument("--num_hyperplanes", type=int, default=16, 
                       help="Number of hyperplanes for LSH")
    parser.add_argument("--fromdataset", type=str, 
                       default='/home/homie/homie/fuzzy_llama_submit/datasets/wiki_for_test.json',
                       help="Path to dataset JSON file")
    parser.add_argument("--max_count", type=int, default=10, 
                       help="Maximum number of samples to process")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct",
                       help="Path to pretrained model")
    return parser.parse_args()

try:
    # 导入Inter-Task KV Reuse模型实现
    from models.modeling_llama_inter_task_kv import LlamaForCausalLMWithKVReuse
    from models.inter_task_kv_manager import InterTaskKVManager
    print("yes 成功加载Inter-Task KV Reuse模型代码！")
except ImportError as e:
    print(f"no 导入错误：{e}")
    print("请确认：")
    print("1. 当前目录存在 models/modeling_llama_inter_task_kv.py")
    print("2. 当前目录存在 models/inter_task_kv_manager.py")
    sys.exit(1)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 环境变量配置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = '1'

dir_path = Path("/mnt/sda1/homie_cache/")

# 清理显存
def garbage_collection():
    gc.collect()
    torch.cuda.empty_cache()


# 加载模型和分词器
def start_model(model_path, similarity_threshold, max_cache_size, num_hyperplanes):
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"Model path: {model_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Max cache size: {max_cache_size}")
    print(f"Num hyperplanes: {num_hyperplanes}")
    print(f"{'='*60}\n")
    
    model = LlamaForCausalLMWithKVReuse.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    # 创建KV Manager
    kv_manager = InterTaskKVManager(
        embedding_dim=model.config.hidden_size,
        max_cache_size=max_cache_size,
        similarity_threshold=similarity_threshold,
        num_hyperplanes=num_hyperplanes,
        device=str(device)
    )
    
    # 连接KV Manager到模型
    model.set_kv_manager(kv_manager)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"yes Model loaded successfully!")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    return model, tokenizer, kv_manager

# 停止模型的函数
def stop_model(model):
    # 删除模型对象
    del model
    garbage_collection()
    print("模型已释放，显存已清理")

def clean_cache(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)  # 删除目录及其内容
        print(f"{dir_path} and its contents have been removed.")
    else:
        print(f"{dir_path} does not exist.")


def extract_last_layer_kv(outputs):
    """
    [DEPRECATED] Safely extract last layer KV from model outputs.
    
    NOTE: This function is no longer needed as KV cache is now auto-saved
    inside the model's forward() method during prefill phase. Kept for
    backward compatibility and debugging purposes.
    
    Args:
        outputs: Model generate outputs
        
    Returns:
        last_layer_kv: (key, value) tuple or None
    """
    last_layer_kv = None
    
    if not hasattr(outputs, 'past_key_values') or outputs.past_key_values is None:
        print("[Extract] outputs.past_key_values is None or not present")
        return None
    
    past_key_values = outputs.past_key_values
    
    # Try direct indexing first
    try:
        last_layer_kv = past_key_values[-1]
        if last_layer_kv is not None:
            print(f"[Extract] Direct indexing successful: key.shape={last_layer_kv[0].shape}, value.shape={last_layer_kv[1].shape}")
            return last_layer_kv
    except (KeyError, IndexError, TypeError) as e:
        print(f"[Extract] Direct indexing failed: {e}")
    
    # Try converting to list
    try:
        pkvs = list(past_key_values)
        if len(pkvs) > 0:
            last_layer_kv = pkvs[-1]
            if last_layer_kv is not None:
                print(f"[Extract] List conversion successful: key.shape={last_layer_kv[0].shape}, value.shape={last_layer_kv[1].shape}")
                return last_layer_kv
    except Exception as e:
        print(f"[Extract] List conversion failed: {e}")
    
    # Try iterating
    try:
        last_kv = None
        for kv in past_key_values:
            if kv is not None:
                last_kv = kv
        if last_kv is not None:
            print(f"[Extract] Iteration successful: key.shape={last_kv[0].shape}, value.shape={last_kv[1].shape}")
            return last_kv
    except Exception as e:
        print(f"[Extract] Iteration failed: {e}")
    
    print("[Extract] All extraction methods failed, returning None")
    return None


def _construct_past_key_values_from_cache(matched_entry, num_layers):
    """
    Construct past_key_values from cached KV entry.
    
    For Inter-Task KV Reuse (SimLLM), we construct a past_key_values structure
    where ALL layers have the same cached KV from the last layer.
    
    For transformers 4.36+, we use DynamicCache directly to avoid conversion issues.
    For older versions, we use a tuple of tuples.
    
    Note: This approach reuses the last layer's KV for all layers, which is
    an approximation. The model will use this cached KV as the "prefix"
    computation result for all layers.
    
    Args:
        matched_entry: TaskCacheEntry with top_layer_kv
        num_layers: Number of decoder layers
        
    Returns:
        past_key_values: DynamicCache object (transformers 4.36+) or tuple of tuples
    """
    cached_key, cached_value = matched_entry.top_layer_kv
    
    if HAS_DYNAMIC_CACHE:
        # Use DynamicCache directly for transformers 4.36+
        # This avoids the from_legacy_cache() conversion issues
        cache = DynamicCache()
        for layer_idx in range(num_layers):
            # All layers: use cached KV (clone to avoid reference issues)
            cache.update(cached_key.clone(), cached_value.clone(), layer_idx)
        return cache
    else:
        # Fallback for older transformers versions
        past_key_values = []
        for layer_idx in range(num_layers):
            past_key_values.append((cached_key.clone(), cached_value.clone()))
        return tuple(past_key_values)


def run_inference_with_kv_reuse(input_ids, attention_mask, model, tokenizer, task_id, kv_manager):
    """
    Run inference with KV reuse enabled.
    
    For Cache HIT:
    - Construct past_key_values from cached KV
    - Truncate input_ids to only the last token
    - Pass cached KV to generate() to skip prefill computation
    
    For Cache MISS:
    - Run full inference with complete input_ids
    - Auto-save KV in forward() during prefill phase
    """
    print(f"\n{'='*70}")
    print(f"Running inference for task: {task_id}")
    print(f"Input length: {input_ids.shape[-1]} tokens")
    print(f"{'='*70}")
    
    # 设置终止符
    terminators = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        terminators.append(tokenizer.pad_token_id)
    
    # 首先计算embedding用于相似度搜索
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        task_embedding = kv_manager._compute_task_embedding(inputs_embeds)
    
    # 搜索相似任务
    matched_entry = kv_manager.search_similar_task(task_embedding, task_id=task_id)
    
    # 记录cache状态（在generate之前）
    cache_size_before = kv_manager.get_statistics()['cache_size']
    original_input_len = input_ids.shape[-1]
    
    # Initialize variables
    cache_hit = False
    add_success = False
    outputs = None
    
    start = time.time()
    
    if matched_entry is not None:
        # ============================================================
        # CACHE HIT: Use layer skipping with cached penultimate hidden states
        # ============================================================
        print(f" Cache HIT! Using cached KV from task: {matched_entry.task_id}")
        print(f" LAYER SKIPPING MODE: Skipping layers 0 to N-2")
        cache_hit = True
        
        # Get cached data
        cached_key, cached_value = matched_entry.top_layer_kv
        cached_seq_len = cached_key.shape[2]
        cached_penultimate_hidden = matched_entry.penultimate_hidden_states
        
        print(f"[Cache HIT] Cached KV seq_len: {cached_seq_len}, Input seq_len: {original_input_len}")
        
        if cached_penultimate_hidden is None:
            print(f"[Cache HIT]  No cached penultimate_hidden_states, falling back to full inference")
            # Fall back to full inference if no penultimate hidden states
            cache_hit = False
            matched_entry = None
        else:
            print(f"[Cache HIT] cached_penultimate_hidden.shape: {cached_penultimate_hidden.shape}")
            print(f"[Cache HIT] last_layer_kv key.shape: {cached_key.shape}")
            
            # Use layer skipping: call forward with skip_to_last_layer=True
            # This will skip layers 0 to N-2 and only compute the last layer
            
            # Prepare attention mask for the cached sequence
            new_attention_mask = torch.ones(
                (1, cached_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            
            # Prepare position_ids for the cached sequence
            position_ids = torch.arange(
                0, cached_seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            
            # Prepare cache_position
            cache_position = torch.arange(
                0, cached_seq_len,
                device=input_ids.device
            )
            
            print(f"[Cache HIT] Running forward with skip_to_last_layer=True...")
            
            # Run forward with layer skipping
            with torch.no_grad():
                outputs = model(
                    input_ids=None,  # Not needed when using cached_penultimate_hidden
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,  # Not using past_key_values in the traditional sense
                    inputs_embeds=None,  # Not needed
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=cache_position,
                    skip_to_last_layer=True,
                    cached_penultimate_hidden=cached_penultimate_hidden,
                    last_layer_kv=(cached_key, cached_value),
                )
            
            # Get logits and generate first token using layer skipping
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Greedy decoding for remaining tokens
            # After the first token (generated with layer skipping), we use normal generation
            generated_ids = [next_token_id]
            max_new_tokens = 50  # Generate more tokens for better results
            
            # Get the updated KV cache from the layer skipping forward pass
            # For subsequent tokens, we'll use normal generation with the model
            current_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            for step in range(max_new_tokens - 1):
                if next_token_id.item() in terminators:
                    break
                
                # For subsequent tokens, run normal forward (single token at a time)
                # This is efficient because we're only processing one new token
                with torch.no_grad():
                    step_outputs = model(
                        input_ids=next_token_id,
                        attention_mask=torch.ones(
                            (1, current_input_ids.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        ),
                        position_ids=torch.tensor([[current_input_ids.shape[1] - 1]], device=input_ids.device),
                        use_cache=False,  # Don't use cache for simplicity
                        return_dict=True,
                    )
                
                next_token_logits = step_outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids.append(next_token_id)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
            
            # Combine original input with generated tokens
            all_generated = torch.cat(generated_ids, dim=-1)
            
            # Create a simple output structure
            class SimpleOutput:
                def __init__(self, sequences):
                    self.sequences = sequences
            
            outputs = SimpleOutput(torch.cat([input_ids, all_generated], dim=-1))
            
            print(f"[Cache HIT]  Layer skipping complete, generated {len(generated_ids)} tokens")
        
        add_success = False  # No need to save on cache hit
    
    # Handle cache MISS or fallback case
    if not cache_hit:
        # ============================================================
        # CACHE MISS: Run full inference and auto-save KV
        # ============================================================
        print(f" Cache MISS, running full inference (KV will be auto-saved)...")
        
        # 设置当前task_id，以便forward()中自动保存KV
        model.model.current_task_id = task_id
        print(f"[Cache MISS] Set model.model.current_task_id = {task_id}")
        
        # 运行生成 - 使用确定性生成避免警告
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Generate longer responses
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        
        # 清除当前task_id
        model.model.current_task_id = None
        
        # 检查KV是否被自动保存（通过比较cache size）
        cache_size_after = kv_manager.get_statistics()['cache_size']
        add_success = cache_size_after > cache_size_before
        
        if add_success:
            print(f"[Auto-Save] yes KV auto-saved for task {task_id}")
        else:
            print(f"[Auto-Save]  KV was not auto-saved (may already exist or save failed)")
        
        # VERIFICATION: Check if the task can now be found in cache
        if add_success:
            verify_entry = kv_manager.search_similar_task(task_embedding, task_id=f"verify_{task_id}")
            if verify_entry is not None:
                print(f"[Verify] yes Task {task_id} successfully found in cache (matched: {verify_entry.task_id})")
            else:
                print(f"[Verify]  Task {task_id} NOT found in cache after save!")
    
    total_latency = time.time() - start
    
    # 解码响应
    if hasattr(outputs, 'sequences'):
        # For cache hit with layer skipping, the sequences already contain input + generated
        if cache_hit:
            response_ids = outputs.sequences[0][original_input_len:]  # Skip original input
        else:
            response_ids = outputs.sequences[0][original_input_len:]
    else:
        if cache_hit:
            response_ids = outputs[0][original_input_len:]
        else:
            response_ids = outputs[0][original_input_len:]
    
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print(f"\n🤖 Generated Answer: {response}")
    print(f"⏱️ Latency: {total_latency:.3f}s")
    print(f"📊 Cache Hit: {cache_hit}")
    
    # Dump cache state after each request
    kv_manager.dump_cache_state(top_n=5)
    
    print(f"{'='*70}\n")
    
    return total_latency, response, cache_hit, add_success


def diagnostic_run(prompts, model, tokenizer, kv_manager):
    """
    Diagnostic run to verify KV cache is being saved correctly
    
    Args:
        prompts: List of prompts to test
        model: The model
        tokenizer: The tokenizer
        kv_manager: The KV manager
        
    Returns:
        True if all assertions pass, False otherwise
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC RUN")
    print("="*80)
    
    initial_cache_size = kv_manager.get_statistics()['cache_size']
    print(f"Initial cache size: {initial_cache_size}")
    
    add_results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Diagnostic prompt {i+1}/{len(prompts)}: '{prompt}' ---")
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        task_id = f"diag_{i}"
        
        # Run inference
        latency, response, cache_hit, add_success = run_inference_with_kv_reuse(
            input_ids, attention_mask, model, tokenizer, task_id, kv_manager
        )
        
        add_results.append({
            'prompt': prompt,
            'task_id': task_id,
            'cache_hit': cache_hit,
            'add_success': add_success
        })
        
        garbage_collection()
        time.sleep(0.5)
    
    # Check final cache size
    final_cache_size = kv_manager.get_statistics()['cache_size']
    print(f"\nFinal cache size: {final_cache_size}")
    
    # Count statistics
    cache_misses = sum(1 for r in add_results if not r['cache_hit'])
    cache_hits = sum(1 for r in add_results if r['cache_hit'])
    successful_saves = sum(1 for r in add_results if not r['cache_hit'] and r['add_success'])
    failed_saves = sum(1 for r in add_results if not r['cache_hit'] and not r['add_success'])
    actual_increase = final_cache_size - initial_cache_size
    
    print(f"\nDiagnostic Summary:")
    print(f"  Prompts processed: {len(prompts)}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Successful saves: {successful_saves}")
    print(f"  Failed saves: {failed_saves}")
    print(f"  Cache size increase: {actual_increase}")
    
    # CRITICAL ASSERTION 1: If all queries were misses and all saves failed, test MUST fail
    if cache_misses > 0 and successful_saves == 0:
        print(f"\nno CRITICAL ASSERTION FAILED: All {cache_misses} cache misses failed to save!")
        print("This indicates a fundamental problem with the auto-save logic.")
        print("Debug info:")
        for r in add_results:
            print(f"  {r['task_id']}: cache_hit={r['cache_hit']}, add_success={r['add_success']}")
        return False
    
    # CRITICAL ASSERTION 2: Final cache size must be > 0 if any saves were attempted
    if cache_misses > 0 and final_cache_size == 0:
        print(f"\nno CRITICAL ASSERTION FAILED: final_cache_size == 0 after {cache_misses} cache misses!")
        print("KV cache is not being saved at all.")
        return False
    
    # ASSERTION 3: Cache size increase should match successful saves
    if actual_increase < successful_saves:
        print(f"\nno ASSERTION FAILED: Cache size increase ({actual_increase}) < successful saves ({successful_saves})")
        print("Debug info:")
        for r in add_results:
            print(f"  {r['task_id']}: cache_hit={r['cache_hit']}, add_success={r['add_success']}")
        return False
    
    print(f"\nyes ASSERTION PASSED: Cache is working correctly")
    print(f"   - {successful_saves}/{cache_misses} cache misses were saved successfully")
    print(f"   - Final cache size: {final_cache_size}")
    return True


def test_cache_functionality(model, tokenizer, kv_manager, args):
    """
    Test cache hit/miss functionality with similar prompts
    使用简短相似的句子进行测试
    """
    print("\n" + "="*80)
    print("CACHE FUNCTIONALITY TEST")
    print("Testing with short similar sentences")
    print("="*80)
    
    # 定义测试用例：使用简短相似的句子
    test_cases = [
        # 第一组：关于人工智能的中文句子（高度相似）- 主要测试用例
        {
            "prompts": [
                "什么是人工智能",
                "人工智能是什么",
                "解释人工智能的含义",
                "人工智能的定义是什么",
                "请说明人工智能的概念",
                "能否解释一下人工智能？",
                "人工智能指的是什么？",
                "请描述人工智能的作用",
                "人工智能有哪些应用？",
                "人工智能的应用是什么",
            ],
            "group": "AI questions (Chinese) - Primary Test"
        },
        # 第二组：关于机器学习的英文句子
        {
            "prompts": [
                "What is machine learning?",
                "Explain machine learning.",
                "Define machine learning.",
                "Can you describe machine learning?",
                "What does machine learning mean?",
                "How would you explain machine learning?",
                "What are the applications of machine learning?",
                "Describe the concept of machine learning.",
                "What is the definition of machine learning?",
                "Please explain the meaning of machine learning.",
            ],
            "group": "ML questions (English)"
        },
    ]
    
    results = []
    total_hits = 0
    total_queries = 0
    
    for group_idx, test_group in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Testing Group {group_idx + 1}: {test_group['group']}")
        print(f"{'='*60}")
        
        for prompt_idx, prompt in enumerate(test_group['prompts']):
            total_queries += 1
            
            # 直接tokenize，不使用chat template
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
            )
            
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # 生成task_id
            task_id = f"group{group_idx}_prompt{prompt_idx}"
            
            # 运行推理
            latency, response, cache_hit, add_success = run_inference_with_kv_reuse(
                input_ids, attention_mask, model, tokenizer, task_id, kv_manager
            )
            
            if cache_hit:
                total_hits += 1
            
            results.append({
                "group": test_group['group'],
                "prompt": prompt,
                "latency": latency,
                "cache_hit": cache_hit,
                "add_success": add_success,
                "response": response  # Store full response
            })
            
            # 清理显存
            garbage_collection()
            time.sleep(0.5)
    
    # 打印汇总结果
    print("\n" + "="*100)
    print("TEST RESULTS SUMMARY")
    print("="*100)
    
    # 打印表头
    print(f"\n{'#':<4} {'Prompt':<35} {'Hit':<6} {'Latency':<10} {'Response (first 100 chars)':<100}")
    print("-"*155)
    
    for idx, r in enumerate(results):
        hit_str = "HIT" if r['cache_hit'] else "MISS"
        # Truncate response for display but keep more characters
        response_display = r['response'][:100].replace('\n', ' ')
        if len(r['response']) > 100:
            response_display += "..."
        print(f"{idx+1:<4} {r['prompt'][:33]:<35} {hit_str:<6} {r['latency']:.3f}s    {response_display}")
    
    print("\n" + "-"*100)
    
    # 打印详细的生成结果
    print("\n" + "="*100)
    print("DETAILED GENERATED RESPONSES")
    print("="*100)
    
    for idx, r in enumerate(results):
        hit_str = " Cache HIT (KV Reuse)" if r['cache_hit'] else " Cache MISS (Full Compute)"
        print(f"\n[{idx+1}] Prompt: {r['prompt']}")
        print(f"    Status: {hit_str}")
        print(f"    Latency: {r['latency']:.3f}s")
        print(f"    Response: {r['response']}")
        print("-"*80)
    
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    
    hit_rate = total_hits / total_queries if total_queries > 0 else 0
    print(f"\nTotal Queries: {total_queries}")
    print(f"Cache Hits: {total_hits}")
    print(f"Cache Misses: {total_queries - total_hits}")
    print(f"Hit Rate: {hit_rate:.2%}")
    
    # 计算平均延迟
    hit_latencies = [r['latency'] for r in results if r['cache_hit']]
    miss_latencies = [r['latency'] for r in results if not r['cache_hit']]
    
    if hit_latencies:
        avg_hit_latency = sum(hit_latencies) / len(hit_latencies)
        print(f"\nAverage Cache HIT Latency: {avg_hit_latency:.3f}s")
    if miss_latencies:
        avg_miss_latency = sum(miss_latencies) / len(miss_latencies)
        print(f"Average Cache MISS Latency: {avg_miss_latency:.3f}s")
    if hit_latencies and miss_latencies:
        speedup = avg_miss_latency / avg_hit_latency
        print(f"Speedup (MISS/HIT): {speedup:.2f}x")
    
    # 打印KV Manager统计
    stats = kv_manager.get_statistics()
    print(f"\nKV Manager Statistics:")
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Num Buckets: {stats['num_buckets']}")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Misses: {stats['cache_misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    
    return results


def main():
    args = parse_args()  # 解析命令行参数
    
    print("\n" + "="*80)
    print("INTER-TASK KV REUSE TEST")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Max Cache Size: {args.max_cache_size}")
    print(f"Num Hyperplanes: {args.num_hyperplanes}")
    print("="*80 + "\n")
    
    # 加载模型
    model, tokenizer, kv_manager = start_model(
        args.model_path, 
        args.similarity_threshold, 
        args.max_cache_size, 
        args.num_hyperplanes
    )
    
    time.sleep(2)
    
    # 首先运行诊断测试
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC TEST FIRST")
    print("="*80)
    
    diagnostic_prompts = [
        "什么是人工智能",
        "人工智能是什么",
        "解释人工智能的含义",
    ]
    
    diag_success = diagnostic_run(diagnostic_prompts, model, tokenizer, kv_manager)
    
    if not diag_success:
        print("\n Diagnostic test failed! Check the logs above for details.")
    
    # 重置缓存后运行完整测试
    kv_manager.reset()
    
    # 运行缓存测试
    if args.cache_test:
        results = test_cache_functionality(model, tokenizer, kv_manager, args)
    
    # 如果需要，可以添加TTFT测试和F1测试
    if args.ttft_test:
        print("\nTTFT test not implemented in this version")
    
    if args.f1_test:
        print("\nF1 test not implemented in this version")
    
    # 汇总结果
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    # 停止模型，释放显存
    stop_model(model)
    clean_cache(dir_path)


if __name__ == "__main__":
    main()
