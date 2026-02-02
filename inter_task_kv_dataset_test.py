#!/usr/bin/env python3
"""
Inter-Task KV Reuse Dataset Test Script
Based on llama3_args.py pattern
Tests the KV cache reuse functionality with dataset-based evaluation
Supports TTFT latency test and F1 score test
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
from utils import load_datasets, compute_f1

# Try to import DynamicCache for transformers 4.36+
try:
    from transformers.cache_utils import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
    DynamicCache = None

import shutil


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Inter-Task KV Reuse Dataset Test Script")
    # Test mode flags
    parser.add_argument("--ttft_test", type=bool, default=False, help="Enable TTFT latency test")
    parser.add_argument("--f1_test", type=bool, default=True, help="Enable F1 score test")
    
    # Inter-Task KV specific parameters
    parser.add_argument("--similarity_threshold", type=float, default=0.7, 
                       help="Cosine similarity threshold for cache hit (0-1)")
    parser.add_argument("--max_cache_size", type=int, default=100, 
                       help="Maximum number of cached tasks")
    parser.add_argument("--num_hyperplanes", type=int, default=16, 
                       help="Number of hyperplanes for LSH")
    
    # Dataset and model parameters
    parser.add_argument("--fromdataset", type=str, 
                       default='/home/homie/homie/fuzzy_llama_submit/datasets/wiki_for_test.json',
                       help="Path to dataset JSON file")
    parser.add_argument("--max_count", type=int, default=100, 
                       help="Maximum number of samples to process")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct",
                       help="Path to pretrained model")
    return parser.parse_args()


# Import model after argument parsing to avoid import errors
try:
    from models.modeling_llama_inter_task_kv import LlamaForCausalLMWithKVReuse
    from models.inter_task_kv_manager import InterTaskKVManager
    print("成功加载Inter-Task KV Reuse模型代码！")
except ImportError as e:
    print(f"导入错误：{e}")
    print("请确认：")
    print("1. 当前目录存在 models/modeling_llama_inter_task_kv.py")
    print("2. 当前目录存在 models/inter_task_kv_manager.py")
    sys.exit(1)

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment configuration
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = '1'

dir_path = Path("/mnt/sda1/homie_cache/")


def garbage_collection():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def start_model(model_path, similarity_threshold, max_cache_size, num_hyperplanes):
    """Load model and tokenizer with KV manager"""
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
    
    # Create KV Manager
    kv_manager = InterTaskKVManager(
        embedding_dim=model.config.hidden_size,
        max_cache_size=max_cache_size,
        similarity_threshold=similarity_threshold,
        num_hyperplanes=num_hyperplanes,
        device=str(device)
    )
    
    # Connect KV Manager to model
    model.set_kv_manager(kv_manager)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Model loaded successfully!")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    return model, tokenizer, kv_manager


def stop_model(model):
    """Release model resources"""
    del model
    garbage_collection()
    print("模型已释放，显存已清理")


def clean_cache(dir_path):
    """Clean cache directory"""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        print(f"{dir_path} and its contents have been removed.")
    else:
        print(f"{dir_path} does not exist.")


def run_inference_with_kv_reuse(input_ids, attention_mask, model, tokenizer, task_id, kv_manager, max_new_tokens=1):
    """
    Run inference with KV reuse enabled.
    
    For Cache HIT:
    - Use cached KV with layer skipping
    
    For Cache MISS:
    - Run full inference and auto-save KV
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        model: The model
        tokenizer: The tokenizer
        task_id: Unique task identifier
        kv_manager: KV cache manager
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        latency: Time taken for inference
        response: Generated text
        cache_hit: Whether cache was hit
    """
    # Set terminators
    terminators = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        terminators.append(tokenizer.pad_token_id)
    
    # Compute embedding for similarity search
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        task_embedding = kv_manager._compute_task_embedding(inputs_embeds)
    
    # Search for similar task
    matched_entry = kv_manager.search_similar_task(task_embedding, task_id=task_id)
    
    # Record cache state before generation
    cache_size_before = kv_manager.get_statistics()['cache_size']
    original_input_len = input_ids.shape[-1]
    
    cache_hit = False
    outputs = None
    
    start = time.time()
    
    if matched_entry is not None:
        # ============================================================
        # CACHE HIT: Use layer skipping with cached penultimate hidden states
        # ============================================================
        print(f"  Cache HIT! Using cached KV from task: {matched_entry.task_id}")
        cache_hit = True
        
        # Get cached data
        cached_key, cached_value = matched_entry.top_layer_kv
        cached_seq_len = cached_key.shape[2]
        cached_penultimate_hidden = matched_entry.penultimate_hidden_states
        
        if cached_penultimate_hidden is None:
            print(f"  No cached penultimate_hidden_states, falling back to full inference")
            cache_hit = False
            matched_entry = None
        else:
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
            
            # Run forward with layer skipping
            with torch.no_grad():
                outputs = model(
                    input_ids=None,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=cache_position,
                    skip_to_last_layer=True,
                    cached_penultimate_hidden=cached_penultimate_hidden,
                    last_layer_kv=(cached_key, cached_value),
                )
            
            # Get logits and generate first token
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Greedy decoding for remaining tokens
            generated_ids = [next_token_id]
            current_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            for step in range(max_new_tokens - 1):
                if next_token_id.item() in terminators:
                    break
                
                with torch.no_grad():
                    step_outputs = model(
                        input_ids=next_token_id,
                        attention_mask=torch.ones(
                            (1, current_input_ids.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        ),
                        position_ids=torch.tensor([[current_input_ids.shape[1] - 1]], device=input_ids.device),
                        use_cache=False,
                        return_dict=True,
                    )
                
                next_token_logits = step_outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids.append(next_token_id)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
            
            # Combine generated tokens
            all_generated = torch.cat(generated_ids, dim=-1)
            
            class SimpleOutput:
                def __init__(self, sequences):
                    self.sequences = sequences
            
            outputs = SimpleOutput(torch.cat([input_ids, all_generated], dim=-1))
    
    # Handle cache MISS or fallback case
    if not cache_hit:
        # ============================================================
        # CACHE MISS: Run full inference and auto-save KV
        # ============================================================
        print(f"  Cache MISS, running full inference...")
        
        # Set current task_id for auto-save in forward()
        model.model.current_task_id = task_id
        
        # Run generation
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        
        # Clear current task_id
        model.model.current_task_id = None
        
        # Check if KV was auto-saved
        cache_size_after = kv_manager.get_statistics()['cache_size']
        if cache_size_after > cache_size_before:
            print(f"  KV auto-saved for task {task_id}")
    
    total_latency = time.time() - start
    
    # Decode response
    if hasattr(outputs, 'sequences'):
        response_ids = outputs.sequences[0][original_input_len:]
    else:
        response_ids = outputs[0][original_input_len:]
    
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return total_latency, response, cache_hit


def run_inference_ttft(input_ids, attention_mask, model, tokenizer, task_id, kv_manager):
    """Run TTFT (Time To First Token) test with KV reuse"""
    return run_inference_with_kv_reuse(
        input_ids, attention_mask, model, tokenizer, task_id, kv_manager, max_new_tokens=1
    )


def run_inference_f1(input_ids, attention_mask, model, tokenizer, task_id, kv_manager):
    """Run F1 test with KV reuse"""
    return run_inference_with_kv_reuse(
        input_ids, attention_mask, model, tokenizer, task_id, kv_manager, max_new_tokens=10
    )


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("INTER-TASK KV REUSE DATASET TEST")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.fromdataset}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Max Cache Size: {args.max_cache_size}")
    print(f"Num Hyperplanes: {args.num_hyperplanes}")
    print(f"Max Count: {args.max_count}")
    print(f"TTFT Test: {args.ttft_test}")
    print(f"F1 Test: {args.f1_test}")
    print("="*80 + "\n")
    
    # Load model
    model, tokenizer, kv_manager = start_model(
        args.model_path, 
        args.similarity_threshold, 
        args.max_cache_size, 
        args.num_hyperplanes
    )
    
    time.sleep(5)
    
    # Load dataset
    ds = load_datasets(args.fromdataset)
    
    # Result storage
    ttft_full = []
    excutetime_full = []
    f1_full = []
    cache_hits = []
    
    # Prompt templates
    prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
    query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
    
    count = 0
    
    # TTFT Test
    if args.ttft_test:
        print("\n" + "="*80)
        print("RUNNING TTFT TEST")
        print("="*80)
        
        for id, ex in enumerate(ds.get("filtered_items", ds)):
            # Handle different dataset formats
            if "text" in ex:
                doc_prompt = ex["text"]
            elif "context" in ex:
                doc_prompt = ex["context"]
            else:
                continue
            
            input_len = len(tokenizer.encode(doc_prompt)[1:])
            if input_len > 20000:
                doc_prompt = doc_prompt[:9000]
            if input_len < 1000:
                continue
            
            print(f"input_len: {input_len}")
            count += 1
            print(f"current id: {count}")
            
            if count == 35 or count == 36:
                continue
            if count > args.max_count:
                break
            
            input_prompt = prefix_prompt + doc_prompt + query_prompt
            time.sleep(1)
            
            messages = [
                {"role": "system", "content": "You are an assistant who provides precise and direct answers."},
                {"role": "user", "content": input_prompt},
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            attention_mask = torch.ones_like(input_ids)
            task_id = f"ttft_task_{count}"
            
            ttft, response, cache_hit = run_inference_ttft(
                input_ids, attention_mask, model, tokenizer, task_id, kv_manager
            )
            garbage_collection()
            
            ttft_full.append(ttft)
            cache_hits.append(cache_hit)
            
            print(f"🤖 Generated: {response}")
            print(f"⏱️ TTFT: {ttft:.3f}s")
            print(f"📊 Cache Hit: {cache_hit}")
            print(f"TTFTs = {ttft_full}")
            
            # Calculate statistics
            hit_ttfts = [t for t, h in zip(ttft_full, cache_hits) if h]
            miss_ttfts = [t for t, h in zip(ttft_full, cache_hits) if not h]
            
            if hit_ttfts:
                print(f"TTFT_with_cache_mean = {np.mean(hit_ttfts):.4f}")
            if miss_ttfts:
                print(f"TTFT_with_full_prefill_mean = {np.mean(miss_ttfts):.4f}")
            
            hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0
            print(f"Cache_hit_rate = {hit_rate:.2%}")
    
    # F1 Test
    if args.f1_test:
        print("\n" + "="*80)
        print("RUNNING F1 TEST")
        print("="*80)
        
        count = 0
        cache_hits = []
        
        for id, ex in enumerate(ds):
            count += 1
            print(f"current id: {count}")
            
            if count > args.max_count:
                break
            
            # Extract data from dataset
            answers = ex.get("answer", ex.get("answers", ""))
            doc_prompt = ex.get("context", ex.get("text", ""))
            question_prompt = ex.get("question", "")
            
            input_len = len(tokenizer.encode(doc_prompt)[1:])
            if input_len > 2000:
                doc_prompt = doc_prompt[:20000]
            
            input_prompt = prefix_prompt + doc_prompt + query_prompt + question_prompt
            
            time.sleep(1)
            
            messages = [
                {"role": "system", "content": "You are an assistant who provides precise and direct answers."},
                {"role": "user", "content": input_prompt},
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            attention_mask = torch.ones_like(input_ids)
            task_id = f"f1_task_{count}"
            
            total_latency, output, cache_hit = run_inference_f1(
                input_ids, attention_mask, model, tokenizer, task_id, kv_manager
            )
            garbage_collection()
            
            # Compute F1 score
            if isinstance(answers, list):
                f1 = max([compute_f1(output, ans, tokenizer) for ans in answers])
            else:
                f1 = compute_f1(output, answers, tokenizer)
            
            f1_full.append(f1)
            excutetime_full.append(total_latency)
            cache_hits.append(cache_hit)
            
            print(f"🤖 Generated Answer: {output}")
            print(f"📝 Expected: {answers}")
            print(f"📊 F1 Score: {f1:.4f}")
            print(f"⏱️ Latency: {total_latency:.3f}s")
            print(f"📊 Cache Hit: {cache_hit}")
            print(f"F1s = {f1_full}")
            
            # Calculate statistics
            hit_f1s = [f for f, h in zip(f1_full, cache_hits) if h]
            miss_f1s = [f for f, h in zip(f1_full, cache_hits) if not h]
            hit_latencies = [l for l, h in zip(excutetime_full, cache_hits) if h]
            miss_latencies = [l for l, h in zip(excutetime_full, cache_hits) if not h]
            
            if hit_f1s:
                print(f"Average_f1_with_cache = {np.mean(hit_f1s):.4f}")
            if miss_f1s:
                print(f"Average_f1_full_prefill = {np.mean(miss_f1s):.4f}")
            if hit_latencies:
                print(f"Average_latency_with_cache = {np.mean(hit_latencies):.4f}")
            if miss_latencies:
                print(f"Average_latency_full_prefill = {np.mean(miss_latencies):.4f}")
            
            hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0
            print(f"Cache_hit_rate = {hit_rate:.2%}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if ttft_full:
        print(f"TTFTs = {ttft_full}")
        hit_ttfts = [t for t, h in zip(ttft_full, cache_hits) if h]
        miss_ttfts = [t for t, h in zip(ttft_full, cache_hits) if not h]
        if hit_ttfts:
            print(f"TTFT_with_cache_mean = {np.mean(hit_ttfts):.4f}")
        if miss_ttfts:
            print(f"TTFT_with_full_prefill_mean = {np.mean(miss_ttfts):.4f}")
    
    if f1_full:
        print(f"F1s = {f1_full}")
        hit_f1s = [f for f, h in zip(f1_full, cache_hits) if h]
        miss_f1s = [f for f, h in zip(f1_full, cache_hits) if not h]
        if hit_f1s:
            print(f"Average_f1_with_cache = {np.mean(hit_f1s):.4f}")
        if miss_f1s:
            print(f"Average_f1_full_prefill = {np.mean(miss_f1s):.4f}")
    
    if excutetime_full:
        hit_latencies = [l for l, h in zip(excutetime_full, cache_hits) if h]
        miss_latencies = [l for l, h in zip(excutetime_full, cache_hits) if not h]
        if hit_latencies:
            print(f"Average_latency_with_cache = {np.mean(hit_latencies):.4f}")
        if miss_latencies:
            print(f"Average_latency_full_prefill = {np.mean(miss_latencies):.4f}")
    
    if cache_hits:
        hit_rate = sum(cache_hits) / len(cache_hits)
        print(f"Total_cache_hit_rate = {hit_rate:.2%}")
    
    # Print KV Manager statistics
    stats = kv_manager.get_statistics()
    print(f"\nKV Manager Statistics:")
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Num Buckets: {stats['num_buckets']}")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Misses: {stats['cache_misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    
    print("="*80)
    
    # Stop model and clean cache
    stop_model(model)
    clean_cache(dir_path)


if __name__ == "__main__":
    main()
