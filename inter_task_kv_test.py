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
    print("✅ 成功加载Inter-Task KV Reuse模型代码！")
except ImportError as e:
    print(f"❌ 导入错误：{e}")
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
    
    print(f"✅ Model loaded successfully!")
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


def run_inference_with_kv_reuse(input_ids, attention_mask, model, tokenizer, task_id, kv_manager):
    """
    Run inference with KV reuse enabled
    """
    print(f"\n{'='*60}")
    print(f"Running inference for task: {task_id}")
    print(f"Input length: {input_ids.shape[-1]} tokens")
    print(f"{'='*60}")
    
    # 设置终止符
    terminators = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        terminators.append(tokenizer.pad_token_id)
    
    # 首先计算embedding用于相似度搜索
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        task_embedding = kv_manager._compute_task_embedding(inputs_embeds)
    
    # 搜索相似任务
    matched_entry = kv_manager.search_similar_task(task_embedding)
    
    start = time.time()
    
    if matched_entry is not None:
        print(f"🎯 Using cached KV from task: {matched_entry.task_id}")
        cache_hit = True
    else:
        print(f"📝 No cache hit, running full inference...")
        cache_hit = False
    
    # 运行生成 - 使用确定性生成避免警告
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            temperature=1.0,  # 显式设置避免警告
            top_p=1.0,        # 显式设置避免警告
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
    
    total_latency = time.time() - start
    
    # 如果是cache miss，保存KV到缓存
    if not cache_hit and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        last_layer_kv = outputs.past_key_values[-1]
        if last_layer_kv is not None:
            kv_manager.add_task(
                task_id=task_id,
                task_embedding=task_embedding,
                top_layer_kv=last_layer_kv
            )
    
    # 解码响应
    if hasattr(outputs, 'sequences'):
        response_ids = outputs.sequences[0][input_ids.shape[-1]:]
    else:
        response_ids = outputs[0][input_ids.shape[-1]:]
    
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print(f"🤖 Generated Answer: {response}")
    print(f"⏱️ Latency: {total_latency:.3f}s")
    print(f"{'='*60}\n")
    
    return total_latency, response, cache_hit


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
        # 第一组：关于人工智能的中文句子（高度相似）
        {
            "prompts": [
                "什么是人工智能",
                "人工智能是什么",
                "解释人工智能的含义",
                "人工智能的定义是什么",
                "给我解释什么是人工智能",
            ],
            "group": "AI questions (Chinese)"
        },
        # 第二组：关于机器学习的英文句子
        {
            "prompts": [
                "What is machine learning?",
                "Explain machine learning.",
                "Define machine learning.",
                "What does machine learning mean?",
            ],
            "group": "ML questions (English)"
        },
        # 第三组：关于猫的简短句子
        {
            "prompts": [
                "The cat is sleeping.",
                "The cat is eating.",
                "The cat is playing.",
                "A cat is sleeping.",
            ],
            "group": "Cat sentences"
        },
        # 第四组：完全不同的句子
        {
            "prompts": [
                "Hello world!",
                "Good morning!",
                "今天天气怎么样",
            ],
            "group": "Unrelated sentences"
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
            latency, response, cache_hit = run_inference_with_kv_reuse(
                input_ids, attention_mask, model, tokenizer, task_id, kv_manager
            )
            
            if cache_hit:
                total_hits += 1
            
            results.append({
                "group": test_group['group'],
                "prompt": prompt,
                "latency": latency,
                "cache_hit": cache_hit,
                "response": response[:50] + "..." if len(response) > 50 else response
            })
            
            # 清理显存
            garbage_collection()
            time.sleep(0.5)
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Prompt':<30} {'Cache Hit':<12} {'Latency':<10}")
    print("-"*52)
    for r in results:
        hit_str = "✅ HIT" if r['cache_hit'] else "❌ MISS"
        print(f"{r['prompt'][:28]:<30} {hit_str:<12} {r['latency']:.3f}s")
    
    print("\n" + "-"*52)
    hit_rate = total_hits / total_queries if total_queries > 0 else 0
    print(f"Total Queries: {total_queries}")
    print(f"Cache Hits: {total_hits}")
    print(f"Cache Misses: {total_queries - total_hits}")
    print(f"Hit Rate: {hit_rate:.2%}")
    
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
