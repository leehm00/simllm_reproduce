#!/usr/bin/env python3
"""
Example script demonstrating Inter-Task KV Reuse with Sim-LLM

This script shows how to use the Inter-Task KV Reuse feature to optimize
inference by reusing KV cache from similar previous requests.
"""

import argparse
import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Sim-LLM Inter-Task KV Reuse Demo")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct",
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--max_cache_size", 
        type=int, 
        default=100,
        help="Maximum number of cached tasks"
    )
    parser.add_argument(
        "--similarity_threshold", 
        type=float, 
        default=0.8,
        help="Cosine similarity threshold for cache hit (0-1)"
    )
    parser.add_argument(
        "--num_hyperplanes", 
        type=int, 
        default=16,
        help="Number of hyperplanes for LSH"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--demo_mode", 
        type=str, 
        default="basic",
        choices=["basic", "similarity", "benchmark"],
        help="Demo mode to run"
    )
    return parser.parse_args()


def demo_basic_usage(inference, args):
    """Basic usage demonstration"""
    print("\n" + "="*60)
    print("DEMO: Basic Usage")
    print("="*60)
    
    # First query - will be a cache miss
    prompt1 = "What is machine learning?"
    print(f"\n[Query 1] {prompt1}")
    response1 = inference.generate(prompt1, max_new_tokens=args.max_new_tokens)
    print(f"[Response] {response1}")
    
    # Second query - similar topic, may be a cache hit
    prompt2 = "What is deep learning?"
    print(f"\n[Query 2] {prompt2}")
    response2 = inference.generate(prompt2, max_new_tokens=args.max_new_tokens)
    print(f"[Response] {response2}")
    
    # Third query - different topic, likely cache miss
    prompt3 = "What is the capital of France?"
    print(f"\n[Query 3] {prompt3}")
    response3 = inference.generate(prompt3, max_new_tokens=args.max_new_tokens)
    print(f"[Response] {response3}")
    
    # Fourth query - similar to first, may be cache hit
    prompt4 = "Explain machine learning in simple terms."
    print(f"\n[Query 4] {prompt4}")
    response4 = inference.generate(prompt4, max_new_tokens=args.max_new_tokens)
    print(f"[Response] {response4}")
    
    # Print statistics
    stats = inference.get_statistics()
    print("\n" + "-"*40)
    print("Cache Statistics:")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Misses: {stats['cache_misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  Cache Size: {stats['cache_size']}")
    print(f"  Avg Inference Time: {stats['avg_inference_time']:.3f}s")


def demo_similarity_threshold(inference, args):
    """Demonstrate effect of similarity threshold"""
    print("\n" + "="*60)
    print("DEMO: Similarity Threshold Effect")
    print("="*60)
    
    # Test with different thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9]
    
    base_prompt = "What is artificial intelligence?"
    similar_prompts = [
        "What is AI?",
        "Explain artificial intelligence.",
        "Define AI technology.",
        "What does artificial intelligence mean?"
    ]
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        inference.reset_cache()
        inference.set_similarity_threshold(threshold)
        
        # First query
        print(f"[Base Query] {base_prompt}")
        inference.generate(base_prompt, max_new_tokens=20)
        
        # Similar queries
        for prompt in similar_prompts:
            print(f"[Similar Query] {prompt}")
            inference.generate(prompt, max_new_tokens=20)
        
        stats = inference.get_statistics()
        print(f"Hit Rate: {stats['hit_rate']:.2%} ({stats['cache_hits']}/{stats['total_queries']})")


def demo_benchmark(inference, args):
    """Benchmark KV reuse performance"""
    print("\n" + "="*60)
    print("DEMO: Performance Benchmark")
    print("="*60)
    
    # Generate test prompts
    test_prompts = [
        # Similar prompts (should benefit from KV reuse)
        "What is machine learning?",
        "Explain machine learning.",
        "Define machine learning.",
        "What does machine learning mean?",
        "How does machine learning work?",
        
        # Different topic
        "What is the weather like today?",
        "Tell me about the weather.",
        "How is the weather?",
        
        # Another topic
        "What is Python programming?",
        "Explain Python language.",
        "What is Python used for?",
    ]
    
    # Benchmark with KV reuse enabled
    print("\n--- With KV Reuse ---")
    inference.reset_cache()
    start_time = time.time()
    
    for prompt in test_prompts:
        inference.generate(prompt, max_new_tokens=30, enable_kv_reuse=True)
    
    time_with_reuse = time.time() - start_time
    stats_with_reuse = inference.get_statistics()
    
    print(f"Total Time: {time_with_reuse:.2f}s")
    print(f"Cache Hit Rate: {stats_with_reuse['hit_rate']:.2%}")
    print(f"Avg Inference Time: {stats_with_reuse['avg_inference_time']:.3f}s")
    
    # Benchmark without KV reuse
    print("\n--- Without KV Reuse ---")
    inference.reset_cache()
    start_time = time.time()
    
    for prompt in test_prompts:
        inference.generate(prompt, max_new_tokens=30, enable_kv_reuse=False)
    
    time_without_reuse = time.time() - start_time
    stats_without_reuse = inference.get_statistics()
    
    print(f"Total Time: {time_without_reuse:.2f}s")
    print(f"Avg Inference Time: {stats_without_reuse['avg_inference_time']:.3f}s")
    
    # Summary
    print("\n--- Summary ---")
    speedup = time_without_reuse / time_with_reuse if time_with_reuse > 0 else 0
    print(f"Speedup with KV Reuse: {speedup:.2f}x")


def main():
    args = parse_args()
    
    print("="*60)
    print("Sim-LLM Inter-Task KV Reuse Demo")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Max Cache Size: {args.max_cache_size}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Device: {args.device}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Import and create inference wrapper
    try:
        from models.sim_llm_inference import SimLLMInference
        
        inference = SimLLMInference.from_pretrained(
            model_path=args.model_path,
            max_cache_size=args.max_cache_size,
            similarity_threshold=args.similarity_threshold,
            num_hyperplanes=args.num_hyperplanes,
            device=args.device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nRunning in mock mode for demonstration...")
        
        # Create mock inference for demonstration
        class MockInference:
            def __init__(self):
                self.cache_hits = 0
                self.cache_misses = 0
                self.total_queries = 0
                self.total_time = 0
                self.threshold = 0.8
                self.cache = {}
            
            def generate(self, prompt, max_new_tokens=50, enable_kv_reuse=True):
                self.total_queries += 1
                time.sleep(0.1)  # Simulate inference
                self.total_time += 0.1
                
                # Simple similarity check
                hit = False
                if enable_kv_reuse:
                    for cached_prompt in self.cache:
                        # Simple word overlap similarity
                        words1 = set(prompt.lower().split())
                        words2 = set(cached_prompt.lower().split())
                        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                        if overlap > self.threshold:
                            hit = True
                            break
                
                if hit:
                    self.cache_hits += 1
                    print(f"  [Cache HIT]")
                else:
                    self.cache_misses += 1
                    self.cache[prompt] = True
                    print(f"  [Cache MISS]")
                
                return f"Mock response for: {prompt[:30]}..."
            
            def get_statistics(self):
                return {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'total_queries': self.total_queries,
                    'hit_rate': self.cache_hits / max(self.total_queries, 1),
                    'cache_size': len(self.cache),
                    'avg_inference_time': self.total_time / max(self.total_queries, 1)
                }
            
            def reset_cache(self):
                self.cache = {}
                self.cache_hits = 0
                self.cache_misses = 0
                self.total_queries = 0
                self.total_time = 0
            
            def set_similarity_threshold(self, threshold):
                self.threshold = threshold
        
        inference = MockInference()
    
    # Run selected demo
    if args.demo_mode == "basic":
        demo_basic_usage(inference, args)
    elif args.demo_mode == "similarity":
        demo_similarity_threshold(inference, args)
    elif args.demo_mode == "benchmark":
        demo_benchmark(inference, args)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
