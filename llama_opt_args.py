import argparse
import gc
import torch
import numpy as np
import time
import sys
import os
from transformers import AutoTokenizer
from pathlib import Path
from utils import load_datasets, calculate_ppl_long_text, calculate_ppl, compute_f1

import shutil

# 在main函数前定义解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser(description="Llama OPT Model Inference Script")
    parser.add_argument("--ttft_test", type=bool, default=False, help="Enable TTFT latency test")
    parser.add_argument("--f1_test", type=bool, default=True, help="Enable F1 score test")
    parser.add_argument("--threshold", type=float, default=0.9, 
                       help="KV cache similarity threshold (0-1)")
    parser.add_argument("--max_kv_size", type=int, default=512, 
                       help="Maximum KV cache size")
    parser.add_argument("--eviction_mode", type=str, default="LRU", 
                       choices=["LRU", "FIFO"],
                       help="KV cache eviction mode")
    parser.add_argument("--num_trained_encoders", type=int, default=1,
                       help="Number of encoders to train")
    parser.add_argument("--num_encoders", type=int, default=8,
                       help="Total number of encoders")
    parser.add_argument("--target_layer", type=int, default=-1,
                       help="Target layer for KV generation (-1 for last layer)")
    parser.add_argument("--fromdataset", type=str, 
                       default='/home/homie/homie/fuzzy_llama_submit/datasets/wiki_for_test.json',
                       help="Path to dataset JSON file")
    parser.add_argument("--max_count", type=int, default=100, 
                       help="Maximum number of samples to process")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct",
                       help="Path to pretrained model")
    return parser.parse_args()

try:
    # 导入本地修改后的模型实现
    from models.modeling_llama_opt import LlamaForCausalLM
    print("成功加载本地修改的Llama OPT模型代码！")
except ImportError as e:
    print(f"导入错误：{e}")
    print("请确认：")
    print("1. 当前目录存在 models/modeling_llama_opt.py")
    print("2. 类名保持为 LlamaForCausalLM")
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
def start_model(model_path, threshold, max_kv_size, eviction_mode, num_trained_encoders, num_encoders, target_layer):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    # 设置OPT相关参数
    if hasattr(model, 'set_opt_args'):
        model.set_opt_args(
            threshold=threshold,
            max_kv_size=max_kv_size,
            eviction_mode=eviction_mode,
            num_trained_encoders=num_trained_encoders,
            num_encoders=num_encoders,
            target_layer=target_layer
        )
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

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

# 推理函数
def run_inference_ttft(input_ids, model, tokenizer):
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start = time.time()
    outputs = model.generate(
        input_ids,
        max_new_tokens=1,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    
    ttft = time.time() - start
    
    return ttft

def run_inference_f1(input_ids, model, tokenizer):
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start = time.time()
    outputs = model.generate(
        input_ids,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    
    response = outputs[0][input_ids.shape[-1]:]  # 切片获取新生成的tokens
    print("🤖 Generated Answer:")
    print(tokenizer.decode(response, skip_special_tokens=True))
    
    total_latency = time.time() - start
    
    return total_latency, tokenizer.decode(response, skip_special_tokens=True)

def main():
    args = parse_args()  # 解析命令行参数
    
    # 使用参数值替换原硬编码
    threshold = args.threshold
    max_kv_size = args.max_kv_size
    eviction_mode = args.eviction_mode
    num_trained_encoders = args.num_trained_encoders
    num_encoders = args.num_encoders
    target_layer = args.target_layer
    fromdataset = args.fromdataset
    max_count = args.max_count
    model_path = args.model_path

    # 加载模型时传递参数
    model, tokenizer = start_model(
        model_path, threshold, max_kv_size, eviction_mode,
        num_trained_encoders, num_encoders, target_layer
    )
    
    time.sleep(5)
    ds = load_datasets(args.fromdataset)

    # 存储结果
    ttft_blend = []
    ttft_full = []
    excutetime_blend = []
    excutetime_full = []
    ppllist_1 = []
    ppllist_2 = []
    f1_full = []
    
    prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
    query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
    messages = [
        {"role": "system", "content": "You are an assistant who provides precise and direct answers."},
        {"role": "user", "content": "In the sentence 'A boy is playing football', what is the exact action activity described? Provide only the exact phrase."},
    ]

    count = 0

    if args.ttft_test:
        # 开始推理ttft
        for id, ex in enumerate(ds["filtered_items"]):
            doc_prompt = ex["text"]
            input_len = tokenizer.encode(doc_prompt)[1:]
            if len(input_len) > 20000:
                doc_prompt = doc_prompt[:9000]
            if len(input_len) < 1000:
                continue
            print(f"input_len: {len(input_len)}")
            count += 1
            print(f"current id:{count}")
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

            ttft = run_inference_ttft(input_ids, model, tokenizer)
            garbage_collection()

            ttft_full.append(ttft)
            
            print(f"TTFTs = {ttft_full}")
            print(f"F1s= {f1_full}")
            print(f"TTFT_with_cache_mean= {np.mean(ttft_full[1::3])}")
            
            print(f"TTFT_with_full_prefill_mean= {np.mean(ttft_full[::3])}")
            print(f"Average_excutetime_blend= {np.mean(excutetime_full[1::3])}")
            print(f"Average_excutetime_full= {np.mean(excutetime_full[::3])}")
            print(f"Average_f1_full= {np.mean(f1_full[::3])}")
            print(f"Average_f1_blend= {np.mean(f1_full[1::3])}")

    if args.f1_test:
        count = 0
        # 开始推理f1
        for id, ex in enumerate(ds):
            count += 1
            print(f"current id:{count}")
            if count > args.max_count:
                break
            
            answers = ex["answer"]
            doc_prompt = ex["context"]
            question_prompt = ex["question"]
            
            input_len = tokenizer.encode(doc_prompt)[1:]
            if len(input_len) > 2000:
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
            
            total_latency, output = run_inference_f1(input_ids, model, tokenizer)
            garbage_collection()
            f1 = max([compute_f1(output, answers, tokenizer)])
            f1_full.append(f1)

            excutetime_full.append(total_latency)
            
            print(f"F1s= {f1_full}")
            print(f"Average_excutetime_blend= {np.mean(excutetime_full[1::2])}")
            print(f"Average_excutetime_full= {np.mean(excutetime_full[::2])}")
            print(f"Average_f1_full= {np.mean(f1_full[::2])}")
            print(f"Average_f1_blend= {np.mean(f1_full[1::2])}")

    # 汇总结果
    print("------------------------------------------------------------------------------------")
    print(f"TTFTs = {ttft_full}")
    print(f"F1s= {f1_full}")
    print(f"TTFT_with_cache_mean= {np.mean(ttft_full[1::2])}")
        
    print(f"TTFT_with_full_prefill_mean= {np.mean(ttft_full[::2])}")
    print(f"Average_excutetime_blend= {np.mean(excutetime_full[1::2])}")
    print(f"Average_excutetime_full= {np.mean(excutetime_full[::2])}")
    print(f"Average_f1_full= {np.mean(f1_full[::2])}")
    print(f"Average_f1_blend= {np.mean(f1_full[1::2])}")
    
    # 停止模型，释放显存
    stop_model(model)
    clean_cache(dir_path)

if __name__ == "__main__":
    main()
