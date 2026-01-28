import argparse  # 新增模块
import gc
import torch
import numpy as np
import time
import sys
import os
from transformers import AutoTokenizer
from pathlib import Path
from utils import load_datasets,calculate_ppl_long_text,calculate_ppl,compute_f1

import shutil
# 在main函数前定义解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttft_test", type=bool, default=False, help="Enable TTFT latency test")
    parser.add_argument("--f1_test", type=bool, default=True, help="Enable F1 score test")
    parser.add_argument("--recompute_selection_rate", type=float, default=0.05, 
                       help="Recompute selection rate (0-1)")
    parser.add_argument("--recompute_wdsw", type=int, default=1, 
                       help="Recompute window size")
    parser.add_argument("--fromdataset", type=str, 
                       default='/home/homie/homie/fuzzy_llama_submit/datasets/wiki_for_test.json',
                       help="Path to dataset JSON file")
    parser.add_argument("--max_count", type=int, default=100, 
                       help="Maximum number of samples to process")
    return parser.parse_args()
try:
    # 导入本地修改后的模型实现
    # from models.modeling_llama import LlamaForCausalLM
    # from models.modeling_llama_exact_reuse import LlamaForCausalLM
    # from models.modeling_llama_fuzzy_reuse import LlamaForCausalLM
    from models.modeling_llama_fuzzy_recompute_random import LlamaForCausalLM
    print("成功加载本地修改的Llama模型代码！")
except ImportError as e:
    print(f"导入错误：{e}")
    print("请确认：")
    print("1. 当前目录存在 modeling_llama.py")
    print("2. 类名保持为 LlamaForCausalLM")
    sys.exit(1)
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 配置参数
model_path = "/mnt/sdb/homie/models/LLM-Research/Meta-Llama-3-8B-Instruct"  # 预训练模型路径（如果需要）
# Llama-3.1-8B
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
dir_path = Path("/mnt/sda1/homie_cache/")

# 清理显存
def garbage_collection():
    gc.collect()
    torch.cuda.empty_cache()


# 加载模型和分词器
def start_model(model_path, recompute_selection_rate, recompute_wdsw):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    model.set_recompute_args(recompute_selection_rate, recompute_wdsw)  # 传递参数
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
    # first_token = model.generate(input_ids, max_new_tokens=1, temperature=1e-6)
    # start_full = time.time()
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
    
    # first_token = model.generate(input_ids, max_new_tokens=1, temperature=1e-6)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start = time.time()
    # first_token = model.generate(input_ids, max_new_tokens=1, temperature=1e-6)
    # start_full = time.time()
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
# 初始化 Accelerator
# accelerator = Accelerator()
def main():
    args = parse_args()  # 解析命令行参数
    
    # 使用参数值替换原硬编码
    recompute_selection_rate = args.recompute_selection_rate
    recompute_wdsw = args.recompute_wdsw
    fromdataset = args.fromdataset
    max_count = args.max_count

    # 加载模型时传递参数
    model, tokenizer = start_model(model_path, recompute_selection_rate, recompute_wdsw)
    
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
            # answers = ex["conversations"][1]["value"]
            # doc_prompt = ex["conversations"][0]["value"]
            # question_prompt = ex["conversations"][1]["question"]
            # answers = ex["answer"]
            
            # question_prompt = ex["question"]
            
            
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
            # input_prompt = input_prompt[:-1000]
            

            # ttft, total_latency,output = run_inference(input_prompt, model, tokenizer)
            # print(f"Normal generation: {output}")
            # garbage_collection()
            # f1 = max([compute_f1(output, answer, tokenizer) for answer in answers])
            # f1_full.append(f1)
            # ttft_full.append(ttft)
            # excutetime_full.append(total_latency)

            # ppllist_1.append(calculate_ppl_long_text(model, tokenizer,input_prompt))
            ttft = run_inference_ttft(input_ids, model, tokenizer)
            garbage_collection()

            ttft_full.append(ttft)
            # excutetime_full.append(total_latency)
            # print(f"TTFT with cache: {ttft_blend}")
            
            # print(f"ppl_2: {ppllist_2}")
            print(f"TTFTs = {ttft_full}")
            print(f"F1s= {f1_full}")
            print(f"TTFT_with_cache_mean= {np.mean(ttft_full[1::3])}")
            
            print(f"TTFT_with_full_prefill_mean= {np.mean(ttft_full[::3])}")
            print(f"Average_excutetime_blend= {np.mean(excutetime_full[1::3])}")
            print(f"Average_excutetime_full= {np.mean(excutetime_full[::3])}")
            print(f"Average_f1_full= {np.mean(f1_full[::3])}")
            print(f"Average_f1_blend= {np.mean(f1_full[1::3])}")
        # ppllist_1.append(calculate_ppl(model, tokenizer, input_prompt, device='cuda'))
        # print(f"ppl_1: {ppllist_1}")
    if args.f1_test:
        count = 0
        # 开始推理f1
        for id, ex in enumerate(ds):
            count += 1
            print(f"current id:{count}")
            if count > args.max_count:
                break
            # answers = ex["conversations"][1]["value"]
            # doc_prompt = ex["conversations"][0]["value"]
            # question_prompt = ex["conversations"][1]["question"]
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
            # input_prompt = input_prompt[:-1000]
            # ttft, total_latency,output = run_inference(input_prompt, model, tokenizer)
            # print(f"Normal generation: {output}")
            # garbage_collection()
            # f1 = max([compute_f1(output, answer, tokenizer) for answer in answers])
            # f1_full.append(f1)
            # ttft_full.append(ttft)
            # excutetime_full.append(total_latency)

            # ppllist_1.append(calculate_ppl_long_text(model, tokenizer,input_prompt))
            
            total_latency, output = run_inference_f1(input_ids, model, tokenizer)
            # print(f"Normal generation: {output}")
            garbage_collection()
            f1 = max([compute_f1(output, answers, tokenizer)])
            f1_full.append(f1)

            # ttft_full.append(ttft)
            excutetime_full.append(total_latency)
            # print(f"TTFT with cache: {ttft_blend}")
            # print(f"TTFTs = {ttft_full}")
            print(f"F1s= {f1_full}")
            # print(f"TTFT_with_cache_mean= {np.mean(ttft_full[1::2])}")   
            # print(f"TTFT_with_full_prefill_mean= {np.mean(ttft_full[::2])}")
            print(f"Average_excutetime_blend= {np.mean(excutetime_full[1::2])}")
            print(f"Average_excutetime_full= {np.mean(excutetime_full[::2])}")
            print(f"Average_f1_full= {np.mean(f1_full[::2])}")
            print(f"Average_f1_blend= {np.mean(f1_full[1::2])}")
                # print(f"ppl_2: {ppllist_2}")
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