import logging
from collections import defaultdict

import torch
import random
import numpy as np
import os


def filter_and_keep_first_duplicate_j(coordinates):
    seen_j = set()  
    result = []  

    for coord in coordinates:
        i, j = coord
        if j not in seen_j:
            result.append(coord)  
            seen_j.add(j)  

    return result


def filter_and_keep_random_duplicate_j(coordinates):
    j_to_coords = defaultdict(list)  


    for coord in coordinates:
        i, j = coord
        j_to_coords[j].append(coord)

    result = []


    for j, coords in j_to_coords.items():
        result.append(random.choice(coords))  

    return result



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def set_logger(args):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    file_handler = logging.FileHandler(args.output_dir + "logs.log")
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(args.model_type)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import collections
import string
import re
from rouge_score import rouge_scorer
import os, sys
import json
import subprocess
import signal
import shlex
import time
from dataclasses import dataclass
from transformers import AutoTokenizer
def calculate_ppl(model, tokenizer, text, device='cuda'):
    """
    计算给定文本的困惑度
    :param model: 预训练语言模型
    :param tokenizer: 对应的tokenizer
    :param text: 要评估的文本
    :param device: 计算设备
    :return: 困惑度值
    """
    # 准备输入
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    # 计算负对数似然
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
        

    
    # 计算困惑度
    ppl = torch.exp(neg_log_likelihood).item()
    return ppl

def calculate_ppl_long_text(model, tokenizer, text, stride=512, max_length=1024, device='cuda'):
    encodings = tokenizer(text, return_tensors='pt')
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # todo:可能与前一个窗口有重叠
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
# 示例使用
# model_name = "gpt2"  # 可以替换为其他模型如"bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

# text = "This is a sample text to calculate perplexity."
# ppl = calculate_ppl(model, tokenizer, text)
# print(f"Perplexity: {ppl:.2f}")



def load_datasets(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.startswith("No") or s.startswith("no"):
        s = "No"
    return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_qa_prompt(example, query_prompt):

    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_standard_qa_prompt(example, query_prompt):

    q = normalize_question(example["question"])
    doc_prompts = [f"{paragraph['paragraph_text']}\n\n" for paragraph in example["paragraphs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_sharegpt_prompt(example, query_prompt):
    # q = normalize_question(example["question"])
    doc_prompts = [f"{paragraph['paragraph_text']}\n\n" for paragraph in example["paragraphs"]]
    doc_prompts = example['conversations'][0]['value']
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

@dataclass
class ProcessHandle:
    process: subprocess.Popen
    stdout_file: object
    stderr_file: object
    stdout_filename: str = None
    stderr_filename: str = None

    def kill_and_close(self, force_kill_after=60):
        """
        Kill the process by sending the SIGINT signal, then close the redirected stderr/stdout files
        """
        if self.is_alive():
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

        if self.stderr_file is not None:
            self.stderr_file.close()
        if self.stdout_file is not None:
            self.stdout_file.close()

        if self.stdout_filename is not None:
            os.remove(self.stdout_filename)
        if self.stderr_filename is not None:
            os.remove(self.stderr_filename)

        countdown = force_kill_after
        while self.is_alive() and countdown > 0:
            time.sleep(1)
            countdown -= 1

        # Force kill the process if it's still alive
        if self.is_alive():
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

    def is_alive(self):
        return self.process.poll() is None

def run_command(command, outfile=None, errfile=None, detach=False, **kwargs):
    """
    Input:
        command: a single string of the shell command
        outfile: redirect output to this file if it's not None
        errfile: redirect stderr to this file if it's not None
        detach: if True, it will start a subprocess and return the handle of that process
                without blocking the caller
                if False, it will block the caller until the subprocess finished. And it
                will return a boolean indicating whether the process successfully finishes
        kwargs: the dictionary of extra environment variables
    Returns:
        If `detach` is False:
            returns (flag, stdout string)
            flag will be True if the process finished without any error
            returns False otherwise
        If `detach` is True:
            returns the handle to the background process (ProcessHandle project)
    Note:
        If outfile and errfile are None, it will be defaulted to print to stdout
    """
    env = os.environ.copy()
    env.update(kwargs)


    out = open(outfile, "w") if outfile is not None else None
    err = open(errfile, "w") if errfile is not None else None

    args = shlex.split(command)

    process = subprocess.Popen(args, stdout=out, stderr=err, env=env, preexec_fn=os.setsid)


    if not detach:
        process.communicate()
        if out is not None:
            out.close()
        if err is not None:
            err.close()
        return process.returncode == 0, process.stdout
    else:
        return ProcessHandle(process, out, err, outfile, errfile)


def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO: do not hard-code tokenizer
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Meta-Llama-3.1-8B-Instruct")

    return len(estimate_num_tokens.tokenizer.tokenize(text))

def read_gpu_memory():
    """
    Read the GPU memory usage by using nvidia-smi command
    """
    command = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    return json.dumps(
            {f"gpu-{id}":int(x) for id, x in enumerate(result.stdout.decode("utf-8").strip().split("\n"))})

# def get_max_context_length(model: str) -> int:
#     match model:
#         case "/data/models/Mistral-7B-Instruct-v0.2":
#             return 17168
#         case "THUDM/glm-4-9b-chat":
#             return 32768
#         case "/data/models/Meta-Llama-3.1-8B-Instruct":
#             return 17168
#         case _:
#             return 32768