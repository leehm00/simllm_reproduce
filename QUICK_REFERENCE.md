# 模型脚本快速参考指南

## 快速开始

### LSH 模型
```bash
# 基本 F1 测试
python llama_lsh_args.py

# TTFT 测试
python llama_lsh_args.py --ttft_test True --f1_test False

# 自定义参数
python llama_lsh_args.py --threshold 0.85 --max_kv_size 1024 --max_count 50
```

### OPT 模型
```bash
# 基本 F1 测试
python llama_opt_args.py

# TTFT 测试
python llama_opt_args.py --ttft_test True --f1_test False

# 自定义编码器
python llama_opt_args.py --num_encoders 16 --num_trained_encoders 2
```

## 参数速查表

### 通用参数（所有模型）

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| --ttft_test | - | False | TTFT 测试 |
| --f1_test | - | True | F1 测试 |
| --max_count | - | 100 | 样本数 |
| --fromdataset | - | (默认路径) | 数据集路径 |
| --model_path | - | (默认路径) | 模型路径 |

### LSH 特定参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| --threshold | 0.9 | 0-1 | 相似度阈值 |
| --max_kv_size | 512 | >0 | KV 缓存大小 |
| --eviction_mode | LRU | LRU/FIFO | 驱逐策略 |

### OPT 特定参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| --threshold | 0.9 | 0-1 | 相似度阈值 |
| --max_kv_size | 512 | >0 | KV 缓存大小 |
| --eviction_mode | LRU | LRU/FIFO | 驱逐策略 |
| --num_trained_encoders | 1 | ≥0 | 训练编码器数 |
| --num_encoders | 8 | >0 | 总编码器数 |
| --target_layer | -1 | -1或≥0 | 目标层 |

## 常用命令组合

### 1. 快速测试（少量样本）
```bash
python llama_lsh_args.py --max_count 10
python llama_opt_args.py --max_count 10
```

### 2. 完整评估
```bash
python llama_lsh_args.py --f1_test True --max_count 100
python llama_opt_args.py --f1_test True --max_count 100
```

### 3. 性能测试
```bash
python llama_lsh_args.py --ttft_test True --f1_test False --max_count 50
python llama_opt_args.py --ttft_test True --f1_test False --max_count 50
```

### 4. 高精度配置
```bash
python llama_lsh_args.py --threshold 0.95 --max_kv_size 1024
python llama_opt_args.py --threshold 0.95 --max_kv_size 1024
```

### 5. 低显存配置
```bash
python llama_lsh_args.py --max_kv_size 256 --max_count 20
python llama_opt_args.py --max_kv_size 256 --max_count 20
```

## 输出指标说明

### TTFT (Time To First Token)
- **含义**: 生成第一个 token 的时间
- **单位**: 秒
- **越低越好**

### F1 Score
- **含义**: 生成答案的准确度
- **范围**: 0-1
- **越高越好**

### 执行时间
- **含义**: 完整推理时间
- **单位**: 秒
- **越低越好**

## 故障排除速查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| CUDA OOM | 显存不足 | 减少 max_kv_size 或 max_count |
| 模型加载失败 | 路径错误 | 检查 --model_path |
| 数据集错误 | 格式不对 | 检查 JSON 格式 |
| 速度慢 | CPU 模式 | 确保 CUDA 可用 |

## 性能调优建议

### 提升速度
1. 减少 `--max_count`
2. 使用 GPU
3. 减少 `--max_kv_size`

### 提升准确度
1. 增加 `--threshold`
2. 增加 `--max_kv_size`
3. 调整编码器参数（OPT）

### 节省显存
1. 减少 `--max_kv_size`
2. 使用 float16（已默认）
3. 减少批次大小

## 文件结构

```
SimLLM/
├── llama3_args.py              # 原始参考脚本
├── llama_lsh_args.py           # LSH 模型脚本
├── llama_opt_args.py           # OPT 模型脚本
├── MODEL_SCRIPTS_README.md     # 详细文档
├── QUICK_REFERENCE.md          # 本文件
├── models/
│   ├── modeling_llama_lsh.py   # LSH 模型实现
│   ├── modeling_llama_opt.py   # OPT 模型实现
│   └── ...
└── utils.py                    # 工具函数
```

## 依赖检查

```bash
# 检查 Python 版本
python --version  # 需要 3.8+

# 检查 CUDA
nvidia-smi

# 检查依赖
pip list | grep -E "torch|transformers|numpy"
```

## 联系和支持

- 详细文档: [`MODEL_SCRIPTS_README.md`](MODEL_SCRIPTS_README.md:1)
- 原始参考: [`llama3_args.py`](llama3_args.py:1)
- 模型实现: `models/` 目录
