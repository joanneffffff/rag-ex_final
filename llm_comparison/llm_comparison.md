# LLM模型对比评估脚本

本项目包含两个独立的LLM模型评估脚本，分别用于英文和中文金融数据集的模型性能对比。

## 📋 概述

- **英文版本**: 专门用于TatQA英文数据集评估
- **中文版本**: 专门用于AlphaFin中文数据集评估
- **支持模型**: Fin-R1 和 Qwen3-8B
- **评估指标**: F1分数、精确匹配率、生成时间

## 🚀 快速开始

### 环境要求

```bash
# 安装依赖
pip install torch transformers tqdm numpy
```

### 基本使用

```bash
# 英文TatQA数据集评估
python llm_comparison/english_llm_evaluation.py

# 中文AlphaFin数据集评估
python llm_comparison/chinese_llm_evaluation.py
```

## 📁 文件结构

```
llm_comparison/
├── README.md                           # 本文档
├── english_llm_evaluation.py           # 英文TatQA评估脚本
├── chinese_llm_evaluation.py           # 中文AlphaFin评估脚本
└── results/                            # 评估结果输出目录
    ├── tatqa_comparison_results_*.json # TatQA评估结果
    └── comparison_results_chinese_*.json # AlphaFin评估结果
```

## 🔧 脚本详情

### 英文评估脚本 (`english_llm_evaluation.py`)

**用途**: 评估Fin-R1和Qwen3-8B在TatQA英文数据集上的表现

**特性**:
- 基于`comprehensive_evaluation_enhanced.py`的逻辑
- 智能答案提取和救援机制
- 混合决策算法选择最佳模板
- 支持doc_id字段追踪

**数据集**: `evaluate_mrr/tatqa_eval_enhanced.jsonl`

**模板系统**:
- `template_for_table_answer.txt` - 表格问题
- `template_for_text_answer.txt` - 文本问题
- `template_for_hybrid_answer.txt` - 混合问题

**命令行参数**:
```bash
python english_llm_evaluation.py \
  --data_path evaluate_mrr/tatqa_eval_enhanced.jsonl \
  --sample_size 500 \
  --max_new_tokens 150 \
  --do_sample False \
  --repetition_penalty 1.1
```

### 中文评估脚本 (`chinese_llm_evaluation.py`)

**用途**: 评估Fin-R1和Qwen3-8B在AlphaFin中文数据集上的表现

**特性**:
- 专门的中文后处理逻辑
- 公司名称翻译修正
- 中文模板支持
- 内存优化管理

**数据集**: `evaluate_mrr/alphafin_eval.jsonl`

**模板系统**:
- `multi_stage_chinese_template.txt` - 多阶段中文模板

**命令行参数**:
```bash
python chinese_llm_evaluation.py \
  --data_path evaluate_mrr/alphafin_eval.jsonl \
  --sample_size 500 \
  --max_new_tokens 150 \
  --do_sample False \
  --repetition_penalty 1.1
```

## 📊 评估指标

### F1分数
- 基于token级别的精确度和召回率
- 支持中英文文本归一化
- 自动处理数字格式、百分号等

### 精确匹配率 (Exact Match)
- 完全匹配检测
- 忽略大小写和标点符号
- 标准化数字格式

### 生成时间
- 每个样本的生成耗时
- 平均生成时间统计
- 性能监控

## 🧠 核心算法

### 智能答案提取
```python
def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型输出中智能提取最终答案
    1. 优先从<answer>标签提取
    2. 救援逻辑从<think>标签提取
    3. 回退到原始输出最后一行
    """
```

### 混合决策算法
```python
def hybrid_decision_enhanced(context: str, query: str) -> Dict[str, Any]:
    """
    根据上下文类型和查询特征选择最佳模板
    - 上下文类型权重: 40%
    - 查询特征权重: 40%
    - 内容比例权重: 20%
    """
```

### 上下文类型判断
- **Table**: 包含表格特征（分隔符、行列标识等）
- **Text**: 包含文本特征（段落、章节标识等）
- **Mixed**: 混合内容

## 📈 输出结果

### 结果文件格式
```json
{
  "model": "Fin-R1",
  "sample_id": 0,
  "doc_id": "raw_doc_154854",
  "query": "在报告期内，该公司的投资净收益是多少？",
  "expected_answer": "在报告期内，该公司的投资净收益为516971292.0。",
  "raw_generated_text": "<think>...</think><answer>516971292.0</answer>",
  "final_answer": "516971292.0",
  "f1_score": 0.85,
  "exact_match": 1.0,
  "generation_time": 2.34
}
```

### 汇总统计
```
--- Fin-R1 评估总结 ---
总样本数: 500
平均 F1-score: 0.8234
平均 Exact Match: 0.7567
平均生成时间: 2.45 秒/样本
--------------------
```

## 🔍 模型配置

### Fin-R1模型
- **路径**: `/users/sgjfei3/data/huggingface/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa`
- **量化**: 8bit
- **设备**: 自动检测CUDA

### Qwen3-8B模型
- **路径**: `Qwen/Qwen2.5-7B-Instruct`
- **量化**: 8bit
- **设备**: 自动检测CUDA

## 🛠️ 高级配置

### 自定义模板
1. 在`data/prompt_templates/`目录下创建新模板
2. 使用`===SYSTEM===`、`===USER===`、`===ASSISTANT===`分隔符
3. 在脚本中指定模板文件名

### 批量评估
```bash
# 后台运行多个评估任务
nohup python llm_comparison/english_llm_evaluation.py --sample_size 1000 > tatqa_eval.log 2>&1 &
nohup python llm_comparison/chinese_llm_evaluation.py --sample_size 1000 > alphafin_eval.log 2>&1 &
```

### 结果分析
```python
import json

# 加载评估结果
with open('tatqa_comparison_results_*.json', 'r') as f:
    results = json.load(f)

# 按模型分组分析
models = {}
for result in results:
    model = result['model']
    if model not in models:
        models[model] = []
    models[model].append(result)

# 计算统计信息
for model, data in models.items():
    avg_f1 = sum(r['f1_score'] for r in data) / len(data)
    avg_em = sum(r['exact_match'] for r in data) / len(data)
    print(f"{model}: F1={avg_f1:.4f}, EM={avg_em:.4f}")
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查CUDA可用性
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 检查模型路径
   ls /users/sgjfei3/data/huggingface/models--SUFE-AIFLM-Lab--Fin-R1/
   ```

2. **内存不足**
   ```bash
   # 减少batch_size或使用更小的模型
   python llm_comparison/english_llm_evaluation.py --sample_size 100
   ```

3. **模板文件缺失**
   ```bash
   # 检查模板文件
   ls data/prompt_templates/
   ls data/prompt_templates/chinese/
   ```

### 日志分析
```bash
# 查看详细日志
tail -f tatqa_eval.log

# 搜索错误信息
grep "ERROR\|Exception" tatqa_eval.log
```

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 支持Fin-R1和Qwen3-8B模型
- 实现智能答案提取算法
- 添加混合决策模板选择
- 支持TatQA和AlphaFin数据集

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 项目讨论区

---

**注意**: 请确保在运行脚本前已正确配置模型路径和环境变量。 