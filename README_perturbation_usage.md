# RAG系统扰动实验使用说明

## 功能概述

`rag_perturbation_experiment.py` 是一个完整的RAG系统扰动实验工具，支持：

- 从指定数据集选择代表性样本
- 应用多种扰动策略（year、trend、term）
- 计算扰动前后的答案相似度和重要性
- 使用LLM Judge评估答案质量
- 将所有结果保存为JSON格式

## 命令行参数

```bash
python rag_perturbation_experiment.py --dataset_path <数据路径> [选项]
```

### 必需参数

- `--dataset_path`: 样本数据路径（JSONL格式）

### 可选参数

- `--num_samples`: 选择的样本数量（默认: 20）
- `--output_dir`: 结果输出目录（默认: perturbation_results）

## 使用示例

### 基本使用
```bash
python rag_perturbation_experiment.py --dataset_path data/alphafin/alphafin_eval_samples_updated.jsonl
```

### 指定样本数量和输出目录
```bash
python rag_perturbation_experiment.py \
    --dataset_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --num_samples 10 \
    --output_dir my_perturbation_results
```

## 输出文件

实验完成后，会在指定的输出目录中生成以下JSON文件：

### 1. 详细实验结果 (`perturbation_results_YYYYMMDD_HHMMSS.json`)
包含完整的实验数据：
- 实验信息（时间戳、数据集路径、样本数量等）
- 所有扰动结果（原始答案、扰动后答案、相似度分数等）
- 选择的样本信息

### 2. 分析报告 (`perturbation_analysis_YYYYMMDD_HHMMSS.json`)
包含统计分析：
- 各扰动器的统计信息
- 平均相似度、重要性、准确性分数
- 整体指标汇总

### 3. 样本选择结果 (`selected_samples_YYYYMMDD_HHMMSS.json`)
包含样本选择信息：
- 选择标准（多样性、复杂度、可解释性）
- 选中的样本详细信息

### 4. 汇总统计 (`experiment_summary_YYYYMMDD_HHMMSS.json`)
包含实验汇总：
- 实验概览信息
- 各扰动器的统计指标
- 整体性能指标

## 数据格式要求

输入数据集应为JSONL格式，每行包含一个样本：

```json
{
  "summary": "上下文摘要",
  "content": "完整内容",
  "generated_question": "生成的问题",
  "expected_answer": "期望答案"
}
```

## 扰动器说明

实验使用三种核心扰动器：

1. **YearPerturber**: 时间相关扰动
2. **TrendPerturber**: 趋势相关扰动  
3. **TermPerturber**: 术语相关扰动

每个扰动器会同时作用于：
- `summary`（上下文摘要）
- `generated_question`（生成的问题）

## 评估指标

实验计算以下指标：

- **相似度分数**: 基于embedding的答案相似度
- **重要性分数**: 基于F1和EM的答案质量评估
- **LLM Judge评分**: 使用大模型评估答案质量
- **F1分数**: 精确率和召回率的调和平均
- **EM分数**: 完全匹配分数

## 注意事项

1. 确保数据集文件存在且格式正确
2. 实验会使用本地模型，无需下载新模型
3. 输出目录会自动创建
4. 所有结果以JSON格式保存，便于后续分析
5. 时间戳确保多次实验的结果不会覆盖

## 错误处理

- 如果数据集文件不存在，程序会报错并退出
- 如果样本选择失败，实验会终止
- 所有错误信息会显示在控制台输出中 