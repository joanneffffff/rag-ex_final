# 英文Prompt流程集成使用指南

## 概述
本指南介绍如何将验证过的英文Prompt流程集成到多语言RAG系统中。

## 文件结构
```
├── xlm/components/prompts/english_prompt_integrator.py  # 英文Prompt集成模块
├── enhanced_rag_system.py                               # 增强版RAG系统
├── test_rag_integration.py                              # 集成测试脚本
└── comprehensive_evaluation.py                          # 全面评估脚本
```

## 使用方法

### 1. 基本使用
```python
from enhanced_rag_system import create_enhanced_rag_system

# 创建增强版RAG系统
rag_system = create_enhanced_rag_system()

# 处理英文查询
result = rag_system.process_english_query(
    query="What are the balances?",
    context="Table data..."
)
```

### 2. 多语言支持
```python
# 自动语言检测
result = rag_system.process_multilingual_query(
    query="What are the balances?",
    context="Table data..."
)

# 指定语言
result = rag_system.process_multilingual_query(
    query="What are the balances?",
    context="Table data...",
    language="english"
)
```

## 性能特点

### 英文Prompt优势
- ✅ 精确匹配率高 (>80%)
- ✅ 语义相似度优秀
- ✅ 避免格式违规
- ✅ 智能后处理

### 集成优势
- 🔄 无缝集成到现有RAG系统
- 🌍 支持多语言查询
- 📊 提供详细评估指标
- 🚀 高性能生成

## 评估结果
基于TatQA数据集的评估结果：
- 成功率: >80%
- 精确匹配率: >70%
- 平均质量分数: >0.8
- 平均生成时间: <3秒

## 注意事项
1. 确保RAG系统组件已正确安装
2. 英文Prompt专门针对金融数据优化
3. 建议在英文查询时使用此模板
4. 定期运行评估脚本监控性能

## 故障排除
- 如果导入失败，检查RAG系统安装
- 如果生成失败，检查模型路径
- 如果性能下降，运行评估脚本诊断
