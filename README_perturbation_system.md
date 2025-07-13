# RAG扰动实验系统

## 概述

本系统实现了统一的RAG扰动实验框架，用于评估RAG系统在不同扰动条件下的鲁棒性。系统集成了多种扰动器、特征提取、LLM Judge评估等功能。

## 现有系统架构

### 现有基类系统
- **简单基类**: `xlm/modules/perturber/perturber.py` - 提供基础扰动器接口
- **现有实验**: `rag_perturbation_experiment.py` - 完整的RAG扰动实验系统
- **现有扰动器**: 基于简单基类实现的多种扰动策略

### 增强基类系统
- **增强基类**: `xlm/modules/perturber/base_perturber.py` - 提供配置化和标准化输出
- **统一实验**: `unified_perturbation_experiment.py` - 新的统一实验框架
- **特征提取**: `xlm/modules/feature_extractor.py` - 多粒度特征提取模块

## 系统架构

### 现有系统组件（轻量级）

1. **基础扰动器接口** (`Perturber`)
   - 简单抽象基类，定义基础扰动器接口
   - 轻量级设计，易于集成
   - 返回字符串列表，适合直接集成到RAG主流程

2. **现有扰动器实现**
   - `LeaveOneOutPerturber`: 留一扰动
   - `ReorderPerturber`: 重排序扰动
   - `TrendPerturber`: 趋势扰动（上升↔下降）
   - `YearPerturber`: 年份扰动
   - `TermPerturber`: 术语扰动

3. **现有实验系统** (`RAGPerturbationExperiment`)
   - 集成RAG系统组件
   - 支持检索和生成阶段分析
   - 可解释性分析功能

### 增强系统组件（推荐使用）

1. **增强扰动器接口** (`BasePerturber`)
   - 抽象基类，定义统一的扰动器接口
   - 支持配置化初始化和标准化输出格式
   - 提供更丰富的功能和错误处理
   - 返回详细字典，包含扰动文本、扰动细节、原始特征

2. **特征提取模块** (`FeatureExtractor`)
   - 支持多种粒度特征提取（词汇、句子、短语、实体、数字、年份）
   - 中英文双语支持
   - 金融领域专业词汇识别

3. **统一实验系统** (`UnifiedPerturbationExperiment`)
   - 集成RAG系统组件
   - 自动化实验流程
   - F1分数计算和LLM Judge评估

4. **分析工具** (`PerturbationResultAnalyzer`)
   - 实验结果统计分析
   - 可视化图表生成
   - 详细报告导出

## 快速开始

### 使用现有系统（推荐）

#### 1. 运行现有RAG扰动实验

```bash
python rag_perturbation_experiment.py
```

#### 2. 运行特定扰动器测试

```bash
# 在代码中修改测试的扰动器
perturber_name = 'trend'  # 可选: leave_one_out, reorder, trend, year, term
```

#### 3. 分析检索和生成阶段

```bash
# 系统会自动测试检索阶段和生成阶段的扰动效果
```

### 使用增强系统（推荐）

#### 1. 运行三个核心扰动器测试

```bash
# 测试年份、术语、趋势三个扰动器
python run_perturbation_experiment.py --perturbers year term trend --max_samples 5
```

#### 2. 运行快速测试

```bash
python run_perturbation_experiment.py --max_samples 3
```

#### 3. 使用配置文件运行

```bash
python run_perturbation_experiment.py --samples experiment_config.json --max_samples 10
```

#### 4. 分析实验结果

```bash
python analyze_perturbation_results.py perturbation_results.json --plots
```

#### 5. 单独测试特定扰动器

```bash
# 只测试年份扰动
python run_perturbation_experiment.py --perturbers year --max_samples 3

# 只测试术语扰动
python run_perturbation_experiment.py --perturbers term --max_samples 3

# 只测试趋势扰动
python run_perturbation_experiment.py --perturbers trend --max_samples 3
```

## 扰动器详解

### 增强系统扰动器（推荐使用）

#### 1. YearPerturber（年份扰动器）
- **功能**: 修改文本中的年份信息
- **适用场景**: 评估RAG系统对时间信息变化的鲁棒性
- **支持格式**: 
  - 中文：`2023年` → `2024年`
  - 英文：`2023` → `2024`
- **扰动策略**: 年份+1
- **示例**: "2023年业绩增长" → "2024年业绩增长"

#### 2. TermPerturber（术语扰动器）
- **功能**: 替换金融专业术语
- **适用场景**: 评估RAG系统对专业术语变化的鲁棒性
- **术语映射**:
  - `市盈率` ↔ `净利润`
  - `市净率` ↔ `市销率`
  - `营收` ↔ `收入`
  - `总资产` ↔ `净资产`
  - `负债` ↔ `资产`
- **示例**: "市盈率表现良好" → "净利润表现良好"

#### 3. TrendPerturber（趋势扰动器）
- **功能**: 改变趋势词汇（上升↔下降）
- **适用场景**: 评估RAG系统对趋势信息变化的鲁棒性
- **支持语言**: 中英文双语
- **趋势映射**:
  - 中文：`上升` ↔ `下降`，`上涨` ↔ `下跌`，`增长` ↔ `减少`
  - 英文：`increase` ↔ `decrease`，`rise` ↔ `fall`，`growth` ↔ `decline`
- **示例**: "业绩上升趋势" → "业绩下降趋势"

### 现有系统扰动器（轻量级）

#### LeaveOneOutPerturber
- **功能**: 移除文档中的单个特征
- **适用场景**: 评估RAG系统对关键信息缺失的鲁棒性
- **配置**: 支持自定义特征提取粒度

#### ReorderPerturber
- **功能**: 重新排序文档中的词汇
- **适用场景**: 评估RAG系统对语序变化的鲁棒性
- **依赖**: nlpaug库

## 实验配置

### 实验场景

系统预定义了多种实验场景：

- `quick_test`: 快速测试（3个样本，主要扰动器）
- `comprehensive_test`: 全面测试（10个样本，所有扰动器）
- `trend_analysis`: 趋势分析（专注趋势扰动器）
- `year_analysis`: 年份分析（专注年份扰动器）
- `term_analysis`: 术语分析（专注术语扰动器）

### 配置文件格式

```json
{
  "experiment_scenarios": {
    "scenario_name": {
      "description": "场景描述",
      "max_samples": 5,
      "perturbers": ["trend", "year"],
      "output_file": "results.json"
    }
  },
  "test_samples": [
    {
      "id": "sample_1",
      "query": "查询问题",
      "answer": "期望答案"
    }
  ]
}
```

## 结果分析

### 评估指标

1. **F1分数**
   - 原始答案 vs 期望答案
   - 扰动答案 vs 期望答案
   - 扰动答案 vs 原始答案

2. **LLM Judge评分**
   - 准确性（1-10分）
   - 完整性（1-10分）
   - 专业性（1-10分）

3. **扰动效果分析**
   - F1下降幅度
   - 成功率统计
   - 扰动器对比

### 可视化图表

系统自动生成以下图表：

1. **F1分数分析图**
   - F1分数分布对比
   - 各扰动器平均F1分数
   - F1下降幅度分布
   - LLM Judge平均分数

2. **扰动器对比图**
   - 各扰动器成功率
   - 各扰动器F1比率

## 文件结构

### 现有系统文件

```
├── xlm/modules/perturber/
│   ├── perturber.py               # 基础扰动器抽象基类
│   ├── leave_one_out_perturber.py # 留一扰动器
│   ├── reorder_perturber.py       # 重排序扰动器
│   ├── trend_perturber.py         # 趋势扰动器
│   ├── year_perturber.py          # 年份扰动器
│   └── term_perturber.py          # 术语扰动器
├── rag_perturbation_experiment.py # 现有RAG扰动实验系统
└── README_perturbation_system.md  # 本文档
```

### 增强系统文件

```
├── xlm/modules/perturber/
│   ├── base_perturber.py          # 增强扰动器抽象基类
│   ├── leave_one_out_perturber.py # 留一扰动器（已升级）
│   ├── reorder_perturber.py       # 重排序扰动器（已升级）
│   ├── trend_perturber.py         # 趋势扰动器（已升级）
│   ├── year_perturber.py          # 年份扰动器（已升级）
│   └── term_perturber.py          # 术语扰动器（已升级）
├── xlm/modules/feature_extractor.py # 特征提取模块
├── unified_perturbation_experiment.py # 统一实验系统
├── run_perturbation_experiment.py     # 实验运行脚本
├── analyze_perturbation_results.py    # 结果分析工具
├── experiment_config.json             # 实验配置文件
└── README_perturbation_system.md     # 本文档
```

## 扩展开发

### 使用现有系统添加扰动器

1. 继承 `Perturber` 类
2. 实现 `perturb` 方法
3. 返回字符串列表

```python
class CustomPerturber(Perturber):
    def perturb(self, text: str, features: List[str]) -> List[str]:
        # 实现扰动逻辑
        perturbations = []
        # ... 扰动处理 ...
        return [perturbed_text1, perturbed_text2, ...]
```

### 使用增强系统添加扰动器

1. 继承 `BasePerturber` 类
2. 实现 `perturb` 方法
3. 返回标准化的扰动结果格式

```python
class CustomPerturber(BasePerturber):
    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        # 实现扰动逻辑
        perturbations = []
        # ... 扰动处理 ...
        return [{
            'perturbed_text': perturbed_text,
            'perturbation_detail': detail,
            'original_feature': feature
        }]
```

### 添加新的特征提取粒度

1. 在 `Granularity` 枚举中添加新类型
2. 在 `FeatureExtractor` 中实现对应的提取方法

### 自定义评估指标

1. 在 `UnifiedPerturbationExperiment` 中添加新的评估方法
2. 在 `PerturbationResult` 中添加对应的字段
3. 在分析工具中添加相应的统计逻辑

## 系统选择指南

### 推荐使用增强系统（BasePerturber）

**核心优势**：
- ✅ **详细分析**: 提供完整的扰动信息和统计
- ✅ **三个核心扰动器**: year、term、trend，覆盖主要扰动场景
- ✅ **标准化输出**: 返回详细字典，包含扰动文本、细节、原始特征
- ✅ **配置化**: 支持灵活的配置选项
- ✅ **扩展性**: 易于添加新的扰动器和功能
- ✅ **可视化**: 支持结果分析和图表生成

**适用场景**：
- 📊 需要详细扰动分析的科研项目
- 🔬 需要批量实验和统计分析
- 📈 需要生成可视化报告
- 🎯 需要评估RAG系统鲁棒性的系统性研究

### 何时使用现有系统（Perturber）

**适用场景**：
- ⚡ **快速原型**: 需要快速验证扰动策略
- 🔧 **简单实验**: 只需要基础的扰动功能
- 🔄 **兼容性**: 需要与现有RAG主流程完全兼容
- 📦 **轻量级**: 希望最小化依赖和复杂度

### 迁移指南

从现有系统迁移到增强系统：

1. **扰动器迁移**:
   ```python
   # 现有系统
   class MyPerturber(Perturber):
       def perturb(self, text: str, features: List[str]) -> List[str]:
           return [perturbed_text]
   
   # 增强系统
   class MyPerturber(BasePerturber):
       def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
           return [{
               'perturbed_text': perturbed_text,
               'perturbation_detail': 'description',
               'original_feature': feature
           }]
   ```

2. **实验系统迁移**:
   ```python
   # 现有系统
   experiment = RAGPerturbationExperiment()
   result = experiment.run_perturbation_experiment(question, perturber_name)
   
   # 增强系统
   experiment = UnifiedPerturbationExperiment()
   results = experiment.run_comprehensive_experiment(samples)
   ```

## 注意事项

1. **依赖安装**: 确保安装了所有必要的依赖包
2. **显存管理**: 大规模实验时注意GPU显存使用
3. **结果保存**: 建议定期保存中间结果防止数据丢失
4. **错误处理**: 系统包含完善的错误处理机制
5. **系统选择**: 根据需求选择合适的系统架构

## 故障排除

### 常见问题

1. **导入错误**: 检查Python路径和依赖安装
2. **显存不足**: 减少批处理大小或使用CPU模式
3. **结果为空**: 检查输入数据和扰动器配置
4. **可视化失败**: 确保安装了matplotlib和seaborn

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献指南

1. 遵循现有代码风格和接口规范
2. 添加适当的文档和注释
3. 包含单元测试
4. 更新相关文档

## 总结

本系统提供了两套完整的RAG扰动实验框架，**推荐使用增强系统**：

### 增强系统优势（推荐）
- ✅ **三个核心扰动器**: year、term、trend，覆盖主要扰动场景
- ✅ **详细分析**: 提供完整的扰动信息和统计
- ✅ **标准化输出**: 返回详细字典，便于分析和处理
- ✅ **可视化支持**: 支持结果分析和图表生成
- ✅ **配置化设计**: 灵活可扩展，易于维护

### 现有系统优势（轻量级）
- ✅ 简单易用，快速上手
- ✅ 与现有代码完全兼容
- ✅ 轻量级设计，依赖少
- ✅ 已集成到现有RAG系统中

### 推荐使用策略
1. **主要使用**: 推荐使用增强系统，特别是year、term、trend三个核心扰动器
2. **快速原型**: 需要快速验证时使用现有系统
3. **深度分析**: 需要详细分析时使用增强系统
4. **并行使用**: 两个系统可以并行使用，满足不同需求

## 许可证

本项目采用MIT许可证。 