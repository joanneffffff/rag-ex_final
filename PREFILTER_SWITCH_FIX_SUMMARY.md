# 预过滤开关修复总结

## 🔍 问题描述

用户发现了一个重要问题：**预过滤没有开关控制，而reranker有开关**。

### 具体问题：
1. **预过滤默认强制执行** - 在`search`函数中，预过滤总是会执行
2. **reranker有开关** - 可以通过`reranker_checkbox`控制是否使用重排序
3. **baseline模式不纯粹** - baseline模式实际上也使用了预过滤，不是真正的baseline

## 🛠️ 修复方案

### 1. 修改检索系统核心逻辑

**文件：** `alphafin_data_process/multi_stage_retrieval_final.py`

**修改内容：**
- 在`search`函数中添加`use_prefilter: bool = True`参数
- 根据开关决定是否执行预过滤
- 添加详细的日志输出

```python
def search(self, 
           query: str,
           company_name: Optional[str] = None,
           stock_code: Optional[str] = None,
           report_date: Optional[str] = None,
           top_k: int = 10,
           use_prefilter: bool = True) -> Dict:  # 添加预过滤开关参数
```

**核心逻辑：**
```python
# 1. Pre-filtering（根据开关决定是否使用）
if use_prefilter and self.dataset_type == "chinese":
    print("第一步：启用元数据预过滤...")
    candidate_indices = self.pre_filter(company_name, stock_code, report_date)
else:
    print("第一步：跳过元数据预过滤，使用全量检索...")
    candidate_indices = list(range(len(self.data)))
```

### 2. 更新评估脚本

**文件：** `alphafin_data_process/alphafin_retrieval_evaluation.py`

**修改内容：**
- 在`get_ranked_documents_for_evaluation`函数中添加`use_prefilter`参数
- 在`evaluate_mrr_and_hitk`函数中添加`use_prefilter`参数
- 修改baseline模式逻辑，根据开关决定是否使用预过滤
- 添加命令行参数支持

**baseline模式新逻辑：**
```python
if mode == "baseline":
    # baseline: 根据预过滤开关决定是否使用元数据过滤
    if use_prefilter:
        # 使用元数据过滤的baseline
        candidate_indices = retrieval_system.pre_filter(company_name, stock_code, report_date)
    else:
        # 不使用元数据过滤的baseline（真正的baseline）
        candidate_indices = list(range(len(retrieval_system.data)))
```

**命令行参数：**
```python
parser.add_argument('--use_prefilter', action='store_true', default=True, help='是否使用预过滤（默认True）')
parser.add_argument('--no_prefilter', dest='use_prefilter', action='store_false', help='关闭预过滤')
```

## 📊 修复效果

### 修复前的问题：
- ❌ 预过滤无法关闭
- ❌ baseline模式实际上使用了预过滤
- ❌ 无法进行真正的baseline对比实验

### 修复后的功能：
- ✅ 预过滤可以通过开关控制
- ✅ baseline模式可以选择是否使用预过滤
- ✅ 支持真正的baseline对比实验
- ✅ 保持向后兼容性（默认开启预过滤）

## 🧪 测试验证

### 测试脚本：`test_prefilter_switch.py`

**测试内容：**
1. **检索系统预过滤开关测试**
   - 测试开启预过滤的baseline模式
   - 测试关闭预过滤的baseline模式
   - 比较两种模式的结果差异

2. **评估脚本预过滤开关测试**
   - 测试评估函数的预过滤开关功能
   - 验证命令行参数的正确性

### 使用方法：

**检索系统：**
```python
# 开启预过滤（默认）
result = retrieval_system.search(query, use_prefilter=True)

# 关闭预过滤
result = retrieval_system.search(query, use_prefilter=False)
```

**评估脚本：**
```bash
# 开启预过滤（默认）
python alphafin_retrieval_evaluation.py --mode baseline

# 关闭预过滤
python alphafin_retrieval_evaluation.py --mode baseline --no_prefilter
```

## 🔄 向后兼容性

- **默认行为保持不变** - `use_prefilter=True`是默认值
- **现有代码无需修改** - 不传递参数时使用默认值
- **新功能可选使用** - 需要时才传递`use_prefilter=False`

## 📋 使用建议

### 1. 实验对比
```python
# 真正的baseline（无预过滤）
baseline_results = retrieval_system.search(query, use_prefilter=False)

# 带预过滤的baseline
prefilter_results = retrieval_system.search(query, use_prefilter=True)

# 带重排序的完整流程
reranker_results = retrieval_system.search(query, use_prefilter=True)
```

### 2. 评估实验
```bash
# 真正的baseline评估
python alphafin_retrieval_evaluation.py --mode baseline --no_prefilter

# 带预过滤的baseline评估
python alphafin_retrieval_evaluation.py --mode baseline

# 完整流程评估
python alphafin_retrieval_evaluation.py --mode reranker
```

## 🎯 总结

这次修复解决了预过滤控制的核心问题：

1. **提供了预过滤开关** - 可以控制是否使用元数据过滤
2. **实现了真正的baseline** - baseline模式可以选择不使用预过滤
3. **保持了向后兼容** - 现有代码无需修改
4. **支持完整实验对比** - 可以进行baseline vs prefilter vs reranker的完整对比

现在系统支持三种真正的检索模式：
- **Baseline（无预过滤）**：纯FAISS检索
- **Prefilter**：元数据过滤 + FAISS检索
- **Reranker**：元数据过滤 + FAISS检索 + 重排序 