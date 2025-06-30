# 多阶段检索系统股票信息提取功能实现

## 概述

本文档总结了在多阶段检索系统中实现的通用股票信息提取功能，该功能支持多种格式的股票代码和公司名称提取，用于元数据过滤。

## 问题背景

### 原有问题
- 原有的股票代码提取逻辑只支持英文括号格式 `r'\((\d{6})\)'`
- 不支持中文括号 `（）` 和其他格式
- 导致部分查询无法正确提取股票信息，影响元数据过滤效果

### 用户需求
- 支持多种格式的股票代码提取
- 兼容中英文括号、无括号、带交易所后缀等格式
- 在多阶段检索系统的元数据过滤中正常工作

## 解决方案

### 1. 创建通用股票信息提取工具

**文件位置**: `xlm/utils/stock_info_extractor.py`

**核心功能**:
```python
def extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从查询中提取股票代码和公司名称
    
    支持的格式：
    1. 德赛电池(000049) - 英文括号
    2. 德赛电池（000049） - 中文括号
    3. 德赛电池000049 - 无括号
    4. 000049 - 纯数字
    5. 德赛电池(000049.SZ) - 带交易所后缀
    6. 德赛电池（000049.SH） - 中文括号+后缀
    7. 德赛电池000049.SZ - 无括号+后缀
    8. 德赛电池 000049 - 空格分隔
    """
```

### 2. 支持的格式

| 格式类型 | 示例 | 提取结果 |
|---------|------|----------|
| 中文括号 | 德赛电池（000049） | 公司: 德赛电池, 股票: 000049 |
| 英文括号 | 德赛电池(000049) | 公司: 德赛电池, 股票: 000049 |
| 无括号 | 德赛电池000049 | 公司: 德赛电池, 股票: 000049 |
| 纯数字 | 000049 | 公司: None, 股票: 000049 |
| 带后缀 | 德赛电池(000049.SZ) | 公司: 德赛电池, 股票: 000049 |
| 空格分隔 | 德赛电池 000049 | 公司: 德赛电池, 股票: 000049 |

### 3. 提取策略

#### 优先级策略
1. **括号格式优先**: 先尝试匹配带括号的格式（中英文括号）
2. **无括号格式**: 如果没有找到括号格式，尝试无括号格式
3. **公司名补充**: 如果找到股票代码但没有公司名，尝试从上下文提取
4. **代码清理**: 移除交易所后缀，提取纯6位数字代码

#### 正则表达式模式
```python
# 括号格式
bracket_patterns = [
    r'([^，。？\s]+)[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 公司名(股票代码)
    r'[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 纯(股票代码)
]

# 无括号格式
no_bracket_patterns = [
    r'([^，。？\s]+)\s*(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码（支持空格）
    r'([^，。？\s]+)(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码（无空格）
    r'(\d{6}(?:\.(?:SZ|SH))?)',  # 纯股票代码
]
```

## 集成到多阶段检索系统

### 1. 修改多阶段检索UI

**文件**: `xlm/ui/optimized_rag_ui_with_multi_stage.py`

**修改内容**:
```python
# 原有代码
stock_match = re.search(r'\((\d{6})\)', question)
if stock_match:
    stock_code = stock_match.group(1)

# 修改后
from xlm.utils.stock_info_extractor import extract_stock_info
company_name, stock_code = extract_stock_info(question)
```

### 2. 修改传统RAG UI

**文件**: `xlm/ui/optimized_rag_ui.py`

**修改内容**:
```python
# 原有代码
stock_match = re.search(r'\((\d{6})\)', question)
if stock_match:
    stock_code = stock_match.group(1)

# 修改后
from xlm.utils.stock_info_extractor import extract_stock_info
company_name, stock_code = extract_stock_info(question)
```

## 多阶段检索元数据过滤

### 过滤策略

多阶段检索系统使用提取的股票信息进行元数据过滤：

1. **组合过滤**（最高精度）: 公司名称 + 股票代码
2. **单一过滤**: 公司名称 或 股票代码
3. **无过滤**: 返回所有记录

### 过滤逻辑

```python
def pre_filter(self, 
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               max_candidates: int = 1000) -> List[int]:
    
    # 优先使用组合索引（公司名称+股票代码）
    if company_name and stock_code:
        key = f"{company_name_lower}_{stock_code_lower}"
        if key in self.metadata_index['company_stock']:
            return self.metadata_index['company_stock'][key]
    
    # 单一过滤
    elif company_name:
        return self.metadata_index['company_name'].get(company_name_lower, [])
    elif stock_code:
        return self.metadata_index['stock_code'].get(stock_code_lower, [])
    
    # 无过滤
    return list(range(len(self.data)))
```

## 测试验证

### 1. 单元测试

**文件**: `test_stock_code_extraction.py`

测试各种格式的股票代码提取：
- 中英文括号格式
- 无括号格式
- 带交易所后缀格式
- 纯数字格式

### 2. 集成测试

**文件**: `test_multi_stage_stock_extraction.py`

测试提取函数在多阶段检索中的集成：
- 基本提取功能
- 多阶段检索调用模拟
- 元数据过滤逻辑验证

### 3. 测试结果

```
✅ 新的股票信息提取函数支持多种格式
✅ 兼容中英文括号、无括号、带交易所后缀等格式
✅ 可以正确集成到多阶段检索系统的元数据过滤中
✅ 对于不包含股票信息的查询，返回None（正常行为）
```

## 使用示例

### 查询示例

```python
# 中文括号格式
query = "德赛电池（000049）的业绩如何？"
company_name, stock_code = extract_stock_info(query)
# 结果: company_name="德赛电池", stock_code="000049"

# 英文括号格式
query = "德赛电池(000049.SZ)的财务表现？"
company_name, stock_code = extract_stock_info(query)
# 结果: company_name="德赛电池", stock_code="000049"

# 无括号格式
query = "000049的股价走势"
company_name, stock_code = extract_stock_info(query)
# 结果: company_name=None, stock_code="000049"
```

### 多阶段检索调用

```python
# 在多阶段检索系统中使用
results = chinese_retrieval_system.search(
    query=question,
    company_name=company_name,  # 从extract_stock_info获取
    stock_code=stock_code,      # 从extract_stock_info获取
    top_k=20
)
```

## 优势

### 1. 兼容性强
- 支持多种股票代码格式
- 兼容中英文标点符号
- 支持带交易所后缀的格式

### 2. 健壮性好
- 优雅处理不包含股票信息的查询
- 自动清理和标准化股票代码
- 向后兼容原有功能

### 3. 集成简单
- 统一的API接口
- 易于集成到现有系统
- 不影响其他功能

### 4. 可扩展性
- 模块化设计
- 易于添加新的格式支持
- 支持自定义提取规则

## 总结

通过实现通用的股票信息提取功能，我们成功解决了多阶段检索系统中股票代码提取不兼容的问题。新功能支持多种格式，能够正确提取股票信息并用于元数据过滤，提高了检索的准确性和用户体验。

该实现不仅解决了当前问题，还为未来的功能扩展奠定了良好的基础。 