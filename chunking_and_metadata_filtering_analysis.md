# TatQA和AlphaFin数据处理逻辑分析

## 问题1：TatQA和AlphaFin的Chunking逻辑

### TatQA的Chunking逻辑

TatQA数据集采用**结构化chunking**策略，主要处理两种数据类型：

#### 核心处理函数：
- **`process_tatqa_to_chunks()`** - 主要处理函数
- **`extract_unit_from_paragraph()`** - 提取数值单位信息
- **`table_to_natural_text()`** - 表格转自然语言

#### Chunking策略：

**段落处理：**
```python
# 每个段落作为一个独立的chunk
for p_idx, para in enumerate(paragraphs):
    para_text = para.get("text", "") if isinstance(para, dict) else para
    if para_text.strip():
        all_chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"para_{p_idx}",
            "text": para_text.strip(),
            "source_type": "paragraph",
            "language": "english"
        })
```

**表格处理：**
```python
# 将表格转换为自然语言描述
for t_idx, table in enumerate(tables):
    table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
    if table_text.strip():
        all_chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"table_{t_idx}",
            "text": table_text.strip(),
            "source_type": "table",
            "language": "english"
        })
```

**表格转文本逻辑：**
- 提取表头和数据行
- 按行处理，生成自然语言描述
- 添加单位信息（如"millions USD"）
- 格式化数值（处理货币符号、负数等）

### AlphaFin的Chunking逻辑

AlphaFin数据集采用**多层级chunking**策略，处理中文金融数据：

#### 核心处理函数：
- **`convert_json_context_to_natural_language_chunks()`** - 主要转换函数
- **`process_alphafin_document()`** - 处理单个文档
- **`process_alphafin_to_chunks()`** - 批量处理

#### Chunking策略：

**1. 研报格式处理：**
```python
# 识别研报格式并提取结构化信息
report_match = re.match(
    r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
    cleaned_initial, 
    re.DOTALL
)
```

**2. 字典格式处理：**
```python
# 解析JSON/字典格式的财务数据
for metric_name, time_series_data in parsed_data.items():
    # 按指标名称和时间序列数据生成chunks
    if len(description_parts) <= 3:
        full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
    else:
        # 数据过多时进行摘要
        full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
```

**3. 文档级别处理（优化版本）：**
```python
# 文档级别处理：保持原始格式，避免过度分割
if self.chinese_document_level:
    if len(content) > 8192:  # 8K字符限制
        # 按段落分割长文档
        paragraphs = content.split('\n\n')
        merged_chunks = self._merge_paragraphs_to_chunks(paragraphs, max_length=8192)
    else:
        # 文档长度适中，直接使用原始内容
        alphafin_doc = DocumentWithMetadata(content=content, metadata=doc_metadata)
```

### 主要差异对比

| 特性 | TatQA | AlphaFin |
|------|-------|----------|
| **数据格式** | 结构化JSON（段落+表格） | 混合格式（研报+字典+纯文本） |
| **语言** | 英文 | 中文 |
| **Chunking粒度** | 段落/表格级别 | 文档级别（优化版）或指标级别 |
| **单位处理** | 自动提取数值单位 | 保持原始单位信息 |
| **表格处理** | 转换为自然语言描述 | 保持结构化数据 |
| **长度控制** | 无固定限制 | 8K字符限制，智能分割 |

### 优化策略

**AlphaFin的文档级别优化：**
- 避免过度分割，保持文档完整性
- 智能长度控制（8K字符限制）
- 按段落或句子分割长文档
- 保持原始格式，减少信息丢失

**TatQA的结构化处理：**
- 保持表格和段落的语义完整性
- 自动提取和标准化单位信息
- 生成适合LLM理解的自然语言描述

---

## 问题2：AlphaFin和中文查询的元数据过滤逻辑

### 元数据索引构建

#### 核心索引结构：
```python
# 多层级元数据索引
self.metadata_index = {
    'company_name': defaultdict(list),      # 公司名称索引
    'stock_code': defaultdict(list),        # 股票代码索引  
    'report_date': defaultdict(list),       # 报告日期索引
    'company_stock': defaultdict(list)      # 公司+股票组合索引
}
```

#### 索引构建逻辑：
```python
def _build_metadata_index(self):
    """构建元数据索引用于pre-filtering（仅中文数据）"""
    if self.dataset_type != "chinese":
        print("非中文数据集，跳过元数据索引构建")
        return
        
    for idx, record in enumerate(self.data):
        # 公司名称索引
        if record.get('company_name'):
            company_name = record['company_name'].strip().lower()
            self.metadata_index['company_name'][company_name].append(idx)
        
        # 股票代码索引
        if record.get('stock_code'):
            stock_code = str(record['stock_code']).strip().lower()
            self.metadata_index['stock_code'][stock_code].append(idx)
        
        # 报告日期索引
        if record.get('report_date'):
            report_date = record['report_date'].strip()
            self.metadata_index['report_date'][report_date].append(idx)
        
        # 公司名称+股票代码组合索引
        if record.get('company_name') and record.get('stock_code'):
            company_name = record['company_name'].strip().lower()
            stock_code = str(record['stock_code']).strip().lower()
            key = f"{company_name}_{stock_code}"
            self.metadata_index['company_stock'][key].append(idx)
```

### 查询解析和元数据提取

#### 中文查询解析逻辑：
```python
def _process_chinese_with_multi_stage(self, question: str, reranker_checkbox: bool):
    """使用多阶段检索系统处理中文查询"""
    
    # 尝试提取公司名称和股票代码用于元数据过滤
    company_name = None
    stock_code = None
    
    # 提取股票代码 - 匹配6位数字
    stock_match = re.search(r'\((\d{6})\)', question)
    if stock_match:
        stock_code = stock_match.group(1)
    
    # 提取公司名称 - 匹配公司后缀
    company_patterns = [
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险)[^，。？\s]*)'
    ]
    
    for pattern in company_patterns:
        company_match = re.search(pattern, question)
        if company_match:
            company_name = company_match.group(1)
            break
```

### 预过滤逻辑

#### 多阶段过滤策略：
```python
def pre_filter(self, 
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               max_candidates: int = 1000) -> List[int]:
    """
    基于元数据进行预过滤（仅中文数据支持）
    """
    if self.dataset_type != "chinese":
        print("非中文数据集，跳过预过滤")
        return list(range(len(self.data)))
    
    # 如果没有提供任何过滤条件，返回所有记录
    if not any([company_name, stock_code, report_date]):
        print("无过滤条件，返回所有记录")
        return list(range(len(self.data)))
    
    # 优先使用组合索引（公司名称+股票代码）
    if company_name and stock_code:
        company_name_lower = company_name.strip().lower()
        stock_code_lower = str(stock_code).strip().lower()
        key = f"{company_name_lower}_{stock_code_lower}"
        
        if key in self.metadata_index['company_stock']:
            indices = self.metadata_index['company_stock'][key]
            print(f"组合过滤: 公司'{company_name}' + 股票'{stock_code}' 匹配 {len(indices)} 条记录")
            return indices[:max_candidates]
        else:
            print(f"组合过滤: 公司'{company_name}' + 股票'{stock_code}' 无匹配记录")
            return []
    
    # 如果只提供了公司名称
    elif company_name:
        company_name_lower = company_name.strip().lower()
        if company_name_lower in self.metadata_index['company_name']:
            indices = self.metadata_index['company_name'][company_name_lower]
            print(f"公司名称过滤: '{company_name}' 匹配 {len(indices)} 条记录")
            return indices[:max_candidates]
        else:
            print(f"公司名称过滤: '{company_name}' 无匹配记录")
            return []
    
    # 如果只提供了股票代码
    elif stock_code:
        stock_code_lower = str(stock_code).strip().lower()
        if stock_code_lower in self.metadata_index['stock_code']:
            indices = self.metadata_index['stock_code'][stock_code_lower]
            print(f"股票代码过滤: '{stock_code}' 匹配 {len(indices)} 条记录")
            return indices[:max_candidates]
        else:
            print(f"股票代码过滤: '{stock_code}' 无匹配记录")
            return []
    
    # 如果只提供了报告日期
    elif report_date:
        report_date_str = report_date.strip()
        if report_date_str in self.metadata_index['report_date']:
            indices = self.metadata_index['report_date'][report_date_str]
            print(f"报告日期过滤: '{report_date}' 匹配 {len(indices)} 条记录")
            return indices[:max_candidates]
        else:
            print(f"报告日期过滤: '{report_date}' 无匹配记录")
            return []
```

### 完整检索流程

#### 多阶段检索系统：
```python
def search(self, 
           query: str,
           company_name: Optional[str] = None,
           stock_code: Optional[str] = None,
           report_date: Optional[str] = None,
           top_k: int = 20) -> Dict:
    """
    完整的多阶段检索流程
    """
    print(f"\n开始多阶段检索...")
    print(f"查询: {query}")
    print(f"数据集类型: {self.dataset_type}")
    
    if self.dataset_type == "chinese":
        if company_name:
            print(f"公司名称: {company_name}")
        if stock_code:
            print(f"股票代码: {stock_code}")
        if report_date:
            print(f"报告日期: {report_date}")
    else:
        print("英文数据集，不支持元数据过滤")
    
    # 1. 元数据预过滤
    candidate_indices = self.pre_filter(
        company_name=company_name,
        stock_code=stock_code,
        report_date=report_date,
        max_candidates=1000
    )
    
    # 2. FAISS向量检索
    faiss_results = self.faiss_search(query, candidate_indices, top_k=100)
    
    # 3. Qwen重排序
    reranked_results = self.rerank(query, faiss_results, top_k=20)
    
    # 4. LLM答案生成
    llm_answer = self.generate_answer(query, reranked_results, top_k_for_context=5)
    
    return {
        'retrieved_documents': self._format_results(reranked_results),
        'llm_answer': llm_answer
    }
```

### 元数据字段映射

#### AlphaFin数据字段：
```python
# 核心元数据字段
metadata_fields = {
    'company_name': '公司名称',
    'stock_code': '股票代码', 
    'report_date': '报告日期',
    'summary': '摘要（用于FAISS索引）',
    'original_context': '原始上下文',
    'generated_question': '生成的问题',
    'original_question': '原始问题',
    'original_answer': '原始答案'
}
```

### 过滤优先级

#### 过滤策略优先级：
1. **组合过滤**：公司名称 + 股票代码（最高精度）
2. **单一过滤**：公司名称 或 股票代码
3. **日期过滤**：报告日期
4. **无过滤**：返回所有记录

#### 示例查询处理：
```python
# 查询示例1：包含公司名称和股票代码
query = "中国宝武(600019)的业绩表现如何？"
# 提取：company_name="中国宝武", stock_code="600019"
# 使用组合索引过滤

# 查询示例2：仅包含公司名称  
query = "中国宝武公司的财务状况"
# 提取：company_name="中国宝武", stock_code=None
# 使用公司名称索引过滤

# 查询示例3：通用查询
query = "钢铁行业发展趋势"
# 提取：company_name=None, stock_code=None  
# 无过滤，返回所有记录
```

### 性能优化

#### 索引优化策略：
- **小写标准化**：所有索引键转换为小写
- **组合索引**：公司+股票组合提高查询精度
- **候选数量限制**：max_candidates防止过度检索
- **缓存机制**：索引构建后持久化存储

### 总结

这两种数据处理策略都针对各自数据集的特点进行了优化：

**TatQA的chunking策略**：
- 保持表格和段落的语义完整性
- 自动提取和标准化单位信息
- 生成适合LLM理解的自然语言描述

**AlphaFin的元数据过滤策略**：
- 根据查询中的实体信息进行精确预过滤
- 多层级索引结构提高查询效率
- 特别适合中文金融数据的检索需求

这种设计确保了在RAG系统中能够提供最相关的上下文信息，显著提高了检索的准确性和效率。 