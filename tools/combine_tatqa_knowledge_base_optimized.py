#!/usr/bin/env python3
"""
合并TAT-QA训练和评估数据作为知识库（优化版本）
使用优化后的表格文本化数据
根据数据类型采用不同的去重策略：
- 单个表格：按table_id去重
- 单个段落：按paragraph_id去重
- 表格+文本：按(table_id + paragraph_id)组合去重，只保留一个
添加表格完整性检查，过滤掉不完整的表格数据
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import re

def is_complete_table(context: str) -> bool:
    """
    检查表格是否完整
    完整表格应该包含：
    1. Table ID
    2. Table columns
    3. 至少一行具体的数据（包含"For"开头的行）
    """
    if not context:
        return False
    
    # 检查是否包含表格标识
    if "Table ID:" not in context:
        return False
    
    # 检查是否包含列信息
    if "Table columns:" not in context:
        return False
    
    # 检查是否包含具体数据行（以"For"开头的行）
    # 这是判断表格是否完整的关键指标
    for_lines = re.findall(r'For [^:]+:', context)
    if len(for_lines) == 0:
        print(f"⚠️ 发现不完整表格，缺少具体数据行: {context[:100]}...")
        return False
    
    return True

def get_data_type(context: str) -> str:
    """
    判断数据类型：
    - 'table': 只包含表格数据（有Table ID但没有Paragraph ID）
    - 'paragraph': 只包含段落数据（有Paragraph ID但没有Table ID）
    - 'table+text': 包含表格和段落数据（同时有Table ID和Paragraph ID）
    """
    has_table = "Table ID:" in context
    has_paragraph = "Paragraph ID:" in context
    
    if has_table and has_paragraph:
        return "table+text"
    elif has_table:
        return "table"
    elif has_paragraph:
        return "paragraph"
    else:
        return "unknown"

def combine_tatqa_knowledge_base_optimized():
    """合并TAT-QA训练和评估数据作为知识库（优化版本）"""
    
    # 输入文件路径（使用enhanced版本）
    train_file = "evaluate_mrr/tatqa_train_qc_enhanced.jsonl"
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    
    # 输出文件路径
    output_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    print("🔄 开始合并TAT-QA知识库（优化版本）...")
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 收集所有文档
    all_documents = []
    doc_id_counter = 0
    incomplete_tables_count = 0
    
    # 处理训练文件
    print(f"📖 处理训练文件: {train_file}")
    if Path(train_file).exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        # 提取context作为知识库内容
                        context = item.get('context', '')
                        if context:
                            # 检查表格完整性
                            if not is_complete_table(context):
                                incomplete_tables_count += 1
                                continue
                            
                            # 提取table_id和paragraph_id用于去重判断
                            table_id = item.get('table_id', '')
                            paragraph_id = item.get('paragraph_id', '')
                            relevant_doc_ids = item.get('relevant_doc_ids', [])
                            
                            # 判断数据类型
                            data_type = get_data_type(context)
                            
                            doc = {
                                'doc_id': f"train_optimized_{doc_id_counter}",
                                'context': context,
                                'source': 'tatqa_train_optimized',
                                'language': 'english',
                                'created_at': '',
                                'author': '',
                                'table_id': table_id,
                                'paragraph_id': paragraph_id,
                                'relevant_doc_ids': relevant_doc_ids,
                                'data_type': data_type
                            }
                            all_documents.append(doc)
                            doc_id_counter += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 训练文件第{line_num}行JSON解析错误: {e}")
                        continue
    else:
        print(f"❌ 训练文件不存在: {train_file}")
        return None
    
    print(f"✅ 从训练文件加载了 {doc_id_counter} 个完整文档")
    print(f"⚠️ 过滤掉了 {incomplete_tables_count} 个不完整表格")
    
    # 处理评估文件
    print(f"📖 处理评估文件: {eval_file}")
    eval_start_id = doc_id_counter
    eval_incomplete_count = 0
    
    if Path(eval_file).exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        # 提取context作为知识库内容
                        context = item.get('context', '')
                        if context:
                            # 检查表格完整性
                            if not is_complete_table(context):
                                eval_incomplete_count += 1
                                continue
                            
                            # 提取table_id和paragraph_id用于去重判断
                            table_id = item.get('table_id', '')
                            paragraph_id = item.get('paragraph_id', '')
                            relevant_doc_ids = item.get('relevant_doc_ids', [])
                            
                            # 判断数据类型
                            data_type = get_data_type(context)
                            
                            doc = {
                                'doc_id': f"eval_optimized_{doc_id_counter}",
                                'context': context,
                                'source': 'tatqa_eval_optimized',
                                'language': 'english',
                                'created_at': '',
                                'author': '',
                                'table_id': table_id,
                                'paragraph_id': paragraph_id,
                                'relevant_doc_ids': relevant_doc_ids,
                                'data_type': data_type
                            }
                            all_documents.append(doc)
                            doc_id_counter += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 评估文件第{line_num}行JSON解析错误: {e}")
                        continue
    else:
        print(f"❌ 评估文件不存在: {eval_file}")
        return None
    
    print(f"✅ 从评估文件加载了 {doc_id_counter - eval_start_id} 个完整文档")
    print(f"⚠️ 过滤掉了 {eval_incomplete_count} 个不完整表格")
    
    # 先合并所有数据，然后按数据类型分组去重
    print("🔄 先合并所有数据，然后按数据类型分组去重...")
    
    # 按数据类型分组
    table_docs = []  # 单个表格
    paragraph_docs = []  # 单个段落
    table_text_docs = []  # 表格+文本
    
    for doc in all_documents:
        data_type = doc.get('data_type', 'unknown')
        if data_type == 'table':
            table_docs.append(doc)
        elif data_type == 'paragraph':
            paragraph_docs.append(doc)
        elif data_type == 'table+text':
            table_text_docs.append(doc)
        else:
            print(f"⚠️ 未知数据类型: {data_type}, doc_id: {doc['doc_id']}")
    
    print(f"📊 数据类型统计:")
    print(f"   - 单个表格: {len(table_docs)}")
    print(f"   - 单个段落: {len(paragraph_docs)}")
    print(f"   - 表格+文本: {len(table_text_docs)}")
    
    # 去重处理
    unique_docs = []
    
    # 1. 单个表格：按relevant_doc_ids去重，只保留一个文档但保存所有相关ID
    print("🔄 处理单个表格数据（按relevant_doc_ids去重，只保留一个文档）...")
    table_seen = {}
    for doc in table_docs:
        relevant_doc_ids = tuple(sorted(doc.get('relevant_doc_ids', [])))
        if relevant_doc_ids not in table_seen:
            table_seen[relevant_doc_ids] = doc
            unique_docs.append(doc)
        else:
            # 只保留第一个文档，但收集所有相关ID
            existing_doc = table_seen[relevant_doc_ids]
            if 'all_doc_ids' not in existing_doc:
                existing_doc['all_doc_ids'] = [existing_doc['doc_id']]
            existing_doc['all_doc_ids'].append(doc['doc_id'])
            print(f"  - 表格去重: {doc['doc_id']} -> {existing_doc['doc_id']} (relevant_doc_ids: {relevant_doc_ids})")
    
    # 2. 单个段落：按relevant_doc_ids去重，只保留一个文档但保存所有相关ID
    print("🔄 处理单个段落数据（按relevant_doc_ids去重，只保留一个文档）...")
    paragraph_seen = {}
    for doc in paragraph_docs:
        relevant_doc_ids = tuple(sorted(doc.get('relevant_doc_ids', [])))
        if relevant_doc_ids not in paragraph_seen:
            paragraph_seen[relevant_doc_ids] = doc
            unique_docs.append(doc)
        else:
            # 只保留第一个文档，但收集所有相关ID
            existing_doc = paragraph_seen[relevant_doc_ids]
            if 'all_doc_ids' not in existing_doc:
                existing_doc['all_doc_ids'] = [existing_doc['doc_id']]
            existing_doc['all_doc_ids'].append(doc['doc_id'])
            print(f"  - 段落去重: {doc['doc_id']} -> {existing_doc['doc_id']} (relevant_doc_ids: {relevant_doc_ids})")
    
    # 3. 表格+文本：按relevant_doc_ids去重，只保留一个
    print("🔄 处理表格+文本数据（按relevant_doc_ids去重，只保留一个）...")
    table_text_seen = {}
    for doc in table_text_docs:
        relevant_doc_ids = tuple(sorted(doc.get('relevant_doc_ids', [])))
        if relevant_doc_ids not in table_text_seen:
            table_text_seen[relevant_doc_ids] = doc
            unique_docs.append(doc)
        else:
            # 只保留第一个，不合并ID信息
            existing_doc = table_text_seen[relevant_doc_ids]
            print(f"  - 表格+文本去重: {doc['doc_id']} -> {existing_doc['doc_id']} (relevant_doc_ids: {relevant_doc_ids})")
    
    print(f"✅ 去重后保留 {len(unique_docs)} 个文档")
    
    # 写入合并后的文件
    print(f"💾 写入合并文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in unique_docs:
            # 保留标准字段，并添加新的ID字段
            output_doc = {
                'doc_id': doc['doc_id'],
                'context': doc['context'],
                'source': doc['source'],
                'language': doc['language'],
                'created_at': doc['created_at'],
                'author': doc['author'],
                'data_type': doc['data_type']
            }
            
            # 添加ID相关字段（如果存在）
            if 'all_doc_ids' in doc:
                output_doc['all_doc_ids'] = doc['all_doc_ids']
            
            f.write(json.dumps(output_doc, ensure_ascii=False) + '\n')
    
    print(f"🎉 合并完成！")
    print(f"📊 统计信息:")
    print(f"   - 总文档数: {len(unique_docs)}")
    print(f"   - 单个表格: {len([d for d in unique_docs if d['data_type'] == 'table'])}")
    print(f"   - 单个段落: {len([d for d in unique_docs if d['data_type'] == 'paragraph'])}")
    print(f"   - 表格+文本: {len([d for d in unique_docs if d['data_type'] == 'table+text'])}")
    print(f"   - 训练文档: {len([d for d in unique_docs if d['source'] == 'tatqa_train_optimized'])}")
    print(f"   - 评估文档: {len([d for d in unique_docs if d['source'] == 'tatqa_eval_optimized'])}")
    print(f"   - 过滤的不完整表格: {incomplete_tables_count + eval_incomplete_count}")
    print(f"   - 输出文件: {output_file}")
    
    # 显示一些样本内容
    print(f"\n📋 样本内容预览:")
    for i, doc in enumerate(unique_docs[:3]):
        print(f"\n样本 {i+1} (ID: {doc['doc_id']}, 类型: {doc['data_type']}):")
        content = doc['context']
        if len(content) > 200:
            print(f"   {content[:200]}...")
        else:
            print(f"   {content}")
    
    return output_file

if __name__ == "__main__":
    combine_tatqa_knowledge_base_optimized() 