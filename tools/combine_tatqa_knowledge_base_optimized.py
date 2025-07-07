#!/usr/bin/env python3
"""
合并TAT-QA训练和评估数据作为知识库（优化版本）
使用优化后的表格文本化数据
当有相同的context但不同的table ID或paragraph ID时，根据relevant_doc_ids来复制context
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
                            
                            doc = {
                                'doc_id': f"train_optimized_{doc_id_counter}",
                                'context': context,
                                'source': 'tatqa_train_optimized',
                                'language': 'english',
                                'created_at': '',
                                'author': '',
                                'table_id': table_id,
                                'paragraph_id': paragraph_id,
                                'relevant_doc_ids': relevant_doc_ids
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
                            
                            doc = {
                                'doc_id': f"eval_optimized_{doc_id_counter}",
                                'context': context,
                                'source': 'tatqa_eval_optimized',
                                'language': 'english',
                                'created_at': '',
                                'author': '',
                                'table_id': table_id,
                                'paragraph_id': paragraph_id,
                                'relevant_doc_ids': relevant_doc_ids
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
    
    # 智能去重和复制处理
    print("🔄 智能去重和复制处理...")
    unique_docs = []
    seen_contexts = {}  # 用于跟踪已处理的context
    
    # 按context去重，保留所有不同的context
    for doc in all_documents:
        context = doc.get('context', '')
        if not context:
            continue
            
        # 标准化context用于比较（去除多余空格）
        normalized_context = ' '.join(context.split())
        
        if normalized_context not in seen_contexts:
            # 新的context，直接添加
            seen_contexts[normalized_context] = doc
            unique_docs.append(doc)
        else:
            # 相同的context，合并ID信息
            existing_doc = seen_contexts[normalized_context]
            
            # 收集所有相关的ID信息
            if 'all_doc_ids' not in existing_doc:
                existing_doc['all_doc_ids'] = [existing_doc['doc_id']]
            if 'all_table_ids' not in existing_doc:
                existing_doc['all_table_ids'] = [existing_doc.get('table_id', '')] if existing_doc.get('table_id') else []
            if 'all_paragraph_ids' not in existing_doc:
                existing_doc['all_paragraph_ids'] = [existing_doc.get('paragraph_id', '')] if existing_doc.get('paragraph_id') else []
            if 'all_relevant_doc_ids' not in existing_doc:
                existing_doc['all_relevant_doc_ids'] = existing_doc.get('relevant_doc_ids', [])
            
            # 添加新的doc_id
            existing_doc['all_doc_ids'].append(doc['doc_id'])
            
            # 添加新的table_id（如果不同且不为空）
            new_table_id = doc.get('table_id', '')
            if new_table_id and new_table_id not in existing_doc['all_table_ids']:
                existing_doc['all_table_ids'].append(new_table_id)
            
            # 添加新的paragraph_id（如果不同且不为空）
            new_paragraph_id = doc.get('paragraph_id', '')
            if new_paragraph_id and new_paragraph_id not in existing_doc['all_paragraph_ids']:
                existing_doc['all_paragraph_ids'].append(new_paragraph_id)
            
            # 合并relevant_doc_ids
            existing_relevant_ids = set(existing_doc['all_relevant_doc_ids'])
            new_relevant_ids = set(doc.get('relevant_doc_ids', []))
            existing_doc['all_relevant_doc_ids'] = list(existing_relevant_ids.union(new_relevant_ids))
            
            print(f"  - 发现相同context但不同ID: {doc['doc_id']} -> 现有文档")
            print(f"    现有doc_id: {existing_doc['doc_id']}")
            print(f"    新doc_id: {doc['doc_id']}")
            if new_table_id and new_table_id not in existing_doc['all_table_ids']:
                print(f"    添加table_id: {new_table_id}")
            if new_paragraph_id and new_paragraph_id not in existing_doc['all_paragraph_ids']:
                print(f"    添加paragraph_id: {new_paragraph_id}")
    
    print(f"✅ 基于context去重后保留 {len(unique_docs)} 个文档")
    
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
                'author': doc['author']
            }
            
            # 添加ID相关字段（如果存在）
            if 'all_doc_ids' in doc:
                output_doc['all_doc_ids'] = doc['all_doc_ids']
            if 'all_table_ids' in doc:
                output_doc['all_table_ids'] = doc['all_table_ids']
            if 'all_paragraph_ids' in doc:
                output_doc['all_paragraph_ids'] = doc['all_paragraph_ids']
            if 'all_relevant_doc_ids' in doc:
                output_doc['all_relevant_doc_ids'] = doc['all_relevant_doc_ids']
            
            f.write(json.dumps(output_doc, ensure_ascii=False) + '\n')
    
    print(f"🎉 合并完成！")
    print(f"📊 统计信息:")
    print(f"   - 总文档数: {len(unique_docs)}")
    print(f"   - 训练文档: {len([d for d in unique_docs if d['source'] == 'tatqa_train_optimized'])}")
    print(f"   - 评估文档: {len([d for d in unique_docs if d['source'] == 'tatqa_eval_optimized'])}")
    print(f"   - 过滤的不完整表格: {incomplete_tables_count + eval_incomplete_count}")
    print(f"   - 输出文件: {output_file}")
    
    # 显示一些样本内容
    print(f"\n📋 样本内容预览:")
    for i, doc in enumerate(unique_docs[:3]):
        print(f"\n样本 {i+1} (ID: {doc['doc_id']}):")
        content = doc['context']
        if len(content) > 200:
            print(f"   {content[:200]}...")
        else:
            print(f"   {content}")
    
    return output_file

if __name__ == "__main__":
    combine_tatqa_knowledge_base_optimized() 