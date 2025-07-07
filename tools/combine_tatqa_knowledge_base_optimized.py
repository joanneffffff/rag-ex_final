#!/usr/bin/env python3
"""
合并TAT-QA训练和评估数据作为知识库（优化版本）
使用优化后的表格文本化数据
当有相同的context但不同的table ID或paragraph ID时，根据relevant_doc_ids来复制context
"""

import json
from pathlib import Path
from typing import List, Dict, Any

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
    
    print(f"✅ 从训练文件加载了 {doc_id_counter} 个文档")
    
    # 处理评估文件
    print(f"📖 处理评估文件: {eval_file}")
    eval_start_id = doc_id_counter
    
    if Path(eval_file).exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        # 提取context作为知识库内容
                        context = item.get('context', '')
                        if context:
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
    
    print(f"✅ 从评估文件加载了 {doc_id_counter - eval_start_id} 个文档")
    
    # 智能去重和复制处理
    print("🔄 智能去重和复制处理...")
    unique_docs = []
    seen_relevant_doc_ids_combinations = set()  # 用于跟踪已处理的relevant_doc_ids组合
    
    # 按relevant_doc_ids去重，保留所有不同的relevant_doc_ids组合
    for doc in all_documents:
        relevant_doc_ids = doc.get('relevant_doc_ids', [])
        if isinstance(relevant_doc_ids, str):
            # 如果是字符串，尝试解析
            try:
                relevant_doc_ids = json.loads(relevant_doc_ids)
            except:
                relevant_doc_ids = [relevant_doc_ids]
        
        # 将relevant_doc_ids排序后转为元组，用于去重
        relevant_doc_ids_tuple = tuple(sorted(relevant_doc_ids)) if relevant_doc_ids else ()
        
        if relevant_doc_ids_tuple not in seen_relevant_doc_ids_combinations:
            seen_relevant_doc_ids_combinations.add(relevant_doc_ids_tuple)
            unique_docs.append(doc)
        else:
            # 如果relevant_doc_ids组合已存在，收集ID信息并更新context
            existing_doc = next(d for d in unique_docs if tuple(sorted(d.get('relevant_doc_ids', []))) == relevant_doc_ids_tuple)
            
            # 收集所有相关的ID信息
            if 'all_doc_ids' not in existing_doc:
                existing_doc['all_doc_ids'] = [existing_doc['doc_id']]
            if 'all_relevant_doc_ids' not in existing_doc:
                existing_doc['all_relevant_doc_ids'] = existing_doc.get('relevant_doc_ids', [])
            
            # 添加新的doc_id
            existing_doc['all_doc_ids'].append(doc['doc_id'])
            
            # 合并relevant_doc_ids
            existing_relevant_ids = set(existing_doc['all_relevant_doc_ids'])
            new_relevant_ids = set(doc.get('relevant_doc_ids', []))
            existing_doc['all_relevant_doc_ids'] = list(existing_relevant_ids.union(new_relevant_ids))
            
            # 如果context不同，选择更长的context
            if existing_doc['context'] != doc['context']:
                print(f"  - 发现相同relevant_doc_ids但不同context: {relevant_doc_ids_tuple}")
                print(f"    现有context长度: {len(existing_doc['context'])}")
                print(f"    新context长度: {len(doc['context'])}")
                
                # 选择更长的context（通常包含更多信息）
                if len(doc['context']) > len(existing_doc['context']):
                    existing_doc['context'] = doc['context']
                    print(f"    更新为更长的context")
            
            print(f"  - 合并doc_id: {doc['doc_id']} -> 现有文档")
    
    print(f"✅ 基于relevant_doc_ids去重后保留 {len(unique_docs)} 个文档")
    
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
            if 'all_relevant_doc_ids' in doc:
                output_doc['all_relevant_doc_ids'] = doc['all_relevant_doc_ids']
            
            f.write(json.dumps(output_doc, ensure_ascii=False) + '\n')
    
    print(f"🎉 合并完成！")
    print(f"📊 统计信息:")
    print(f"   - 总文档数: {len(unique_docs)}")
    print(f"   - 训练文档: {len([d for d in unique_docs if d['source'] == 'tatqa_train_optimized'])}")
    print(f"   - 评估文档: {len([d for d in unique_docs if d['source'] == 'tatqa_eval_optimized'])}")
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