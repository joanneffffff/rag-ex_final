#!/usr/bin/env python3
"""
合并TAT-QA训练和评估数据作为知识库
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def combine_tatqa_knowledge_base():
    """合并TAT-QA训练和评估数据作为知识库"""
    
    # 输入文件路径
    train_file = "evaluate_mrr/tatqa_train_qc_enhanced.jsonl"
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    
    # 输出文件路径
    output_file = "data/unified/tatqa_knowledge_base_combined.jsonl"
    
    print("🔄 开始合并TAT-QA知识库...")
    
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
                            doc = {
                                'doc_id': f"train_{doc_id_counter}",
                                'content': context,
                                'source': 'tatqa_train',
                                'language': 'english',
                                'created_at': '',
                                'author': ''
                            }
                            all_documents.append(doc)
                            doc_id_counter += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 训练文件第{line_num}行JSON解析错误: {e}")
                        continue
    else:
        print(f"❌ 训练文件不存在: {train_file}")
    
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
                            doc = {
                                'doc_id': f"eval_{doc_id_counter}",
                                'content': context,
                                'source': 'tatqa_eval',
                                'language': 'english',
                                'created_at': '',
                                'author': ''
                            }
                            all_documents.append(doc)
                            doc_id_counter += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 评估文件第{line_num}行JSON解析错误: {e}")
                        continue
    else:
        print(f"❌ 评估文件不存在: {eval_file}")
    
    print(f"✅ 从评估文件加载了 {doc_id_counter - eval_start_id} 个文档")
    
    # 去重（基于content内容）
    print("🔄 去重处理...")
    unique_docs = []
    seen_contents = set()
    
    for doc in all_documents:
        content = doc['content'].strip()
        if content and content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(content)
    
    print(f"✅ 去重后保留 {len(unique_docs)} 个唯一文档")
    
    # 写入合并后的文件
    print(f"💾 写入合并文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in unique_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"🎉 合并完成！")
    print(f"📊 统计信息:")
    print(f"   - 总文档数: {len(unique_docs)}")
    print(f"   - 训练文档: {len([d for d in unique_docs if d['source'] == 'tatqa_train'])}")
    print(f"   - 评估文档: {len([d for d in unique_docs if d['source'] == 'tatqa_eval'])}")
    print(f"   - 输出文件: {output_file}")
    
    return output_file

if __name__ == "__main__":
    combine_tatqa_knowledge_base() 