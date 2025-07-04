#!/usr/bin/env python3
"""
测试中文查询时summary和context的集成效果
验证UI显示完整context，prompt同时使用summary和context
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.utils.optimized_data_loader import OptimizedDataLoader
from config.parameters import Config

def test_summary_context_integration():
    """测试summary和context集成效果"""
    print("=" * 80)
    print("测试中文查询时summary和context的集成效果")
    print("=" * 80)
    
    config = Config()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = OptimizedDataLoader(
        data_dir=config.data.data_dir,
        max_samples=5,  # 只加载少量样本进行测试
        chinese_document_level=True,
        english_chunk_level=True,
        include_eval_data=True
    )
    
    # 2. 查找德赛电池相关文档
    print("\n2. 查找德赛电池相关文档...")
    desai_docs = []
    
    for doc in loader.chinese_docs:
        if "德赛电池" in doc.content:
            desai_docs.append(doc)
    
    print(f"找到 {len(desai_docs)} 个德赛电池相关文档")
    
    # 3. 验证数据字段
    print("\n3. 验证数据字段...")
    for i, doc in enumerate(desai_docs[:2]):  # 只检查前2个
        print(f"文档 {i+1}:")
        print(f"  内容长度: {len(doc.content)} 字符")
        print(f"  内容预览: {doc.content[:200]}...")
        print(f"  元数据: {doc.metadata}")
        print()
    
    # 4. 检查原始数据文件中的字段
    print("\n4. 检查原始数据文件中的字段...")
    import json
    
    alphafin_path = Path(config.data.chinese_data_path)
    if alphafin_path.exists():
        with open(alphafin_path, 'r', encoding='utf-8') as f:
            if alphafin_path.suffix == '.jsonl':
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                data = json.load(f)
        
        # 查找德赛电池记录
        desai_records = []
        for record in data:
            if isinstance(record, dict) and "德赛电池" in str(record):
                desai_records.append(record)
        
        print(f"原始数据中找到 {len(desai_records)} 个德赛电池记录")
        
        for i, record in enumerate(desai_records[:2]):  # 只检查前2个
            print(f"\n记录 {i+1}:")
            context_length = len(record.get('context', ''))
            original_content_length = len(record.get('original_content', ''))
            summary_length = len(record.get('summary', ''))
            
            print(f"  context长度: {context_length}")
            print(f"  original_content长度: {original_content_length}")
            print(f"  summary长度: {summary_length}")
            
            print(f"  summary内容: {record.get('summary', '')[:100]}...")
            print(f"  context内容: {record.get('context', '')[:100]}...")
    
    # 5. 模拟UI显示和Prompt生成
    print("\n5. 模拟UI显示和Prompt生成...")
    
    # 模拟UI显示（只显示context）
    print("UI显示内容（context）:")
    for i, doc in enumerate(desai_docs[:1]):
        print(f"  文档 {i+1}: {doc.content[:300]}...")
    
    # 模拟Prompt生成（中文查询同时使用summary和context）
    print("\nPrompt生成（中文查询）:")
    sample_question = "德赛电池的业绩如何？"
    
    # 从原始数据中获取summary
    if desai_records:
        record = desai_records[0]
        summary = record.get('summary', '')
        context = record.get('context', '')
        
        if summary and context:
            prompt = f"摘要：{summary}\n\n完整上下文：{context}\n\n问题：{sample_question}\n\n回答："
            print(f"  生成的prompt长度: {len(prompt)} 字符")
            print(f"  summary长度: {len(summary)} 字符")
            print(f"  context长度: {len(context)} 字符")
            print(f"  prompt预览: {prompt[:200]}...")
        else:
            print("  未找到summary或context字段")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_summary_context_integration() 