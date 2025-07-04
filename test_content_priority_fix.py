#!/usr/bin/env python3
"""
测试内容字段优先级修复
验证UI是否能正确显示完整的原始context
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.utils.optimized_data_loader import OptimizedDataLoader
from config.parameters import Config

def test_content_priority_fix():
    """测试内容字段优先级修复"""
    print("=" * 80)
    print("测试内容字段优先级修复 - 验证完整原始context显示")
    print("=" * 80)
    
    config = Config()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = OptimizedDataLoader(
        data_dir=config.data.data_dir,
        max_samples=10,  # 只加载少量样本进行测试
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
    
    # 3. 验证内容长度
    print("\n3. 验证内容长度...")
    for i, doc in enumerate(desai_docs[:3]):  # 只检查前3个
        content_length = len(doc.content)
        print(f"文档 {i+1}:")
        print(f"  内容长度: {content_length} 字符")
        print(f"  内容预览: {doc.content[:200]}...")
        print(f"  是否包含完整研报内容: {'是' if content_length > 1000 else '否'}")
        print()
    
    # 4. 检查原始数据文件
    print("\n4. 检查原始数据文件...")
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
            
            # 判断应该使用哪个字段
            if context_length > 500:
                should_use = "context"
            elif original_content_length > context_length:
                should_use = "original_content"
            else:
                should_use = "context"
            
            print(f"  应该使用: {should_use}")
            print(f"  实际内容预览: {record.get(should_use, '')[:200]}...")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_content_priority_fix() 