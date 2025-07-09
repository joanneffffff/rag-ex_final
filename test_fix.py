#!/usr/bin/env python3
"""
测试修复后的数据加载功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import load_json_or_jsonl

def test_fixed_loading():
    """测试修复后的数据加载功能"""
    test_file = "data/alphafin/alphafin_eval_samples.jsonl"
    
    print(f"测试文件: {test_file}")
    print(f"文件是否存在: {Path(test_file).exists()}")
    
    if not Path(test_file).exists():
        print("❌ 文件不存在")
        return False
    
    try:
        # 使用修复后的加载函数
        data = load_json_or_jsonl(test_file)
        
        print(f"✅ 成功加载数据，样本数: {len(data)}")
        
        # 显示前几个样本的结构
        if data:
            print(f"\n前3个样本的字段:")
            for i, sample in enumerate(data[:3]):
                print(f"样本 {i+1}: {list(sample.keys())}")
                if 'generated_question' in sample:
                    print(f"  问题: {sample['generated_question'][:50]}...")
                if 'doc_id' in sample:
                    print(f"  文档ID: {sample['doc_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_loading()
    if success:
        print("\n✅ 修复成功！数据加载功能正常工作。")
    else:
        print("\n❌ 修复失败！") 