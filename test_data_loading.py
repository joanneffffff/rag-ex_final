#!/usr/bin/env python3
"""
测试数据加载功能
"""

import json
from pathlib import Path

def test_data_loading(file_path: str):
    """测试数据加载功能"""
    print(f"正在测试数据加载: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 智能检测文件格式
            first_char = f.read(1)
            f.seek(0)  # 重置文件指针
            
            if first_char == '[':
                # 标准JSON数组格式
                print("检测到JSON数组格式，正在加载...")
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("文件以[开头但不是数组格式")
                print(f"✅ 成功加载JSON数组，样本数: {len(data)}")
            else:
                # 尝试JSONL格式（每行一个JSON对象）
                print("检测到JSONL格式，正在逐行加载...")
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                            print(f"问题行内容: {line[:100]}...")
                            continue
                
                if data:
                    print(f"✅ 成功加载JSONL格式，样本数: {len(data)}")
                else:
                    raise ValueError("JSONL解析失败，没有有效数据")
        
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
        print(f"❌ 加载评测数据失败: {e}")
        print(f"请检查文件格式是否正确，支持JSON数组格式和JSONL格式")
        return False

if __name__ == "__main__":
    # 测试实际的数据文件
    test_file = "data/alphafin/alphafin_eval_samples.jsonl"
    
    if Path(test_file).exists():
        success = test_data_loading(test_file)
        if success:
            print("\n✅ 数据加载测试成功！")
        else:
            print("\n❌ 数据加载测试失败！")
    else:
        print(f"❌ 文件不存在: {test_file}") 