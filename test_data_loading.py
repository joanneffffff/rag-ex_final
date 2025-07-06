#!/usr/bin/env python3
"""
测试数据加载的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def main():
    print("=== 测试数据加载 ===")
    
    # 初始化数据加载器
    data_loader = DualLanguageLoader()
    config = Config()
    
    print(f"英文数据路径: {config.data.english_data_path}")
    print(f"中文数据路径: {config.data.chinese_data_path}")
    
    # 测试英文数据加载
    print("\n1. 测试英文数据加载...")
    english_docs = data_loader.load_tatqa_context_only(config.data.english_data_path)
    print(f"✅ 英文文档加载完成: {len(english_docs)} 个文档")
    
    if english_docs:
        print("前3个英文文档预览:")
        for i, doc in enumerate(english_docs[:3]):
            content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            print(f"  文档 {i+1}: {content_preview}")
    
    # 测试中文数据加载
    print("\n2. 测试中文数据加载...")
    if config.data.chinese_data_path.endswith('.json'):
        chinese_docs = data_loader.load_alphafin_data(config.data.chinese_data_path)
    elif config.data.chinese_data_path.endswith('.jsonl'):
        chinese_docs = data_loader.load_jsonl_data(config.data.chinese_data_path, 'chinese')
    else:
        chinese_docs = []
    
    print(f"✅ 中文文档加载完成: {len(chinese_docs)} 个文档")
    
    if chinese_docs:
        print("前3个中文文档预览:")
        for i, doc in enumerate(chinese_docs[:3]):
            content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            print(f"  文档 {i+1}: {content_preview}")
    
    print(f"\n=== 总结 ===")
    print(f"英文文档: {len(english_docs)} 个")
    print(f"中文文档: {len(chinese_docs)} 个")
    
    if len(english_docs) == 0:
        print("❌ 英文文档加载失败！")
    else:
        print("✅ 英文文档加载成功！")
    
    if len(chinese_docs) == 0:
        print("❌ 中文文档加载失败！")
    else:
        print("✅ 中文文档加载成功！")

if __name__ == "__main__":
    main() 