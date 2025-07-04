#!/usr/bin/env python3
"""
分析文档长度分布，确定最佳的预览长度
"""

import json
import sys
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.utils.optimized_data_loader import OptimizedDataLoader
from config.parameters import Config

def analyze_document_lengths():
    """分析文档长度分布"""
    print("=" * 80)
    print("分析文档长度分布 - 确定最佳预览长度")
    print("=" * 80)
    
    config = Config()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = OptimizedDataLoader(
        data_dir=config.data.data_dir,
        max_samples=1000,  # 加载更多样本进行分析
        chinese_document_level=True,
        english_chunk_level=True,
        include_eval_data=True
    )
    
    # 2. 收集长度数据
    print("\n2. 收集长度数据...")
    chinese_lengths = []
    english_lengths = []
    
    # 中文数据长度
    if loader.chinese_docs:
        for doc in loader.chinese_docs:
            chinese_lengths.append(len(doc.content))
    
    # 英文数据长度
    if loader.english_docs:
        for doc in loader.english_docs:
            english_lengths.append(len(doc.content))
    
    print(f"   中文文档数: {len(chinese_lengths)}")
    print(f"   英文文档数: {len(english_lengths)}")
    
    # 3. 分析中文数据
    if chinese_lengths:
        print("\n3. 中文数据长度分析:")
        chinese_lengths.sort()
        
        print(f"   最小长度: {min(chinese_lengths)} 字符")
        print(f"   最大长度: {max(chinese_lengths)} 字符")
        print(f"   平均长度: {sum(chinese_lengths) / len(chinese_lengths):.1f} 字符")
        print(f"   中位数长度: {chinese_lengths[len(chinese_lengths)//2]} 字符")
        
        # 计算不同预览长度的覆盖率
        preview_lengths = [500, 800, 1000, 1200, 1500, 2000, 2500, 3000]
        print(f"\n   不同预览长度的覆盖率:")
        for length in preview_lengths:
            covered = sum(1 for l in chinese_lengths if l <= length)
            percentage = (covered / len(chinese_lengths)) * 100
            print(f"     {length}字符: {covered}/{len(chinese_lengths)} ({percentage:.1f}%)")
        
        # 长度分布
        print(f"\n   长度分布:")
        ranges = [(0, 500), (501, 1000), (1001, 1500), (1501, 2000), (2001, 3000), (3001, 5000), (5001, float('inf'))]
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for l in chinese_lengths if l > start)
                range_str = f"{start}+"
            else:
                count = sum(1 for l in chinese_lengths if start <= l <= end)
                range_str = f"{start}-{end}"
            percentage = (count / len(chinese_lengths)) * 100
            print(f"     {range_str}字符: {count}个 ({percentage:.1f}%)")
    
    # 4. 分析英文数据
    if english_lengths:
        print("\n4. 英文数据长度分析:")
        english_lengths.sort()
        
        print(f"   最小长度: {min(english_lengths)} 字符")
        print(f"   最大长度: {max(english_lengths)} 字符")
        print(f"   平均长度: {sum(english_lengths) / len(english_lengths):.1f} 字符")
        print(f"   中位数长度: {english_lengths[len(english_lengths)//2]} 字符")
        
        # 计算不同预览长度的覆盖率
        print(f"\n   不同预览长度的覆盖率:")
        for length in preview_lengths:
            covered = sum(1 for l in english_lengths if l <= length)
            percentage = (covered / len(english_lengths)) * 100
            print(f"     {length}字符: {covered}/{len(english_lengths)} ({percentage:.1f}%)")
        
        # 长度分布
        print(f"\n   长度分布:")
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for l in english_lengths if l > start)
                range_str = f"{start}+"
            else:
                count = sum(1 for l in english_lengths if start <= l <= end)
                range_str = f"{start}-{end}"
            percentage = (count / len(english_lengths)) * 100
            print(f"     {range_str}字符: {count}个 ({percentage:.1f}%)")
    
    # 5. 推荐预览长度
    print("\n5. 推荐预览长度:")
    
    if chinese_lengths:
        # 找到覆盖90%中文文档的长度
        chinese_lengths.sort()
        index_90 = int(len(chinese_lengths) * 0.9)
        length_90_chinese = chinese_lengths[index_90]
        print(f"   中文数据90%覆盖率长度: {length_90_chinese} 字符")
    
    if english_lengths:
        # 找到覆盖90%英文文档的长度
        english_lengths.sort()
        index_90 = int(len(english_lengths) * 0.9)
        length_90_english = english_lengths[index_90]
        print(f"   英文数据90%覆盖率长度: {length_90_english} 字符")
    
    # 综合推荐
    if chinese_lengths and english_lengths:
        recommended_length = max(length_90_chinese, length_90_english)
        print(f"\n   综合推荐预览长度: {recommended_length} 字符")
        print(f"   理由: 覆盖90%的中英文文档，用户体验最佳")
        
        # 当前1500字符的效果
        chinese_covered = sum(1 for l in chinese_lengths if l <= 1500)
        english_covered = sum(1 for l in english_lengths if l <= 1500)
        total_covered = chinese_covered + english_covered
        total_docs = len(chinese_lengths) + len(english_lengths)
        current_coverage = (total_covered / total_docs) * 100
        
        print(f"\n   当前1500字符效果:")
        print(f"   总覆盖率: {total_covered}/{total_docs} ({current_coverage:.1f}%)")
        print(f"   中文覆盖率: {chinese_covered}/{len(chinese_lengths)} ({(chinese_covered/len(chinese_lengths)*100):.1f}%)")
        print(f"   英文覆盖率: {english_covered}/{len(english_lengths)} ({(english_covered/len(english_lengths)*100):.1f}%)")
        
        if current_coverage >= 90:
            print(f"   ✅ 当前1500字符已经很好，覆盖率超过90%")
        elif current_coverage >= 80:
            print(f"   ⚠️  当前1500字符还可以，覆盖率80%+，可以考虑增加到{recommended_length}")
        else:
            print(f"   ❌ 当前1500字符不够，建议增加到{recommended_length}")

if __name__ == "__main__":
    analyze_document_lengths() 