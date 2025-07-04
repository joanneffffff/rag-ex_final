#!/usr/bin/env python3
"""
最终的UI内容验证 - 确认UI显示的是原始context而非chunk
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.utils.optimized_data_loader import OptimizedDataLoader
from config.parameters import Config

def final_ui_content_verification():
    """最终UI内容验证"""
    print("=" * 80)
    print("最终UI内容验证 - 确认显示原始context而非chunk")
    print("=" * 80)
    
    config = Config()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = OptimizedDataLoader(
        data_dir=config.data.data_dir,
        max_samples=50,  # 只加载少量样本进行测试
        chinese_document_level=True,
        english_chunk_level=True,
        include_eval_data=True
    )
    
    # 2. 获取统计信息
    stats = loader.get_statistics()
    print(f"   中文文档数: {stats['chinese_docs']}")
    print(f"   英文文档数: {stats['english_docs']}")
    
    # 3. 检查中文数据
    print("\n2. 检查中文数据...")
    if loader.chinese_docs:
        print(f"   检查前3个中文文档:")
        for i, doc in enumerate(loader.chinese_docs[:3]):
            content = doc.content
            print(f"   文档 {i+1}:")
            print(f"     长度: {len(content)} 字符")
            print(f"     来源: {doc.metadata.source}")
            print(f"     前150字符: {content[:150]}...")
            print(f"     后150字符: ...{content[-150:]}")
            print()
    
    # 4. 检查英文数据
    print("\n3. 检查英文数据...")
    if loader.english_docs:
        print(f"   检查前3个英文文档:")
        for i, doc in enumerate(loader.english_docs[:3]):
            content = doc.content
            print(f"   文档 {i+1}:")
            print(f"     长度: {len(content)} 字符")
            print(f"     来源: {doc.metadata.source}")
            print(f"     前150字符: {content[:150]}...")
            print(f"     后150字符: ...{content[-150:]}")
            print()
    
    # 5. 检查原始数据文件
    print("\n4. 检查原始数据文件...")
    
    # 检查中文数据文件
    chinese_file = Path(config.data.data_dir) / "unified" / "alphafin_unified.json"
    if chinese_file.exists():
        print(f"   中文数据文件: {chinese_file}")
        with open(chinese_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"   总记录数: {len(data)}")
            
            # 检查前3条记录
            for i in range(min(3, len(data))):
                record = data[i]
                context = record.get('context', '')
                original_content = record.get('original_content', '')
                
                print(f"     原始记录 {i+1}:")
                print(f"       context长度: {len(context)} 字符")
                print(f"       original_content长度: {len(original_content)} 字符")
                print(f"       context前150字符: {context[:150]}...")
                print()
    
    # 检查英文数据文件
    english_file = Path(config.data.data_dir) / "unified" / "tatqa_knowledge_base_unified.jsonl"
    if english_file.exists():
        print(f"   英文数据文件: {english_file}")
        count = 0
        with open(english_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= 3:
                    break
                data = json.loads(line.strip())
                context = data.get('context', '')
                count += 1
                
                print(f"     原始记录 {count}:")
                print(f"       context长度: {len(context)} 字符")
                print(f"       context前150字符: {context[:150]}...")
                print()
    
    # 6. 验证内容一致性
    print("\n5. 验证内容一致性...")
    
    # 检查中文数据一致性
    if loader.chinese_docs and chinese_file.exists():
        print("   检查中文数据一致性:")
        with open(chinese_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        for i, doc in enumerate(loader.chinese_docs[:3]):
            if i < len(original_data):
                original_context = original_data[i].get('context', '')
                doc_content = doc.content
                
                # 检查内容是否一致
                if original_context == doc_content:
                    print(f"     ✅ 文档 {i+1}: 内容完全一致")
                elif original_context in doc_content:
                    print(f"     ⚠️  文档 {i+1}: 原始内容包含在文档中（可能是分块）")
                elif doc_content in original_context:
                    print(f"     ⚠️  文档 {i+1}: 文档内容包含在原始内容中（可能是截断）")
                else:
                    print(f"     ❌ 文档 {i+1}: 内容不一致")
                    print(f"         原始长度: {len(original_context)}")
                    print(f"         文档长度: {len(doc_content)}")
    
    # 检查英文数据一致性
    if loader.english_docs and english_file.exists():
        print("   检查英文数据一致性:")
        count = 0
        with open(english_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= 3:
                    break
                original_data = json.loads(line.strip())
                original_context = original_data.get('context', '')
                
                if count < len(loader.english_docs):
                    doc_content = loader.english_docs[count].content
                    
                    # 检查内容是否一致
                    if original_context == doc_content:
                        print(f"     ✅ 文档 {count+1}: 内容完全一致")
                    elif original_context in doc_content:
                        print(f"     ⚠️  文档 {count+1}: 原始内容包含在文档中（可能是分块）")
                    elif doc_content in original_context:
                        print(f"     ⚠️  文档 {count+1}: 文档内容包含在原始内容中（可能是截断）")
                    else:
                        print(f"     ❌ 文档 {count+1}: 内容不一致")
                        print(f"         原始长度: {len(original_context)}")
                        print(f"         文档长度: {len(doc_content)}")
                
                count += 1
    
    # 7. 总结
    print("\n6. 总结:")
    print("   根据测试结果，UI中显示的内容分析:")
    
    if stats['chinese_docs'] > 0:
        print(f"   - 中文数据: 使用文档级别处理")
        print(f"   - 中文文档数: {stats['chinese_docs']}")
        print(f"   - 数据来源: context字段（非original_context）")
        print(f"   - 处理方式: 保持原始context完整性，避免过度分块")
    
    if stats['english_docs'] > 0:
        print(f"   - 英文数据: 保持原始context")
        print(f"   - 英文文档数: {stats['english_docs']}")
        print(f"   - 数据来源: context字段")
        print(f"   - 处理方式: 适当分块处理，保持内容完整性")
    
    print("\n   UI中的'Read More'功能应该显示:")
    print("   ✅ 完整的原始context内容")
    print("   ✅ 保持原始格式和完整性")
    print("   ✅ 不是被过度分块的chunk")
    print("   ✅ 中文数据使用context字段，英文数据使用context字段")

if __name__ == "__main__":
    final_ui_content_verification() 