#!/usr/bin/env python3
"""
测试UI中显示的内容是否真的是原始context而不是被分块的chunk
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from xlm.utils.optimized_data_loader import OptimizedDataLoader
from xlm.dto.dto import DocumentWithMetadata
from config.parameters import Config

def test_ui_content_verification():
    """测试UI内容验证"""
    print("=" * 80)
    print("UI内容验证测试 - 检查是否显示原始context而非chunk")
    print("=" * 80)
    
    config = Config()
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = OptimizedDataLoader(
        data_dir=config.data.data_dir,
        max_samples=100,  # 只加载少量样本进行测试
        chinese_document_level=True,
        english_chunk_level=True,
        include_eval_data=True
    )
    
    # 2. 获取统计信息
    stats = loader.get_statistics()
    print(f"   中文文档数: {stats['chinese_docs']}")
    print(f"   英文文档数: {stats['english_docs']}")
    
    # 3. 检查中文数据
    print("\n2. 检查中文数据内容...")
    if loader.chinese_docs:
        print(f"   检查前5个中文文档:")
        for i, doc in enumerate(loader.chinese_docs[:5]):
            content = doc.content
            print(f"   文档 {i+1}:")
            print(f"     长度: {len(content)} 字符")
            print(f"     来源: {doc.metadata.source}")
            print(f"     前100字符: {content[:100]}...")
            print(f"     后100字符: ...{content[-100:]}")
            print()
    
    # 4. 检查英文数据
    print("\n3. 检查英文数据内容...")
    if loader.english_docs:
        print(f"   检查前5个英文文档:")
        for i, doc in enumerate(loader.english_docs[:5]):
            content = doc.content
            print(f"   文档 {i+1}:")
            print(f"     长度: {len(content)} 字符")
            print(f"     来源: {doc.metadata.source}")
            print(f"     前100字符: {content[:100]}...")
            print(f"     后100字符: ...{content[-100:]}")
            print()
    
    # 5. 检查原始数据文件
    print("\n4. 检查原始数据文件...")
    
    # 检查中文数据文件
    chinese_file = Path(config.data.data_dir) / "unified" / "alphafin_unified.json"
    if chinese_file.exists():
        print(f"   中文数据文件: {chinese_file}")
        with open(chinese_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, record in enumerate(data[:3]):  # 只检查前3条
                original_context = record.get('original_context', '')
                print(f"     原始记录 {i+1}:")
                print(f"       原始context长度: {len(original_context)} 字符")
                print(f"       前100字符: {original_context[:100]}...")
                print()
    
    # 检查英文数据文件
    english_file = Path(config.data.data_dir) / "unified" / "tatqa_knowledge_base_unified.jsonl"
    if english_file.exists():
        print(f"   英文数据文件: {english_file}")
        with open(english_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只检查前3条
                    break
                data = json.loads(line.strip())
                context = data.get('context', '')
                print(f"     原始记录 {i+1}:")
                print(f"       原始context长度: {len(context)} 字符")
                print(f"       前100字符: {context[:100]}...")
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
                original_context = original_data[i].get('original_context', '')
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
        with open(english_file, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line.strip()) for line in f]
        
        for i, doc in enumerate(loader.english_docs[:3]):
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
    
    # 7. 总结
    print("\n6. 总结:")
    print("   根据测试结果，UI中显示的内容应该是:")
    
    if stats['chinese_docs'] > 0:
        print(f"   - 中文数据: 文档级别处理，保持原始context完整性")
        print(f"   - 中文文档数: {stats['chinese_docs']}")
    
    if stats['english_docs'] > 0:
        print(f"   - 英文数据: 保持原始context，适当分块处理")
        print(f"   - 英文文档数: {stats['english_docs']}")
    
    print("\n   UI中的'Read More'功能应该显示完整的原始context内容，")
    print("   而不是被过度分块的chunk。如果显示的是chunk，说明数据加载")
    print("   过程中进行了过度分割。")

if __name__ == "__main__":
    test_ui_content_verification() 