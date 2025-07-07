#!/usr/bin/env python3
"""
诊断英文数据加载问题的脚本
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from xlm.utils.dual_language_loader import DualLanguageLoader
from xlm.dto.dto import DocumentWithMetadata
from config.parameters import Config

def debug_english_data_loading():
    """诊断英文数据加载问题"""
    print("=== 英文数据加载诊断 ===")
    
    # 1. 检查配置文件
    config = Config()
    english_data_path = config.data.english_data_path
    print(f"1. 配置文件中的英文数据路径: {english_data_path}")
    
    # 2. 检查文件是否存在
    if Path(english_data_path).exists():
        print(f"✅ 文件存在")
        file_size = Path(english_data_path).stat().st_size
        print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ 文件不存在")
        return
    
    # 3. 检查文件内容
    print(f"\n2. 检查文件内容...")
    try:
        with open(english_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   总行数: {len(lines)}")
            
            # 检查前几行
            for i in range(min(3, len(lines))):
                try:
                    item = json.loads(lines[i].strip())
                    context = item.get('context', '')
                    print(f"   第{i+1}行 - context长度: {len(context)}")
                    print(f"   第{i+1}行 - context预览: {context[:100]}...")
                except json.JSONDecodeError as e:
                    print(f"   第{i+1}行 - JSON解析失败: {e}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 4. 使用DualLanguageLoader加载
    print(f"\n3. 使用DualLanguageLoader加载...")
    data_loader = DualLanguageLoader()
    
    try:
        english_docs = data_loader.load_tatqa_context_only(english_data_path)
        print(f"   加载的英文文档数量: {len(english_docs)}")
        
        if english_docs:
            # 检查第一个文档
            first_doc = english_docs[0]
            print(f"   第一个文档类型: {type(first_doc)}")
            print(f"   第一个文档content长度: {len(first_doc.content)}")
            print(f"   第一个文档content预览: {first_doc.content[:100]}...")
            print(f"   第一个文档metadata: {first_doc.metadata}")
            
            # 检查语言分布
            languages = {}
            for doc in english_docs:
                lang = doc.metadata.language
                languages[lang] = languages.get(lang, 0) + 1
            
            print(f"   语言分布: {languages}")
        else:
            print("   ⚠️ 没有加载到任何英文文档")
            
    except Exception as e:
        print(f"❌ DualLanguageLoader加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 检查语言检测
    print(f"\n4. 测试语言检测...")
    test_texts = [
        "What is the revenue of the company?",
        "The financial statements show",
        "Table 1 presents the quarterly results",
        "Paragraph 2 discusses the market trends"
    ]
    
    for i, text in enumerate(test_texts):
        try:
            detected_lang = data_loader.detect_language(text)
            print(f"   测试文本{i+1}: {detected_lang} - {text[:50]}...")
        except Exception as e:
            print(f"   测试文本{i+1}: 检测失败 - {e}")

if __name__ == "__main__":
    debug_english_data_loading() 