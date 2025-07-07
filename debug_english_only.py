#!/usr/bin/env python3
"""
专门检查英文数据加载问题的脚本
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def debug_english_only():
    """专门诊断英文数据加载问题"""
    print("=== 英文数据加载问题诊断 ===")
    
    # 1. 检查配置文件中的英文数据路径
    config = Config()
    english_data_path = config.data.english_data_path
    print(f"1. 英文数据路径: {english_data_path}")
    
    # 2. 检查文件是否存在和大小
    if not Path(english_data_path).exists():
        print(f"❌ 英文数据文件不存在")
        return
    
    file_size = Path(english_data_path).stat().st_size
    print(f"✅ 文件存在，大小: {file_size / 1024 / 1024:.2f} MB")
    
    # 3. 检查文件内容结构
    print(f"\n2. 检查文件内容结构...")
    try:
        with open(english_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   总行数: {len(lines)}")
            
            # 检查前3行的JSON结构
            for i in range(min(3, len(lines))):
                try:
                    item = json.loads(lines[i].strip())
                    context = item.get('context', '')
                    text = item.get('text', '')
                    content = item.get('content', '')
                    
                    print(f"   第{i+1}行:")
                    print(f"     - context字段长度: {len(context)}")
                    print(f"     - text字段长度: {len(text)}")
                    print(f"     - content字段长度: {len(content)}")
                    
                    # 显示实际使用的字段内容
                    actual_content = context or text or content
                    if actual_content:
                        print(f"     - 实际内容预览: {actual_content[:100]}...")
                    else:
                        print(f"     - ⚠️ 所有内容字段都为空")
                        
                except json.JSONDecodeError as e:
                    print(f"   第{i+1}行 - JSON解析失败: {e}")
                    
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 4. 使用DualLanguageLoader加载英文数据
    print(f"\n3. 使用DualLanguageLoader加载英文数据...")
    data_loader = DualLanguageLoader()
    
    try:
        english_docs = data_loader.load_tatqa_context_only(english_data_path)
        print(f"   加载的英文文档数量: {len(english_docs)}")
        
        if english_docs:
            # 检查第一个文档的详细信息
            first_doc = english_docs[0]
            print(f"   第一个文档:")
            print(f"     - 类型: {type(first_doc)}")
            print(f"     - content长度: {len(first_doc.content)}")
            print(f"     - content预览: {first_doc.content[:100]}...")
            print(f"     - metadata.language: {first_doc.metadata.language}")
            print(f"     - metadata.source: {first_doc.metadata.source}")
            
            # 统计语言分布
            languages = {}
            for doc in english_docs:
                lang = doc.metadata.language
                languages[lang] = languages.get(lang, 0) + 1
            
            print(f"   语言分布: {languages}")
            
            # 检查是否有非英文文档
            non_english = {lang: count for lang, count in languages.items() if lang != 'english'}
            if non_english:
                print(f"   ⚠️ 发现非英文文档: {non_english}")
            
        else:
            print("   ❌ 没有加载到任何英文文档")
            
    except Exception as e:
        print(f"❌ DualLanguageLoader加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 测试语言检测功能
    print(f"\n4. 测试语言检测功能...")
    test_texts = [
        "What is the revenue of the company?",
        "The financial statements show quarterly results",
        "Table 1 presents the market analysis",
        "Paragraph 2 discusses the investment strategy"
    ]
    
    for i, text in enumerate(test_texts):
        try:
            detected_lang = data_loader.detect_language(text)
            print(f"   测试{i+1}: '{text[:30]}...' -> {detected_lang}")
        except Exception as e:
            print(f"   测试{i+1}: 检测失败 - {e}")

if __name__ == "__main__":
    debug_english_only() 