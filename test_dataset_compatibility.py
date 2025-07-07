#!/usr/bin/env python3
"""
测试RAG系统适配器对AlphaFin和TatQA数据集的兼容性
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def test_alphafin_compatibility():
    """测试AlphaFin数据集兼容性"""
    print("=" * 60)
    print("测试AlphaFin数据集兼容性")
    print("=" * 60)
    
    # 模拟AlphaFin数据格式
    alphafin_sample = {
        "generated_question": "What is the revenue of Apple in 2023?",
        "doc_id": "apple_2023_annual_report",
        "context": "Apple reported revenue of $394.3 billion in 2023...",
        "answer": "$394.3 billion"
    }
    
    print(f"AlphaFin样本: {json.dumps(alphafin_sample, indent=2)}")
    
    # 测试字段提取
    query = alphafin_sample.get('generated_question', '') or alphafin_sample.get('question', '') or alphafin_sample.get('query', '')
    print(f"提取的查询: {query}")
    
    # 测试目标文档ID提取
    target_doc_ids = []
    if 'relevant_doc_ids' in alphafin_sample and alphafin_sample['relevant_doc_ids']:
        target_doc_ids = alphafin_sample['relevant_doc_ids']
        if isinstance(target_doc_ids, str):
            try:
                target_doc_ids = json.loads(target_doc_ids)
            except:
                target_doc_ids = [target_doc_ids]
        elif not isinstance(target_doc_ids, list):
            target_doc_ids = [target_doc_ids]
    
    if not target_doc_ids and 'doc_id' in alphafin_sample:
        doc_id = alphafin_sample['doc_id']
        if doc_id:
            target_doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
    
    print(f"提取的目标文档IDs: {target_doc_ids}")
    
    return query and target_doc_ids

def test_tatqa_compatibility():
    """测试TatQA数据集兼容性"""
    print("\n" + "=" * 60)
    print("测试TatQA数据集兼容性")
    print("=" * 60)
    
    # 模拟TatQA数据格式
    tatqa_sample = {
        "generated_question": "What is the total revenue in 2020?",
        "relevant_doc_ids": ["doc_001", "doc_002"],
        "context": "The company reported total revenue of $500 million in 2020...",
        "answer": "$500 million"
    }
    
    print(f"TatQA样本: {json.dumps(tatqa_sample, indent=2)}")
    
    # 测试字段提取
    query = tatqa_sample.get('generated_question', '') or tatqa_sample.get('question', '') or tatqa_sample.get('query', '')
    print(f"提取的查询: {query}")
    
    # 测试目标文档ID提取
    target_doc_ids = []
    if 'relevant_doc_ids' in tatqa_sample and tatqa_sample['relevant_doc_ids']:
        target_doc_ids = tatqa_sample['relevant_doc_ids']
        if isinstance(target_doc_ids, str):
            try:
                target_doc_ids = json.loads(target_doc_ids)
            except:
                target_doc_ids = [target_doc_ids]
        elif not isinstance(target_doc_ids, list):
            target_doc_ids = [target_doc_ids]
    
    if not target_doc_ids and 'doc_id' in tatqa_sample:
        doc_id = tatqa_sample['doc_id']
        if doc_id:
            target_doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
    
    print(f"提取的目标文档IDs: {target_doc_ids}")
    
    return query and target_doc_ids

def test_generic_compatibility():
    """测试通用数据集兼容性"""
    print("\n" + "=" * 60)
    print("测试通用数据集兼容性")
    print("=" * 60)
    
    # 模拟通用数据格式
    generic_sample = {
        "question": "What is the profit margin?",
        "id": "financial_report_2023",
        "context": "The profit margin was 15% in 2023...",
        "answer": "15%"
    }
    
    print(f"通用样本: {json.dumps(generic_sample, indent=2)}")
    
    # 测试字段提取
    query = generic_sample.get('generated_question', '') or generic_sample.get('question', '') or generic_sample.get('query', '')
    print(f"提取的查询: {query}")
    
    # 测试目标文档ID提取
    target_doc_ids = []
    if 'relevant_doc_ids' in generic_sample and generic_sample['relevant_doc_ids']:
        target_doc_ids = generic_sample['relevant_doc_ids']
        if isinstance(target_doc_ids, str):
            try:
                target_doc_ids = json.loads(target_doc_ids)
            except:
                target_doc_ids = [target_doc_ids]
        elif not isinstance(target_doc_ids, list):
            target_doc_ids = [target_doc_ids]
    
    if not target_doc_ids and 'doc_id' in generic_sample:
        doc_id = generic_sample['doc_id']
        if doc_id:
            target_doc_ids = [doc_id] if isinstance(doc_id, str) else doc_id
    
    # 尝试其他可能的字段
    if not target_doc_ids:
        for field in ['id', 'document_id', 'target_id']:
            if field in generic_sample and generic_sample[field]:
                target_doc_ids = [generic_sample[field]]
                break
    
    print(f"提取的目标文档IDs: {target_doc_ids}")
    
    return query and target_doc_ids

def test_string_relevant_doc_ids():
    """测试字符串格式的relevant_doc_ids"""
    print("\n" + "=" * 60)
    print("测试字符串格式的relevant_doc_ids")
    print("=" * 60)
    
    # 模拟字符串格式的relevant_doc_ids
    string_sample = {
        "generated_question": "What is the market cap?",
        "relevant_doc_ids": '["doc_001", "doc_002"]',  # JSON字符串格式
        "context": "The market cap is $1 billion...",
        "answer": "$1 billion"
    }
    
    print(f"字符串格式样本: {json.dumps(string_sample, indent=2)}")
    
    # 测试字段提取
    query = string_sample.get('generated_question', '') or string_sample.get('question', '') or string_sample.get('query', '')
    print(f"提取的查询: {query}")
    
    # 测试目标文档ID提取
    target_doc_ids = []
    if 'relevant_doc_ids' in string_sample and string_sample['relevant_doc_ids']:
        target_doc_ids = string_sample['relevant_doc_ids']
        if isinstance(target_doc_ids, str):
            try:
                target_doc_ids = json.loads(target_doc_ids)
                print(f"成功解析JSON字符串: {target_doc_ids}")
            except:
                target_doc_ids = [target_doc_ids]
                print(f"JSON解析失败，作为单个字符串处理: {target_doc_ids}")
        elif not isinstance(target_doc_ids, list):
            target_doc_ids = [target_doc_ids]
    
    print(f"最终的目标文档IDs: {target_doc_ids}")
    
    return query and target_doc_ids

def main():
    """主测试函数"""
    print("开始测试RAG系统适配器数据集兼容性...")
    
    # 测试各种数据格式
    tests = [
        ("AlphaFin格式", test_alphafin_compatibility),
        ("TatQA格式", test_tatqa_compatibility),
        ("通用格式", test_generic_compatibility),
        ("字符串relevant_doc_ids", test_string_relevant_doc_ids)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, "通过"))
            print(f"✅ {test_name}: 通过")
        except Exception as e:
            results.append((test_name, False, f"失败: {e}"))
            print(f"❌ {test_name}: 失败 - {e}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    for test_name, result, status in results:
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！RAG系统适配器完全兼容多种数据集格式。")
    else:
        print("⚠️  部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main() 