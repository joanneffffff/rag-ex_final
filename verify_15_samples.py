#!/usr/bin/env python3
"""
验证15个样本文件
"""

import json

def verify_samples():
    """验证15个样本文件"""
    print("🔍 验证15个样本文件")
    print("=" * 50)
    
    # 加载文件
    try:
        with open("evaluate_mrr/tatqa_test_15_samples.json", 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"✅ 成功加载文件，共 {len(samples)} 个样本")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 按类型统计
    table_count = 0
    text_count = 0
    table_text_count = 0
    
    print("\n📊 样本分布:")
    for sample in samples:
        answer_from = sample.get('answer_from', '')
        if answer_from == 'table':
            table_count += 1
        elif answer_from == 'text':
            text_count += 1
        elif answer_from == 'table-text':
            table_text_count += 1
    
    print(f"   - Table 样本: {table_count} 个")
    print(f"   - Text 样本: {text_count} 个")
    print(f"   - Table-Text 样本: {table_text_count} 个")
    print(f"   - 总计: {len(samples)} 个")
    
    # 验证是否满足要求
    if table_count == 5 and text_count == 5 and table_text_count == 5:
        print("\n✅ 样本分布符合要求！")
    else:
        print(f"\n⚠️ 样本分布不符合要求，期望各5个，实际: table={table_count}, text={text_count}, table-text={table_text_count}")
    
    # 显示样本预览
    print("\n📋 样本预览:")
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. 类型: {sample['answer_from']}")
        print(f"   问题: {sample['query'][:80]}...")
        print(f"   答案: {sample['answer']}")
        print(f"   文档ID: {sample.get('doc_id', 'N/A')}")
    
    print(f"\n🎉 验证完成！文件: evaluate_mrr/tatqa_test_15_samples.json")

if __name__ == "__main__":
    verify_samples() 