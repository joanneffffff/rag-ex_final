#!/usr/bin/env python3
"""
快速测试上下文分离功能
使用少量 TATQA 样本来验证集成效果
"""

import json
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_samples():
    """创建测试样本"""
    samples = []
    
    # 样本1：纯文本数据
    samples.append({
        "query": "What method did the company use when Topic 606 in fiscal 2019 was adopted?",
        "context": "Paragraph ID: 4202457313786d975b89fabc695c3efb\nWe utilized a comprehensive approach to evaluate and document the impact of the guidance on our current accounting policies and practices in order to identify material differences, if any, that would result from applying the new requirements to our revenue contracts. We did not identify any material differences resulting from applying the new requirements to our revenue contracts. In addition, we did not identify any significant changes to our business processes, systems, and controls to support recognition and disclosure requirements under the new guidance. We adopted the provisions of Topic 606 in fiscal 2019 utilizing the modified retrospective method. We recorded a $0.5 million cumulative effect adjustment, net of tax, to the opening balance of fiscal 2019 retained earnings, a decrease to receivables of $7.6 million, an increase to inventories of $2.8 million, an increase to prepaid expenses and other current assets of $6.9 million, an increase to other accrued liabilities of $1.4 million, and an increase to other noncurrent liabilities of $0.2 million. The adjustments primarily related to the timing of recognition of certain customer charges, trade promotional expenditures, and volume discounts.",
        "answer": "the modified retrospective method",
        "answer_from": "text"
    })
    
    # 样本2：混合数据（1个 Table ID + 多个 Paragraph ID）
    samples.append({
        "query": "What are the sales figures for Drinkable Kefir in 2019?",
        "context": """Table ID: 991d23d7-f32d-4954-8e1d-87ad22470fcf
Headers: 2019 | 2018
In thousands:  is $; 2019 is %;  is $; 2018 is %
Drinkable Kefir other than ProBugs:  is $ 71,822; 2019 is 77%;  is $ 78,523; 2018 is 76%
Cheese:  is $11,459; 2019 is 12%;  is $11,486; 2018 is 11%

Paragraph ID: a4d3952f-4390-4ab2-b6f3-460d14653c10
Drinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types, including low fat, non-fat, whole milk, protein, and BioKefir (a 3.5 oz. kefir with additional probiotic cultures).

Paragraph ID: d623137a-e787-4204-952a-af9d4ed3a2db
European-style soft cheeses, including farmer cheese in resealable cups.""",
        "answer": "71,822",
        "answer_from": "table"
    })
    
    # 样本3：纯表格数据
    samples.append({
        "query": "What is the rate of inflation in 2019?",
        "context": """Table ID: e78f8b29-6085-43de-b32f-be1a68641be3
Headers: 2019 % | 2018 % | 2017 %
Rate of inflation2: 2019 % is $2.9; 2018 % is $2.9; 2017 % is $3.0""",
        "answer": "2.9",
        "answer_from": "table"
    })
    
    return samples

def test_context_separation():
    """测试上下文分离功能"""
    print("🧪 快速测试上下文分离功能")
    print("=" * 60)
    
    # 1. 测试导入
    print("📦 测试导入...")
    try:
        from xlm.utils.context_separator import context_separator
        print("✅ 上下文分离器导入成功")
    except ImportError as e:
        print(f"❌ 上下文分离器导入失败: {e}")
        return
    
    try:
        from comprehensive_evaluation_enhanced import get_final_prompt, hybrid_decision
        print("✅ comprehensive_evaluation_enhanced 函数导入成功")
    except ImportError as e:
        print(f"❌ comprehensive_evaluation_enhanced 导入失败: {e}")
        return
    
    print()
    
    # 2. 测试样本
    samples = create_test_samples()
    
    for i, sample in enumerate(samples, 1):
        print(f"🔍 测试样本 {i}: {sample['answer_from']} 类型")
        print(f"问题: {sample['query']}")
        print(f"期望答案: {sample['answer']}")
        
        try:
            # 测试混合决策
            decision = hybrid_decision(sample['context'], sample['query'])
            print(f"混合决策: {decision}")
            
            # 测试上下文分离
            separated = context_separator.separate_context(sample['context'])
            print(f"上下文类型: {separated.context_type}")
            print(f"表格行数: {separated.metadata.get('table_lines_count', 0)}")
            print(f"文本行数: {separated.metadata.get('text_lines_count', 0)}")
            
            # 测试 prompt 生成
            messages = get_final_prompt(sample['context'], sample['query'])
            print(f"✅ Prompt 生成成功，消息数量: {len(messages)}")
            
            # 检查是否使用了分离的上下文
            user_content = messages[1]['content'] if len(messages) > 1 else ""
            if "Table Context:" in user_content and "Text Context:" in user_content:
                print("✅ 使用分离的上下文格式")
            else:
                print("⚠️ 使用原始上下文格式")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        
        print("-" * 40)
    
    print("🎉 快速测试完成！")

def save_test_data():
    """保存测试数据到文件"""
    samples = create_test_samples()
    
    output_file = "test_context_separation_samples.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 测试数据已保存到: {output_file}")
    print(f"📊 包含 {len(samples)} 个测试样本")
    
    return output_file

def main():
    """主函数"""
    print("🚀 上下文分离功能快速测试")
    print("=" * 60)
    print()
    
    # 1. 测试上下文分离功能
    test_context_separation()
    
    # 2. 保存测试数据
    print("\n💾 保存测试数据...")
    test_file = save_test_data()
    
    print(f"\n📋 测试命令:")
    print(f"python comprehensive_evaluation_enhanced.py --data_path {test_file} --sample_size 3")
    print(f"\n这将使用 {len(create_test_samples())} 个测试样本验证上下文分离功能。")

if __name__ == "__main__":
    main() 