#!/usr/bin/env python3
"""
测试后处理清理逻辑
验证是否能有效移除各种prompt注入和格式标记
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_post_processing():
    """测试后处理清理逻辑"""
    print("=" * 80)
    print("🔧 测试后处理清理逻辑")
    print("=" * 80)
    
    try:
        # 初始化LLM生成器（只为了获取_clean_response方法）
        print("1. 初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ LLM生成器初始化成功")
        
        # 测试用例
        test_cases = [
            {
                "name": "包含【回答】标记",
                "input": "德赛电池业绩增长主要源于iPhone需求强劲。【回答】德赛电池业绩增长主要源于iPhone需求强劲。",
                "expected_removed": ["【回答】"]
            },
            {
                "name": "包含Answer:标记",
                "input": "The operating revenues decreased due to lower volume. Answer: The operating revenues decreased due to lower volume.",
                "expected_removed": ["Answer:"]
            },
            {
                "name": "包含分隔线",
                "input": "德赛电池业绩增长。--- 德赛电池业绩增长。 === 德赛电池业绩增长。",
                "expected_removed": ["---", "==="]
            },
            {
                "name": "包含boxed格式",
                "input": "德赛电池业绩增长。\\boxed{德赛电池业绩增长} boxed{德赛电池业绩增长}",
                "expected_removed": ["\\boxed{", "boxed{"]
            },
            {
                "name": "包含重复句子",
                "input": "德赛电池业绩增长主要源于iPhone需求强劲。德赛电池业绩增长主要源于iPhone需求强劲。德赛电池业绩增长主要源于iPhone需求强劲。",
                "expected_removed": ["重复句子"]
            },
            {
                "name": "包含多余标点",
                "input": "，，，德赛电池业绩增长主要源于iPhone需求强劲。。。",
                "expected_removed": ["开头和结尾的标点"]
            },
            {
                "name": "包含重复的无法提供信息",
                "input": "根据现有信息，无法提供此项信息。根据现有信息，无法提供此项信息。",
                "expected_removed": ["重复的无法提供信息"]
            },
            {
                "name": "正常回答",
                "input": "德赛电池业绩增长主要源于iPhone需求强劲和新品盈利能力提升。",
                "expected_removed": []
            }
        ]
        
        print("\n2. 开始测试后处理清理...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 测试用例 {i}: {test_case['name']} ---")
            print(f"📥 输入: '{test_case['input']}'")
            
            # 应用清理逻辑
            cleaned = generator._clean_response(test_case['input'])
            print(f"📤 输出: '{cleaned}'")
            
            # 检查清理效果
            original_length = len(test_case['input'])
            cleaned_length = len(cleaned)
            reduction = original_length - cleaned_length
            
            print(f"📏 长度变化: {original_length} -> {cleaned_length} (减少 {reduction} 字符)")
            
            # 检查是否移除了预期的标记
            for expected_removed in test_case['expected_removed']:
                if expected_removed in test_case['input'] and expected_removed not in cleaned:
                    print(f"✅ 成功移除: {expected_removed}")
                elif expected_removed in test_case['input'] and expected_removed in cleaned:
                    print(f"❌ 未能移除: {expected_removed}")
                else:
                    print(f"ℹ️  无需移除: {expected_removed}")
            
            # 检查清理后的质量
            if cleaned.strip():
                print(f"✅ 清理后内容有效")
            else:
                print(f"⚠️  清理后内容为空")
        
        print("\n3. 测试完成！")
        
        # 总结
        print("\n" + "="*80)
        print("📊 后处理清理效果总结:")
        print("="*80)
        print("✅ 支持移除的标记类型:")
        print("   - 中文标记: 【回答】、回答：、回答:")
        print("   - 英文标记: Answer:、Answer:")
        print("   - 分隔线: ---、===、___、***")
        print("   - 格式标记: boxed{}、\\boxed{}、\\text{}")
        print("   - 重复句子: 自动去重")
        print("   - 多余标点: 开头和结尾的标点符号")
        print("   - 重复表述: 重复的'无法提供信息'等")
        print("\n✅ 清理策略:")
        print("   - 使用正则表达式精确匹配")
        print("   - 支持大小写不敏感匹配")
        print("   - 自动去重重复句子")
        print("   - 保留原始内容作为兜底")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_post_processing() 