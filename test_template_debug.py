#!/usr/bin/env python3
"""
调试模板问题
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_template_loading():
    """测试模板加载和格式化"""
    print("🧪 测试模板加载")
    
    try:
        from comprehensive_evaluation_enhanced_new import load_and_format_template_with_separated_context
        
        # 测试数据
        table_context = "| Year | Revenue |\n|------|---------|\n| 2023 | 1200    |"
        text_context = "The company showed strong growth in 2023."
        query = "What is the revenue in 2023?"
        
        messages = load_and_format_template_with_separated_context(
            "unified_english_template_no_think.txt", 
            table_context, 
            text_context, 
            query
        )
        
        print(f"✅ 模板加载成功，消息数量: {len(messages)}")
        
        for i, msg in enumerate(messages):
            print(f"\n消息 {i+1} ({msg['role']}):")
            print(f"长度: {len(msg['content'])} 字符")
            print(f"内容预览: {msg['content'][:200]}...")
            
            # 检查关键元素
            checks = [
                ("包含问题", query in msg['content']),
                ("包含表格数据", "1200" in msg['content']),
                ("包含<answer>标签", "<answer>" in msg['content']),
                ("不包含<think>标签", "<think>" not in msg['content'])
            ]
            
            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"   {status} {check_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模板加载失败: {e}")
        return False

def test_prompt_generation():
    """测试完整提示生成"""
    print("\n🧪 测试完整提示生成")
    
    try:
        from comprehensive_evaluation_enhanced_new import get_final_prompt, ComprehensiveEvaluator
        
        # 测试数据
        context = "| Year | Revenue |\n|------|---------|\n| 2023 | 1200    |\n\nThe company showed strong growth in 2023."
        query = "What is the revenue in 2023?"
        
        messages = get_final_prompt(context, query)
        
        print(f"✅ 提示生成成功，消息数量: {len(messages)}")
        
        # 创建临时评估器来转换消息
        temp_evaluator = ComprehensiveEvaluator("dummy_model", "cpu")
        prompt_text = temp_evaluator._convert_messages_to_text(messages)
        
        print(f"📝 最终提示长度: {len(prompt_text)} 字符")
        print(f"📝 提示预览: {prompt_text[:500]}...")
        
        # 检查关键元素
        checks = [
            ("包含问题", query in prompt_text),
            ("包含表格数据", "1200" in prompt_text),
            ("包含<answer>标签", "<answer>" in prompt_text),
            ("不包含<think>标签", "<think>" not in prompt_text),
            ("包含ChatML格式", "<|im_start|>" in prompt_text)
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")
            if not check_result:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 提示生成失败: {e}")
        return False

def main():
    """运行调试测试"""
    print("🚀 调试模板问题")
    print("=" * 50)
    
    test1 = test_template_loading()
    test2 = test_prompt_generation()
    
    print(f"\n📊 总结: {sum([test1, test2])}/2 测试通过")
    
    if test1 and test2:
        print("🎉 模板工作正常，问题可能在其他地方")
    else:
        print("⚠️ 模板有问题，需要修复")

if __name__ == "__main__":
    main() 