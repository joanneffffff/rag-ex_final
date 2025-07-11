#!/usr/bin/env python3
"""
测试新的无思考过程模板和答案提取逻辑
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_answer_extraction():
    """测试答案提取逻辑"""
    print("🧪 测试答案提取逻辑")
    print("=" * 50)
    
    # 导入修改后的函数
    from comprehensive_evaluation_enhanced_new import extract_final_answer_with_rescue
    
    # 测试用例
    test_cases = [
        {
            "name": "标准<answer>标签",
            "input": "Q: What is the revenue?\n<answer>1200</answer>",
            "expected": "1200"
        },
        {
            "name": "带空格的<answer>标签",
            "input": "Q: What is the revenue?\n<answer> 1200 </answer>",
            "expected": "1200"
        },
        {
            "name": "多行<answer>内容",
            "input": "Q: What is the revenue?\n<answer>\n1200\n</answer>",
            "expected": "1200"
        },
        {
            "name": "百分比答案",
            "input": "Q: What is the growth rate?\n<answer>25%</answer>",
            "expected": "25%"
        },
        {
            "name": "负数答案",
            "input": "Q: What is the change?\n<answer>-15%</answer>",
            "expected": "-15%"
        },
        {
            "name": "无<answer>标签",
            "input": "Q: What is the revenue?\nThe revenue is 1200.",
            "expected": ""
        },
        {
            "name": "空<answer>标签",
            "input": "Q: What is the revenue?\n<answer></answer>",
            "expected": ""
        },
        {
            "name": "只有<think>标签（应该返回空）",
            "input": "Q: What is the revenue?\n<think>Let me calculate... The revenue is 1200.</think>",
            "expected": ""
        },
        {
            "name": "复杂格式答案",
            "input": "Q: What is the revenue?\n<answer>$1,200.50</answer>",
            "expected": "1200.50"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        result = extract_final_answer_with_rescue(test_case["input"])
        success = result == test_case["expected"]
        
        print(f"\n{i}. {test_case['name']}")
        print(f"   输入: {repr(test_case['input'])}")
        print(f"   期望: {repr(test_case['expected'])}")
        print(f"   实际: {repr(result)}")
        print(f"   结果: {'✅ 通过' if success else '❌ 失败'}")
        
        if success:
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    return passed == total

def test_template_loading():
    """测试模板加载"""
    print("\n🧪 测试模板加载")
    print("=" * 50)
    
    try:
        from comprehensive_evaluation_enhanced_new import load_and_format_template_with_separated_context
        
        # 测试加载新模板
        template_name = "unified_english_template_no_think.txt"
        table_context = "| Year | Revenue |\n|------|---------|\n| 2023 | 1200    |"
        text_context = "The company showed strong growth in 2023."
        query = "What is the revenue in 2023?"
        
        messages = load_and_format_template_with_separated_context(
            template_name, table_context, text_context, query
        )
        
        print("✅ 模板加载成功")
        print(f"📝 消息数量: {len(messages)}")
        
        for i, msg in enumerate(messages):
            print(f"   消息 {i+1}: {msg['role']} - {len(msg['content'])} 字符")
        
        # 检查是否包含正确的占位符替换
        content = messages[-1]['content'] if messages else ""
        if "1200" in content and "2023" in content:
            print("✅ 占位符替换正确")
            return True
        else:
            print("❌ 占位符替换可能有问题")
            return False
            
    except Exception as e:
        print(f"❌ 模板加载失败: {e}")
        return False

def test_prompt_generation():
    """测试完整提示生成"""
    print("\n🧪 测试完整提示生成")
    print("=" * 50)
    
    try:
        from comprehensive_evaluation_enhanced_new import get_final_prompt
        from comprehensive_evaluation_enhanced_new import ComprehensiveEvaluator
        
        context = "| Year | Revenue |\n|------|---------|\n| 2023 | 1200    |\n\nThe company showed strong growth in 2023."
        query = "What is the revenue in 2023?"
        
        messages = get_final_prompt(context, query)
        # 创建临时评估器来使用其转换方法
        temp_evaluator = ComprehensiveEvaluator("dummy_model", "cpu")
        prompt_text = temp_evaluator._convert_messages_to_text(messages)
        
        print("✅ 提示生成成功")
        print(f"📝 提示长度: {len(prompt_text)} 字符")
        
        # 检查关键元素
        checks = [
            ("包含<answer>标签", "<answer>" in prompt_text),
            ("包含问题", "What is the revenue" in prompt_text),
            ("包含表格数据", "1200" in prompt_text),
            ("包含ChatML格式", "<|im_start|>" in prompt_text),
            ("不包含<think>标签", "<think>" not in prompt_text)
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
    """运行所有测试"""
    print("🚀 开始测试新的无思考过程模板")
    print("=" * 60)
    
    tests = [
        ("答案提取逻辑", test_answer_extraction),
        ("模板加载", test_template_loading),
        ("提示生成", test_prompt_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！新模板和逻辑工作正常。")
        return True
    else:
        print("⚠️ 部分测试失败，需要检查问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 