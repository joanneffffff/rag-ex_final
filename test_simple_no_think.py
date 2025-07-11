#!/usr/bin/env python3
"""
简化的无思考过程模板测试
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_answer_extraction():
    """测试答案提取逻辑"""
    print("🧪 测试答案提取逻辑")
    
    # 导入修改后的函数
    from comprehensive_evaluation_enhanced_new import extract_final_answer_with_rescue
    
    # 简单测试用例
    test_cases = [
        ("<answer>1200</answer>", "1200"),
        ("<answer>25%</answer>", "25%"),
        ("<answer></answer>", ""),
        ("No answer tags", ""),
        ("<think>1200</think>", "")  # 应该返回空，因为不再从think提取
    ]
    
    passed = 0
    for input_text, expected in test_cases:
        result = extract_final_answer_with_rescue(input_text)
        success = result == expected
        print(f"   {input_text} -> {result} ({'✅' if success else '❌'})")
        if success:
            passed += 1
    
    print(f"📊 答案提取测试: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)

def test_template_file():
    """测试模板文件是否存在"""
    print("\n🧪 测试模板文件")
    
    template_path = Path("alphafin_data_process/templates/unified_english_template_no_think.txt")
    if template_path.exists():
        print(f"✅ 模板文件存在: {template_path}")
        
        # 检查文件内容
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("包含<answer>标签", "<answer>" in content),
            ("不包含<think>标签", "<think>" not in content),
            ("包含示例", "Q:" in content),
            ("包含系统指令", "You are a financial" in content)
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")
            if not check_result:
                all_passed = False
        
        return all_passed
    else:
        print(f"❌ 模板文件不存在: {template_path}")
        return False

def main():
    """运行测试"""
    print("🚀 简化测试新的无思考过程模板")
    print("=" * 50)
    
    test1 = test_answer_extraction()
    test2 = test_template_file()
    
    print(f"\n📊 总结: {sum([test1, test2])}/2 测试通过")
    
    if test1 and test2:
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 