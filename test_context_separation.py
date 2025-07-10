#!/usr/bin/env python3
"""
测试context分离功能
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_context_separation():
    """测试context分离功能"""
    print("🧪 测试context分离功能...")
    
    try:
        from xlm.utils.context_separator import context_separator
        print("✅ 成功导入context_separator")
    except ImportError as e:
        print(f"❌ 无法导入context_separator: {e}")
        return False
    
    # 测试数据
    test_context = """
    Table ID: test_table_123
    Headers: Year | Revenue | Profit
    Row 1: 2023 | $1000 | $200
    Row 2: 2024 | $1200 | $250
    
    The company reported strong growth in 2023 and 2024. The revenue increased by 20% year-over-year.
    """
    
    test_query = "What was the revenue in 2024?"
    
    try:
        # 测试分离功能
        print("\n📊 测试context分离...")
        separated = context_separator.separate_context(test_context)
        print(f"✅ 分离成功: {type(separated)}")
        
        # 测试格式化功能
        print("\n📝 测试prompt格式化...")
        prompt_params = context_separator.format_for_prompt(separated, test_query)
        print(f"✅ 格式化成功")
        print(f"   Table Context: {prompt_params['table_context'][:100]}...")
        print(f"   Text Context: {prompt_params['text_context'][:100]}...")
        
        # 测试模板加载
        print("\n📋 测试模板加载...")
        from comprehensive_evaluation_enhanced_new import load_and_format_template_with_separated_context
        
        messages = load_and_format_template_with_separated_context(
            'unified_english_template.txt',
            prompt_params["table_context"],
            prompt_params["text_context"],
            test_query
        )
        
        print(f"✅ 模板加载成功，生成 {len(messages)} 条消息")
        for i, msg in enumerate(messages):
            print(f"   消息 {i+1}: {msg['role']} - {len(msg['content'])} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_functionality():
    """测试回退功能"""
    print("\n🔄 测试回退功能...")
    
    try:
        from comprehensive_evaluation_enhanced_new import get_final_prompt
        
        # 测试统一上下文
        test_context = "This is a simple test context with some information."
        test_query = "What is the test about?"
        
        messages = get_final_prompt(test_context, test_query)
        print(f"✅ 回退功能正常，生成 {len(messages)} 条消息")
        
        return True
        
    except Exception as e:
        print(f"❌ 回退功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试context分离功能")
    print("="*50)
    
    # 测试1: context分离
    test1_passed = test_context_separation()
    
    # 测试2: 回退功能
    test2_passed = test_fallback_functionality()
    
    print("\n" + "="*50)
    print("📋 测试结果总结:")
    print(f"   Context分离测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"   回退功能测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！context分离功能正常工作。")
    else:
        print("\n⚠️ 部分测试失败，请检查相关功能。") 