#!/usr/bin/env python3
"""
调试LLM生成器是否正确加载了修复代码
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def debug_llm_generator():
    """调试LLM生成器"""
    print("=== 调试LLM生成器 ===")
    
    try:
        # 导入LLM生成器
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 检查是否有_fix_company_name_translation方法
        if hasattr(LocalLLMGenerator, '_fix_company_name_translation'):
            print("✅ LocalLLMGenerator 包含 _fix_company_name_translation 方法")
            
            # 检查方法内容
            method = getattr(LocalLLMGenerator, '_fix_company_name_translation')
            if callable(method):
                print("✅ _fix_company_name_translation 是可调用的方法")
            else:
                print("❌ _fix_company_name_translation 不是可调用的方法")
        else:
            print("❌ LocalLLMGenerator 缺少 _fix_company_name_translation 方法")
        
        # 检查_clean_response方法是否调用了修正方法
        if hasattr(LocalLLMGenerator, '_clean_response'):
            print("✅ LocalLLMGenerator 包含 _clean_response 方法")
            
            # 获取方法源码（如果可能）
            import inspect
            try:
                source = inspect.getsource(LocalLLMGenerator._clean_response)
                if '_fix_company_name_translation' in source:
                    print("✅ _clean_response 方法调用了 _fix_company_name_translation")
                else:
                    print("❌ _clean_response 方法没有调用 _fix_company_name_translation")
            except Exception as e:
                print(f"⚠️  无法获取方法源码: {e}")
        else:
            print("❌ LocalLLMGenerator 缺少 _clean_response 方法")
        
        # 测试实例化（不加载模型）
        print("\n=== 测试实例化 ===")
        try:
            # 创建一个不加载模型的实例
            generator = LocalLLMGenerator(
                model_name="test",
                device="cpu"
            )
            print("✅ LocalLLMGenerator 实例化成功")
            
            # 测试修正方法
            test_text = "德赛 battery (00) 的业绩表现良好"
            if hasattr(generator, '_fix_company_name_translation'):
                fixed_text = generator._fix_company_name_translation(test_text)
                print(f"测试修正:")
                print(f"  原始: {test_text}")
                print(f"  修正后: {fixed_text}")
                
                if "德赛电池" in fixed_text:
                    print("✅ 修正方法工作正常")
                else:
                    print("❌ 修正方法没有正常工作")
            else:
                print("❌ 实例缺少 _fix_company_name_translation 方法")
                
        except Exception as e:
            print(f"❌ 实例化失败: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()

def test_current_system():
    """测试当前系统是否使用了修复后的代码"""
    print("\n=== 测试当前系统 ===")
    
    try:
        # 导入多阶段检索系统
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        print("✅ 成功导入 MultiStageRetrievalSystem")
        
        # 检查是否包含LLM生成器
        if hasattr(MultiStageRetrievalSystem, 'llm_generator'):
            print("✅ MultiStageRetrievalSystem 包含 llm_generator 属性")
        else:
            print("❌ MultiStageRetrievalSystem 缺少 llm_generator 属性")
        
        # 检查初始化方法
        if hasattr(MultiStageRetrievalSystem, '__init__'):
            print("✅ MultiStageRetrievalSystem 包含 __init__ 方法")
            
            # 获取初始化方法源码
            import inspect
            try:
                source = inspect.getsource(MultiStageRetrievalSystem.__init__)
                if 'LocalLLMGenerator' in source:
                    print("✅ __init__ 方法使用了 LocalLLMGenerator")
                else:
                    print("❌ __init__ 方法没有使用 LocalLLMGenerator")
            except Exception as e:
                print(f"⚠️  无法获取初始化方法源码: {e}")
        else:
            print("❌ MultiStageRetrievalSystem 缺少 __init__ 方法")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_generator()
    test_current_system() 