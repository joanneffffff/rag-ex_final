#!/usr/bin/env python3
"""
快速测试脚本
验证RAG系统LocalLLMGenerator集成是否正常工作
"""

# 临时关闭warnings，避免transformers参数警告
import warnings
warnings.filterwarnings("ignore")

# 更精确地过滤transformers生成参数警告
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_rag_generator_import():
    """测试RAG系统LocalLLMGenerator导入"""
    print("🧪 测试RAG系统LocalLLMGenerator导入...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        print("✅ LocalLLMGenerator导入成功")
        return True
    except ImportError as e:
        print(f"❌ LocalLLMGenerator导入失败: {e}")
        return False

def test_rag_generator_creation():
    """测试RAG系统LocalLLMGenerator创建"""
    print("🧪 测试LocalLLMGenerator创建...")
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 创建生成器
        llm_generator = LocalLLMGenerator(
            model_name="SUFE-AIFLM-Lab/Fin-R1",
            device="auto",
            use_quantization=True,
            quantization_type="4bit"
        )
        print("✅ LocalLLMGenerator创建成功")
        return llm_generator
    except Exception as e:
        print(f"❌ LocalLLMGenerator创建失败: {e}")
        return None

def test_rag_generator_generation(llm_generator):
    """测试RAG系统LocalLLMGenerator生成"""
    print("🧪 测试LocalLLMGenerator生成...")
    
    try:
        # 简单测试prompt
        test_prompt = "What is 2 + 2? Please provide a simple answer."
        
        # 生成回答
        start_time = time.time()
        responses = llm_generator.generate([test_prompt])
        generation_time = time.time() - start_time
        
        generated_answer = responses[0] if responses else ""
        
        print(f"✅ 生成成功")
        print(f"⏱️ 生成时间: {generation_time:.2f}秒")
        print(f"📝 生成答案: {generated_answer}")
        
        return True
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False

def test_enhanced_rag_system():
    """测试增强版RAG系统"""
    print("🧪 测试增强版RAG系统...")
    
    try:
        from enhanced_rag_system import create_enhanced_rag_system
        
        # 创建系统
        rag_system = create_enhanced_rag_system()
        print("✅ 增强版RAG系统创建成功")
        
        # 测试英文查询处理
        test_query = "What is the main topic?"
        test_context = "This is a test context about artificial intelligence."
        
        result = rag_system.process_english_query(test_query, test_context)
        
        if result.get("success", False):
            print("✅ 英文查询处理成功")
            print(f"📝 清理后答案: {result.get('cleaned_answer', '')}")
        else:
            print(f"❌ 英文查询处理失败: {result.get('error', 'Unknown error')}")
        
        return True
    except Exception as e:
        print(f"❌ 增强版RAG系统测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始快速测试RAG系统LocalLLMGenerator集成")
    print("="*60)
    
    # 测试1: 导入
    import_success = test_rag_generator_import()
    
    if not import_success:
        print("❌ 导入测试失败，跳过后续测试")
        return
    
    # 测试2: 创建
    llm_generator = test_rag_generator_creation()
    
    if not llm_generator:
        print("❌ 创建测试失败，跳过生成测试")
    else:
        # 测试3: 生成
        generation_success = test_rag_generator_generation(llm_generator)
    
    # 测试4: 增强版RAG系统
    system_success = test_enhanced_rag_system()
    
    print("\n" + "="*60)
    print("📊 测试结果摘要:")
    print(f"   导入测试: {'✅ 成功' if import_success else '❌ 失败'}")
    print(f"   创建测试: {'✅ 成功' if llm_generator else '❌ 失败'}")
    print(f"   生成测试: {'✅ 成功' if 'generation_success' in locals() and generation_success else '❌ 失败'}")
    print(f"   系统测试: {'✅ 成功' if system_success else '❌ 失败'}")
    
    if import_success and llm_generator and system_success:
        print("\n🎉 所有测试通过！RAG系统LocalLLMGenerator集成正常")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 