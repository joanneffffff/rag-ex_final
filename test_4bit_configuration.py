#!/usr/bin/env python3
"""
4bit 量化配置验证测试
验证所有组件都正确配置为 4bit 量化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.components.retriever.reranker import QwenReranker

def test_configuration():
    """测试配置是否正确"""
    print("🎯 4bit 量化配置验证")
    print("=" * 50)
    
    # 加载配置
    config = Config()
    
    print("📋 配置检查:")
    print(f"   - 生成器模型: {config.generator.model_name}")
    print(f"   - 生成器设备: {config.generator.device}")
    print(f"   - 生成器量化: {config.generator.use_quantization}")
    print(f"   - 生成器量化类型: {config.generator.quantization_type}")
    print(f"   - 重排序器量化: {config.reranker.use_quantization}")
    print(f"   - 重排序器量化类型: {config.reranker.quantization_type}")
    
    # 验证配置
    config_ok = True
    
    if config.generator.quantization_type != "4bit":
        print("❌ 生成器量化类型不是 4bit")
        config_ok = False
    else:
        print("✅ 生成器配置正确")
    
    if config.reranker.quantization_type != "4bit":
        print("❌ 重排序器量化类型不是 4bit")
        config_ok = False
    else:
        print("✅ 重排序器配置正确")
    
    return config_ok

def test_generator_initialization():
    """测试生成器初始化"""
    print("\n🔧 测试生成器初始化...")
    
    try:
        config = Config()
        generator = LocalLLMGenerator(
            model_name=config.generator.model_name,
            device=config.generator.device,
            use_quantization=config.generator.use_quantization,
            quantization_type=config.generator.quantization_type,
            cache_dir=config.generator.cache_dir
        )
        
        print("✅ 生成器初始化成功")
        print(f"   - 模型: {generator.model_name}")
        print(f"   - 设备: {generator.device}")
        print(f"   - 量化: {generator.use_quantization}")
        print(f"   - 量化类型: {generator.quantization_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成器初始化失败: {e}")
        return False

def test_reranker_initialization():
    """测试重排序器初始化"""
    print("\n🔧 测试重排序器初始化...")
    
    try:
        config = Config()
        reranker = QwenReranker(
            model_name=config.reranker.model_name,
            device=config.reranker.device,
            cache_dir=config.reranker.cache_dir,
            use_quantization=config.reranker.use_quantization,
            quantization_type=config.reranker.quantization_type
        )
        
        print("✅ 重排序器初始化成功")
        print(f"   - 模型: {reranker.model_name}")
        print(f"   - 设备: {reranker.device}")
        print(f"   - 量化: {config.reranker.use_quantization}")
        print(f"   - 量化类型: {config.reranker.quantization_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ 重排序器初始化失败: {e}")
        return False

def test_generation():
    """测试生成功能"""
    print("\n🔧 测试生成功能...")
    
    try:
        config = Config()
        generator = LocalLLMGenerator(
            model_name=config.generator.model_name,
            device=config.generator.device,
            use_quantization=config.generator.use_quantization,
            quantization_type=config.generator.quantization_type,
            cache_dir=config.generator.cache_dir
        )
        
        # 测试生成
        test_prompt = "请简要介绍一下金融分析的基本方法。"
        response = generator.generate([test_prompt])
        
        print("✅ 生成功能正常")
        print(f"   - 输入: {test_prompt}")
        print(f"   - 输出: {response[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成功能失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始 4bit 量化配置验证")
    print("=" * 60)
    
    # 1. 配置检查
    config_ok = test_configuration()
    
    if not config_ok:
        print("\n❌ 配置检查失败，请检查配置文件")
        return
    
    # 2. 生成器测试
    generator_ok = test_generator_initialization()
    
    # 3. 重排序器测试
    reranker_ok = test_reranker_initialization()
    
    # 4. 生成功能测试
    generation_ok = test_generation()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    if config_ok and generator_ok and reranker_ok and generation_ok:
        print("🎉 所有测试通过！")
        print("✅ 4bit 量化配置正确")
        print("✅ 生成器工作正常")
        print("✅ 重排序器工作正常")
        print("✅ 生成功能正常")
        print("\n💡 您的 Fin-R1 模型已成功配置为 4bit 量化！")
    else:
        print("❌ 部分测试失败")
        print(f"   - 配置检查: {'✅' if config_ok else '❌'}")
        print(f"   - 生成器初始化: {'✅' if generator_ok else '❌'}")
        print(f"   - 重排序器初始化: {'✅' if reranker_ok else '❌'}")
        print(f"   - 生成功能: {'✅' if generation_ok else '❌'}")

if __name__ == "__main__":
    main() 