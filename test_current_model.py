#!/usr/bin/env python3
"""
测试当前使用的LLM生成器模型
"""

from config.parameters import config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_current_model():
    """测试当前使用的模型配置"""
    print("=== 当前LLM生成器模型配置 ===\n")
    
    # 1. 显示配置文件中的设置
    print("1. 配置文件中的模型设置:")
    print(f"   - 模型名称: {config.generator.model_name}")
    print(f"   - 设备: {config.generator.device}")
    print(f"   - 量化: {config.generator.use_quantization}")
    print(f"   - 量化类型: {config.generator.quantization_type}")
    print(f"   - 缓存目录: {config.generator.cache_dir}")
    print(f"   - 最大新token数: {config.generator.max_new_tokens}")
    print(f"   - 温度: {config.generator.temperature}")
    print(f"   - Top-p: {config.generator.top_p}")
    print()
    
    # 2. 检查是否是Fin-R1模型
    is_fin_r1 = "Fin-R1" in config.generator.model_name
    print("2. 模型类型检查:")
    print(f"   - 是否使用Fin-R1: {is_fin_r1}")
    print(f"   - 模型描述: {'上海财经大学金融推理大模型，专门针对金融领域优化' if is_fin_r1 else '其他模型'}")
    print()
    
    # 3. 尝试初始化生成器（不加载模型，只检查配置）
    print("3. 生成器配置验证:")
    try:
        # 创建一个简化的生成器实例来验证配置
        generator = LocalLLMGenerator()
        print(f"   - 生成器初始化成功")
        print(f"   - 实际使用的模型: {generator.model_name}")
        print(f"   - 实际使用的设备: {generator.device}")
        print(f"   - 实际使用的量化: {generator.use_quantization}")
        print(f"   - 实际使用的量化类型: {generator.quantization_type}")
    except Exception as e:
        print(f"   - 生成器初始化失败: {e}")
    print()
    
    # 4. 总结
    print("4. 总结:")
    if is_fin_r1:
        print("   ✅ 当前配置使用Fin-R1模型")
        print("   ✅ 这是专门针对金融领域优化的模型")
        print("   ✅ 适合处理金融相关的查询和回答")
    else:
        print("   ❌ 当前配置未使用Fin-R1模型")
        print("   💡 如需使用Fin-R1，请检查配置文件")

if __name__ == "__main__":
    test_current_model() 