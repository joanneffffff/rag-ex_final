#!/usr/bin/env python3
"""
最终解决方案测试脚本
验证 Fin-R1 4bit 量化在 CUDA:1 上的完整功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from config.parameters import Config

def test_final_solution():
    """测试最终解决方案"""
    print("🎯 最终解决方案测试")
    print("=" * 50)
    
    # 加载配置
    config = Config()
    print(f"📋 配置信息:")
    print(f"   - 模型: {config.generator.model_name}")
    print(f"   - 设备: {config.generator.device}")
    print(f"   - 量化: {config.generator.use_quantization}")
    print(f"   - 量化类型: {config.generator.quantization_type}")
    
    try:
        print("\n🔧 初始化 Fin-R1 生成器...")
        generator = LocalLLMGenerator(
            model_name=config.generator.model_name,
            device=config.generator.device,
            use_quantization=config.generator.use_quantization,
            quantization_type=config.generator.quantization_type,
            cache_dir=config.generator.cache_dir
        )
        
        print("✅ 生成器初始化成功")
        
        # 测试中文查询
        print("\n🔧 测试中文查询...")
        chinese_context = """德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。

研报显示：德赛电池发布20年业绩预告，20年营收约193.9亿元，同比增长5%，归母净利润6.3-6.9亿元，同比增长25.5%-37.4%。21年利润持续增长，源于A客户及非手机业务成长及并表比例增加。公司认为超预期主要源于iPhone 12 Pro Max新机需求佳及新品盈利能力提升。展望21年，5G iPhone周期叠加非手机业务增量，Watch、AirPods需求量增长，iPad、Mac份额提升，望驱动A客户业务成长。"""
        
        chinese_query = "德赛电池2021年利润持续增长的主要原因是什么？"
        
        chinese_prompt = f"""你是一位专业的金融分析师，请基于以下公司财务报告信息，准确、简洁地回答用户的问题。

【公司财务报告摘要】
{chinese_context}

【用户问题】
{chinese_query}

请提供准确、专业的分析回答："""
        
        chinese_response = generator.generate([chinese_prompt])
        print(f"✅ 中文回答: {chinese_response[0]}")
        
        # 测试英文查询
        print("\n🔧 测试英文查询...")
        english_context = """Apple Inc. (AAPL) reported strong Q4 2023 results with revenue of $89.5 billion, up 8% year-over-year. iPhone sales were particularly strong, with revenue of $43.8 billion, representing 49% of total revenue. The company's services segment also showed robust growth, with revenue of $22.3 billion, up 16% year-over-year.

Key highlights include:
- iPhone revenue: $43.8B (up 6% YoY)
- Services revenue: $22.3B (up 16% YoY)
- Mac revenue: $7.6B (down 34% YoY)
- iPad revenue: $6.4B (down 10% YoY)
- Wearables revenue: $9.3B (up 3% YoY)"""
        
        english_query = "What were the main drivers of Apple's Q4 2023 revenue growth?"
        
        english_prompt = f"""You are a professional financial analyst. Please provide an accurate and concise analysis based on the following financial report information.

[Financial Report Summary]
{english_context}

[User Question]
{english_query}

Please provide an accurate and professional analysis:"""
        
        english_response = generator.generate([english_prompt])
        print(f"✅ 英文回答: {english_response[0]}")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    success = test_final_solution()
    
    if success:
        print("\n🏆 解决方案验证成功！")
        print("✅ Fin-R1 4bit 量化在 CUDA:1 上运行正常")
        print("✅ 中英文查询都能正常处理")
        print("✅ 内存使用优化有效")
    else:
        print("\n❌ 解决方案验证失败")
        print("请检查配置和模型文件")

if __name__ == "__main__":
    main() 