#!/usr/bin/env python3
"""
测试 Fin-R1 生成器模块
验证更新后的 local_llm_generator.py 功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator

def test_fin_r1_generator():
    """测试 Fin-R1 生成器"""
    print("🚀 开始测试 Fin-R1 生成器...")
    
    try:
        # 初始化生成器，使用 Fin-R1 模型
        generator = LocalLLMGenerator(
            model_name="SUFE-AIFLM-Lab/Fin-R1",
            device="cuda:0",  # 使用 GPU
            use_quantization=True,
            quantization_type="8bit"
        )
        
        print("✅ 生成器初始化成功")
        
        # 读取 multi_stage_chinese_template.txt
        template_path = "data/prompt_templates/multi_stage_chinese_template.txt"
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        print(f"📋 模板文件长度: {len(template_content)} 字符")
        
        # 准备测试数据
        test_context = """德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。

研报显示：德赛电池发布20年业绩预告，20年营收约193.9亿元，同比增长5%，归母净利润6.3-6.9亿元，同比增长25.5%-37.4%。21年利润持续增长，源于A客户及非手机业务成长及并表比例增加。公司认为超预期主要源于iPhone 12 Pro Max新机需求佳及新品盈利能力提升。展望21年，5G iPhone周期叠加非手机业务增量，Watch、AirPods需求量增长，iPad、Mac份额提升，望驱动A客户业务成长。"""
        
        test_query = "德赛电池2021年利润持续增长的主要原因是什么？"
        
        # 格式化模板
        formatted_prompt = template_content.format(
            summary=test_context,
            context=test_context,
            query=test_query
        )
        
        print(f"📝 格式化后 Prompt 长度: {len(formatted_prompt)} 字符")
        
        # 测试 convert_to_json_chat_format
        print("\n🔧 测试 convert_to_json_chat_format...")
        json_chat = generator.convert_to_json_chat_format(formatted_prompt)
        print(f"✅ JSON 聊天格式转换成功，长度: {len(json_chat)} 字符")
        
        # 测试 convert_json_to_model_format
        print("\n🔧 测试 convert_json_to_model_format...")
        model_format = generator.convert_json_to_model_format(json_chat)
        print(f"✅ 模型格式转换成功，长度: {len(model_format)} 字符")
        
        # 测试生成
        print("\n🤖 开始生成回答...")
        responses = generator.generate([formatted_prompt])
        
        if responses:
            print(f"\n✅ 生成成功！回答长度: {len(responses[0])} 字符")
            print(f"\n📄 生成的回答:")
            print("="*50)
            print(responses[0])
            print("="*50)
        else:
            print("❌ 生成失败，未获得回答")
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fin_r1_generator() 