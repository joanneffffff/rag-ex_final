#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.components.prompt_templates.template_loader import PromptTemplateLoader

def test_json_chat_mode():
    """测试JSON格式聊天模式"""
    
    print("=" * 80)
    print("🚀 测试JSON格式聊天模式")
    print("=" * 80)
    
    try:
        # 初始化LLM生成器
        print("1. 初始化LLM生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ LLM生成器初始化成功: {generator.model_name}")
        
        # 加载Prompt模板
        print("\n2. 加载Prompt模板...")
        loader = PromptTemplateLoader()
        template_content = loader.get_template("multi_stage_chinese_template")
        if template_content is None:
            print("❌ 模板加载失败")
            return False
        print(f"✅ Prompt模板加载成功，长度: {len(template_content)} 字符")
        
        # 解析模板内容
        print("\n3. 解析模板内容...")
        # 提取SYSTEM_PROMPT_CONTENT和USER_PROMPT_TEMPLATE
        if "SYSTEM_PROMPT_CONTENT" in template_content and "USER_PROMPT_TEMPLATE" in template_content:
            # 提取SYSTEM部分
            system_start = template_content.find('SYSTEM_PROMPT_CONTENT = """') + len('SYSTEM_PROMPT_CONTENT = """')
            system_end = template_content.find('"""', system_start)
            system_prompt = template_content[system_start:system_end].strip()
            
            # 提取USER部分
            user_start = template_content.find('USER_PROMPT_TEMPLATE = """') + len('USER_PROMPT_TEMPLATE = """')
            user_end = template_content.rfind('"""')
            user_template = template_content[user_start:user_end].strip()
            
            print(f"✅ 系统指令长度: {len(system_prompt)} 字符")
            print(f"✅ 用户模板长度: {len(user_template)} 字符")
        else:
            print("❌ 模板格式不正确")
            return False
        
        # 准备测试数据
        print("\n4. 准备测试数据...")
        context = "德赛电池（000049）2021年业绩预告显示，公司营收约193.9亿元，同比增长5%，净利润7.07亿元，同比增长45.13%，归母净利润6.37亿元，同比增长25.5%。业绩超出预期主要源于iPhone 12 Pro Max需求佳及盈利能力提升。"
        query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        summary = "德赛电池2021年业绩超出预期，主要受益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。"
        
        # 格式化Prompt
        print("\n5. 格式化Prompt...")
        try:
            user_content = user_template.format(
                summary=summary,
                context=context,
                query=query
            )
            # 构造完整的Prompt（系统指令 + 用户内容）
            prompt = f"{system_prompt}\n\n{user_content}"
        except Exception as e:
            print(f"❌ Prompt格式化失败: {e}")
            return False
        print(f"✅ Prompt格式化完成，长度: {len(prompt)} 字符")
        
        # 测试JSON格式转换
        print("\n6. 测试JSON格式转换...")
        json_chat = generator.convert_to_json_chat_format(prompt)
        print(f"✅ JSON格式转换完成，长度: {len(json_chat)} 字符")
        
        # 打印JSON格式预览
        print("\n7. JSON格式预览:")
        print("-" * 50)
        import json
        try:
            json_data = json.loads(json_chat)
            print(json.dumps(json_data, ensure_ascii=False, indent=2))
        except:
            print("JSON格式转换失败，使用原始格式")
        print("-" * 50)
        
        # 测试Fin-R1格式转换
        print("\n8. 测试Fin-R1格式转换...")
        fin_r1_format = generator.convert_json_to_fin_r1_format(json_chat)
        print(f"✅ Fin-R1格式转换完成，长度: {len(fin_r1_format)} 字符")
        
        # 打印Fin-R1格式预览
        print("\n9. Fin-R1格式预览:")
        print("-" * 50)
        print(fin_r1_format[:500] + "..." if len(fin_r1_format) > 500 else fin_r1_format)
        print("-" * 50)
        
        # 调用LLM生成器
        print("\n10. 调用LLM生成器...")
        print("🚀 开始生成答案...")
        
        response = generator.generate([prompt])
        answer = response[0] if response else "生成失败"
        
        print("\n" + "=" * 80)
        print("📝 生成结果")
        print("=" * 80)
        print(f"问题: {query}")
        print(f"答案: {answer}")
        print("=" * 80)
        
        # 分析结果
        print("\n📊 结果分析:")
        print(f"✅ JSON格式转换: {'成功' if json_chat != prompt else '失败'}")
        print(f"✅ Fin-R1格式转换: {'成功' if fin_r1_format != json_chat else '失败'}")
        print(f"✅ 答案生成: {'成功' if answer and answer != '生成失败' else '失败'}")
        print(f"✅ 答案长度: {len(answer)} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_json_chat_mode()
    if success:
        print("\n🎉 JSON格式聊天模式测试完成！")
    else:
        print("\n💥 JSON格式聊天模式测试失败！") 