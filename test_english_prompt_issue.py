#!/usr/bin/env python3
"""
测试英文查询得到中文响应的问题
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_english_prompt_issue():
    """测试英文 Prompt 问题"""
    
    print("=== 英文查询响应语言问题测试 ===")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        from xlm.components.prompt_templates.template_loader import template_loader
        
        # 初始化生成器
        print("1. 初始化 LLM 生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        
        # 测试数据
        test_context = """
        Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. 
        iPhone sales increased 2.8% to $39.7 billion. The company's services revenue 
        grew 8.2% to $21.2 billion, while Mac and iPad sales declined.
        """
        
        test_query = "How did Apple perform in Q3 2023?"
        
        # 测试不同的 Prompt 模板
        prompt_templates = {
            "英文模板": "rag_english_template",
            "多阶段英文模板": "multi_stage_english_template"
        }
        
        for template_name, template_key in prompt_templates.items():
            print(f"\n2. 测试 {template_name}...")
            
            # 生成 Prompt
            if template_key == "rag_english_template":
                prompt = template_loader.format_template(
                    template_key,
                    context=test_context,
                    question=test_query
                )
            else:
                prompt = template_loader.format_template(
                    template_key,
                    context=test_context,
                    query=test_query
                )
            
            if prompt is None:
                print(f"❌ {template_name} 模板加载失败")
                continue
                
            print(f"✅ Prompt 生成成功，长度: {len(prompt)} 字符")
            print(f"✅ Prompt 预览:\n{prompt[:300]}...")
            
            # 检查格式转换
            print(f"\n3. 检查格式转换...")
            if "Fin-R1" in generator.model_name:
                print("🔍 检测到 Fin-R1 模型，检查格式转换...")
                
                # 检查是否会进行格式转换
                json_chat = generator.convert_to_json_chat_format(prompt)
                print(f"JSON 格式转换结果: {'会转换' if json_chat != prompt else '不会转换'}")
                
                if json_chat != prompt:
                    print(f"转换后的 JSON 格式: {json_chat[:200]}...")
                    
                    # 转换为 Fin-R1 格式
                    fin_r1_format = generator.convert_json_to_fin_r1_format(json_chat)
                    print(f"Fin-R1 格式预览: {fin_r1_format[:300]}...")
                else:
                    print("⚠️ 英文 Prompt 不会进行格式转换，可能影响 Fin-R1 模型理解")
            
            # 生成响应
            print(f"\n4. 生成响应...")
            print("🚀 开始生成，请稍候...")
            
            responses = generator.generate([prompt])
            response = responses[0] if responses else "生成失败"
            
            print(f"\n5. 生成结果:")
            print("=" * 60)
            print(f"问题: {test_query}")
            print(f"答案: {response}")
            print("=" * 60)
            
            # 分析响应语言
            print(f"\n6. 语言分析:")
            
            # 检测响应语言
            try:
                from langdetect import detect
                response_lang = detect(response)
                print(f"   响应语言: {response_lang}")
                
                # 检查是否包含中文字符
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
                print(f"   包含中文字符: {'是' if has_chinese else '否'}")
                
                # 检查是否包含英文字符
                has_english = any(char.isalpha() and ord(char) < 128 for char in response)
                print(f"   包含英文字符: {'是' if has_english else '否'}")
                
                # 判断语言一致性
                if response_lang.startswith('en') and not has_chinese:
                    print("   ✅ 语言一致：英文查询得到英文响应")
                elif response_lang.startswith('zh') and not has_english:
                    print("   ✅ 语言一致：中文查询得到中文响应")
                else:
                    print("   ❌ 语言不一致：查询和响应语言不匹配")
                    
            except Exception as e:
                print(f"   语言检测失败: {e}")
            
            # 检查响应质量
            print(f"\n7. 响应质量:")
            length = len(response.strip())
            print(f"   响应长度: {length} 字符")
            
            # 检查是否包含关键信息
            key_terms = ["Apple", "revenue", "billion", "iPhone", "sales"]
            found_terms = [term for term in key_terms if term.lower() in response.lower()]
            print(f"   关键信息: {found_terms}")
            print(f"   准确性: {'✅' if len(found_terms) >= 3 else '❌'} (找到{len(found_terms)}个关键词)")
            
            # 检查格式标记
            unwanted_patterns = ["【", "】", "回答：", "Answer:", "---", "===", "___"]
            has_unwanted = any(pattern in response for pattern in unwanted_patterns)
            print(f"   纯粹性: {'✅' if not has_unwanted else '❌'} (无格式标记)")
            
            print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fin_r1_format_conversion():
    """测试 Fin-R1 格式转换逻辑"""
    
    print("\n=== Fin-R1 格式转换测试 ===")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        generator = LocalLLMGenerator()
        
        # 测试不同的 Prompt 内容
        test_cases = [
            {
                "name": "中文 Prompt",
                "content": """你是一位专业的金融分析师。请基于以下信息回答问题：

摘要：德赛电池2021年业绩增长主要受益于iPhone需求强劲。

详细内容：德赛电池2021年业绩预告显示，公司预计实现净利润为6.5亿元至7.5亿元。

问题：德赛电池2021年利润增长的原因是什么？

回答："""
            },
            {
                "name": "英文 Prompt",
                "content": """You are a financial analyst. Please answer the following question based on the provided information:

Summary: Apple Inc. reported Q3 2023 revenue of $81.8 billion.

Details: Apple's iPhone sales increased 2.8% to $39.7 billion.

Question: How did Apple perform in Q3 2023?

Answer:"""
            },
            {
                "name": "英文模板 Prompt",
                "content": """You are a highly analytical and precise financial expert. Your task is to answer the user's question **strictly based on the provided context information**.

**CRITICAL: Your output must be a pure, direct answer. Do NOT include any self-reflection, thinking process, prompt analysis, irrelevant comments, format markers (like boxed, numbered lists, bold text), or any form of meta-commentary. Do NOT quote or restate the prompt content. Your answer must end directly and concisely without any follow-up explanations.**

Requirements:
1.  **Strictly adhere to the provided context. Do not use any external knowledge or make assumptions.**
2.  If the context does not contain sufficient information to answer the question, state: "The answer cannot be found in the provided context."
3.  For questions involving financial predictions or future outlook, prioritize information explicitly stated as forecasts or outlooks within the context.
4.  Provide a concise and direct answer in complete sentences.
5.  Do not repeat the question or add conversational fillers.

Context:
Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year.

Question: How did Apple perform in Q3 2023?

"""
            }
        ]
        
        for test_case in test_cases:
            print(f"\n测试: {test_case['name']}")
            print("-" * 40)
            
            content = test_case['content']
            print(f"原始内容长度: {len(content)} 字符")
            print(f"包含中文关键词: {'是' if '你是一位专业的金融分析师' in content else '否'}")
            
            # 测试格式转换
            json_chat = generator.convert_to_json_chat_format(content)
            will_convert = json_chat != content
            
            print(f"会进行格式转换: {'是' if will_convert else '否'}")
            
            if will_convert:
                print(f"转换后长度: {len(json_chat)} 字符")
                fin_r1_format = generator.convert_json_to_fin_r1_format(json_chat)
                print(f"Fin-R1 格式长度: {len(fin_r1_format)} 字符")
                print(f"Fin-R1 格式预览: {fin_r1_format[:200]}...")
            else:
                print("⚠️ 不会进行格式转换")
            
            print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_english_prompt_issue()
    test_fin_r1_format_conversion() 