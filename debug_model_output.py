#!/usr/bin/env python3
"""
调试模型输出，检查是否包含<answer>标签
"""

import json
import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 导入RAG系统的LocalLLMGenerator
try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
except ImportError:
    print("⚠️ 无法导入LocalLLMGenerator，使用备用方案")
    USE_RAG_GENERATOR = False

def extract_final_answer(raw_output: str) -> str:
    """从模型的原始输出中提取<answer>标签内的内容"""
    print(f"\n🔍 原始输出长度: {len(raw_output)}")
    print(f"🔍 原始输出前200字符: {raw_output[:200]}...")
    
    # 查找<answer>标签
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        print(f"✅ 找到<answer>标签，内容: {answer_content}")
        return answer_content
    
    # 查找<think>标签
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if think_match:
        print(f"⚠️ 只找到<think>标签，没有<answer>标签")
        print(f"🔍 <think>内容: {think_match.group(1).strip()[:100]}...")
    
    # 检查是否包含未闭合的<answer>标签
    if '<answer>' in raw_output and '</answer>' not in raw_output:
        print("⚠️ 发现未闭合的<answer>标签")
        # 提取<answer>后的所有内容
        answer_start = raw_output.find('<answer>') + len('<answer>')
        answer_content = raw_output[answer_start:].strip()
        print(f"🔍 <answer>后的内容: {answer_content[:100]}...")
        return answer_content
    
    # 如果没找到，返回最后一行
    lines = raw_output.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        print(f"⚠️ 未找到标签，使用最后一行: {last_line}")
        return last_line
    
    print("❌ 没有找到任何答案内容")
    return ""

def get_detailed_english_prompt_messages(context_content: str, question_text: str) -> List[Dict[str, str]]:
    """生成LLM期望的messages列表"""
    try:
        with open('rag_english_template.txt', 'r', encoding='utf-8') as f:
            template_content = f.read().strip()
    except FileNotFoundError:
        print("⚠️ 模板文件未找到，使用默认模板")
        return [
            {"role": "system", "content": "You are a world-class quantitative financial analyst AI."},
            {"role": "user", "content": f"Context:\n{context_content}\n\nQuestion:\n{question_text}\n\nA:"}
        ]
    
    # 解析system和user标签
    system_match = re.search(r'<system>(.*?)</system>', template_content, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
    else:
        system_content = "You are a world-class quantitative financial analyst AI."
    
    user_match = re.search(r'<user>(.*?)</user>', template_content, re.DOTALL)
    if user_match:
        user_template = user_match.group(1).strip()
        user_content = user_template.replace('{context}', context_content).replace('{question}', question_text)
    else:
        user_content = f"Context:\n{context_content}\n\nQuestion:\n{question_text}\n\nA:"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def convert_messages_to_text(messages: List[Dict[str, str]]) -> str:
    """将messages转换为文本格式"""
    text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += f"System: {content}\n\n"
        elif role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n\n"
    return text.strip()

def test_model_output():
    """测试模型输出"""
    print("🚀 开始调试模型输出...")
    
    # 加载一个简单的测试样本
    test_context = "Table ID: 1\nCompany: Apple Inc.\nRevenue: $394.3 billion\nProfit: $96.9 billion"
    test_question = "What is Apple's revenue?"
    test_answer = "$394.3 billion"
    
    print(f"📝 测试问题: {test_question}")
    print(f"📊 测试上下文: {test_context}")
    print(f"✅ 期望答案: {test_answer}")
    
    # 构建prompt
    messages = get_detailed_english_prompt_messages(test_context, test_question)
    prompt_text = convert_messages_to_text(messages)
    
    print(f"\n📋 完整Prompt:")
    print("="*60)
    print(prompt_text)
    print("="*60)
    
    if USE_RAG_GENERATOR:
        try:
            # 使用RAG系统的LocalLLMGenerator
            llm_generator = LocalLLMGenerator(
                model_name="SUFE-AIFLM-Lab/Fin-R1",
                device="auto"
            )
            
            # 设置生成参数
            generation_params = {
                "max_new_tokens": 2048,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "temperature": 0.1  # 降低温度以获得更确定的输出
            }
            
            print(f"\n🔧 生成参数: {generation_params}")
            
            # 生成回答
            start_time = time.time()
            generated_text = llm_generator.generate([prompt_text])[0]
            generation_time = time.time() - start_time
            
            print(f"\n⏱️ 生成时间: {generation_time:.2f}秒")
            print(f"📏 生成文本长度: {len(generated_text)}")
            
            # 分析输出
            print(f"\n🤖 模型完整输出:")
            print("="*80)
            print(generated_text)
            print("="*80)
            
            # 提取答案
            final_answer = extract_final_answer(generated_text)
            
            print(f"\n🎯 提取的最终答案: '{final_answer}'")
            print(f"🎯 期望答案: '{test_answer}'")
            
            # 检查匹配
            if final_answer.strip().lower() == test_answer.strip().lower():
                print("✅ 答案完全匹配！")
            elif test_answer.strip().lower() in final_answer.strip().lower():
                print("✅ 答案包含期望内容！")
            else:
                print("❌ 答案不匹配")
                
        except Exception as e:
            print(f"❌ RAG生成器错误: {e}")
    else:
        print("❌ RAG生成器不可用")

if __name__ == "__main__":
    test_model_output() 