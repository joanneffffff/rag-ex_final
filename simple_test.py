#!/usr/bin/env python3
"""
极简测试 - 直接测试模型生成，移除所有复杂逻辑
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def simple_test():
    print("🚀 极简测试开始...")
    
    # 使用更小的模型
    model_name = "Qwen/Qwen2-1.5B-Instruct"  # 使用更小的模型
    print(f"📋 使用模型: {model_name}")
    
    try:
        print("🔧 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("🔧 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 简单测试
        test_prompt = "请用一句话回答：德赛电池的主要业务是什么？"
        print(f"\n📝 测试问题: {test_prompt}")
        
        # 编码
        print("🔤 编码输入...")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        print("🤖 开始生成...")
        start_time = time.time()
        
        # 极简生成参数
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 限制token数
                do_sample=False,    # 贪婪解码
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        print(f"⏱️ 生成时间: {end_time - start_time:.2f}秒")
        
        # 解码
        print("🔤 解码输出...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        input_length = inputs.input_ids.shape[1]
        generated_text = response[input_length:]
        
        print(f"\n✅ 回答:")
        print(f"{'='*50}")
        print(generated_text.strip())
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test() 