#!/usr/bin/env python3
"""
量化质量对比测试
比较 4bit 和 8bit 量化的 Fin-R1 响应质量
"""

import torch
import time
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model_with_quantization(quantization_type="4bit"):
    """加载指定量化类型的模型"""
    model_name = "SUFE-AIFLM-Lab/Fin-R1"
    cache_dir = "/users/sgjfei3/data/huggingface"
    
    print(f"🔧 加载 {quantization_type} 量化模型...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化
    if quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
        local_files_only=True
    )
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_new_tokens=150):
    """生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # 移动到正确的设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = "cuda:1"
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的部分（去掉输入）
    generated_text = response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    return generated_text.strip(), generation_time

def evaluate_response_quality(response):
    """评估响应质量"""
    quality_score = 0
    feedback = []
    
    # 检查长度
    if len(response) > 50:
        quality_score += 20
        feedback.append("✅ 回答长度适中")
    else:
        feedback.append("⚠️ 回答过短")
    
    # 检查专业性
    professional_keywords = ["分析", "增长", "收入", "利润", "业务", "市场", "财务", "业绩"]
    professional_count = sum(1 for keyword in professional_keywords if keyword in response)
    if professional_count >= 2:
        quality_score += 30
        feedback.append("✅ 使用专业术语")
    else:
        feedback.append("⚠️ 专业术语较少")
    
    # 检查逻辑性
    if "因为" in response or "由于" in response or "主要" in response:
        quality_score += 25
        feedback.append("✅ 逻辑结构清晰")
    else:
        feedback.append("⚠️ 逻辑结构一般")
    
    # 检查完整性
    if response.endswith(("。", "！", "？", ".", "!", "?")):
        quality_score += 15
        feedback.append("✅ 回答完整")
    else:
        feedback.append("⚠️ 回答可能不完整")
    
    # 检查相关性
    if "德赛电池" in response or "iPhone" in response or "A客户" in response:
        quality_score += 10
        feedback.append("✅ 内容相关性强")
    else:
        feedback.append("⚠️ 内容相关性一般")
    
    return quality_score, feedback

def test_quantization_comparison():
    """测试量化对比"""
    print("🎯 Fin-R1 量化质量对比测试")
    print("=" * 60)
    
    # 测试用例
    test_context = """德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。

研报显示：德赛电池发布20年业绩预告，20年营收约193.9亿元，同比增长5%，归母净利润6.3-6.9亿元，同比增长25.5%-37.4%。21年利润持续增长，源于A客户及非手机业务成长及并表比例增加。公司认为超预期主要源于iPhone 12 Pro Max新机需求佳及新品盈利能力提升。展望21年，5G iPhone周期叠加非手机业务增量，Watch、AirPods需求量增长，iPad、Mac份额提升，望驱动A客户业务成长。"""
    
    test_query = "德赛电池2021年利润持续增长的主要原因是什么？"
    
    test_prompt = f"""你是一位专业的金融分析师，请基于以下公司财务报告信息，准确、简洁地回答用户的问题。

【公司财务报告摘要】
{test_context}

【用户问题】
{test_query}

请提供准确、专业的分析回答："""
    
    results = {}
    
    # 测试 4bit 量化
    try:
        print("\n🔧 测试 4bit 量化...")
        tokenizer_4bit, model_4bit = load_model_with_quantization("4bit")
        
        response_4bit, time_4bit = generate_response(tokenizer_4bit, model_4bit, test_prompt)
        quality_4bit, feedback_4bit = evaluate_response_quality(response_4bit)
        
        results["4bit"] = {
            "response": response_4bit,
            "time": time_4bit,
            "quality": quality_4bit,
            "feedback": feedback_4bit
        }
        
        print(f"✅ 4bit 响应: {response_4bit}")
        print(f"⏱️ 生成时间: {time_4bit:.2f}秒")
        print(f"📊 质量评分: {quality_4bit}/100")
        
        # 清理内存
        del model_4bit
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 4bit 测试失败: {e}")
    
    # 测试 8bit 量化
    try:
        print("\n🔧 测试 8bit 量化...")
        tokenizer_8bit, model_8bit = load_model_with_quantization("8bit")
        
        response_8bit, time_8bit = generate_response(tokenizer_8bit, model_8bit, test_prompt)
        quality_8bit, feedback_8bit = evaluate_response_quality(response_8bit)
        
        results["8bit"] = {
            "response": response_8bit,
            "time": time_8bit,
            "quality": quality_8bit,
            "feedback": feedback_8bit
        }
        
        print(f"✅ 8bit 响应: {response_8bit}")
        print(f"⏱️ 生成时间: {time_8bit:.2f}秒")
        print(f"📊 质量评分: {quality_8bit}/100")
        
        # 清理内存
        del model_8bit
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 8bit 测试失败: {e}")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("📊 量化对比结果")
    print("=" * 60)
    
    if "4bit" in results and "8bit" in results:
        print(f"🔍 质量评分对比:")
        print(f"   4bit: {results['4bit']['quality']}/100")
        print(f"   8bit: {results['8bit']['quality']}/100")
        print(f"   差异: {results['8bit']['quality'] - results['4bit']['quality']}")
        
        print(f"\n⏱️ 生成时间对比:")
        print(f"   4bit: {results['4bit']['time']:.2f}秒")
        print(f"   8bit: {results['8bit']['time']:.2f}秒")
        print(f"   速度比: {results['8bit']['time'] / results['4bit']['time']:.2f}x")
        
        print(f"\n📝 响应内容对比:")
        print(f"   4bit: {results['4bit']['response'][:100]}...")
        print(f"   8bit: {results['8bit']['response'][:100]}...")
        
        # 质量评估
        quality_diff = results['8bit']['quality'] - results['4bit']['quality']
        if quality_diff > 10:
            print(f"\n⚠️ 8bit 质量明显优于 4bit (差异: {quality_diff})")
        elif quality_diff > 5:
            print(f"\n📊 8bit 质量略优于 4bit (差异: {quality_diff})")
        elif abs(quality_diff) <= 5:
            print(f"\n✅ 4bit 和 8bit 质量相当 (差异: {quality_diff})")
        else:
            print(f"\n🎉 4bit 质量优于 8bit (差异: {quality_diff})")
    
    elif "4bit" in results:
        print("✅ 仅 4bit 测试成功")
        print(f"质量评分: {results['4bit']['quality']}/100")
        print(f"生成时间: {results['4bit']['time']:.2f}秒")
    
    else:
        print("❌ 所有测试都失败")

def main():
    """主函数"""
    test_quantization_comparison()

if __name__ == "__main__":
    main() 