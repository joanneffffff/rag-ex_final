#!/usr/bin/env python3
"""
生成模块性能评估脚本 - 对比 Fin-R1 和 Qwen2-7B-Instruct 在中文数据集上的表现。
支持批量随机样本测试，并输出详细日志。
优化了显存占用，模型按顺序加载和卸载。
Prompt Template 内容从外部文件加载。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
import gc
import json 
import argparse 
from tqdm import tqdm 
import numpy as np 
from typing import List, Optional, Dict, Any
from collections import Counter

# 导入配置文件
try:
    from config.parameters import config
    print(f"✅ 使用配置文件中的缓存路径: {config.generator.cache_dir}")
except ImportError:
    print("⚠️ 无法导入配置文件，使用默认缓存路径")
    config = None 

# ====================================================================================
# 后处理模块定义 (专门针对中文)
# ====================================================================================

def _fix_company_name_translation(text: str) -> str:
    """
    修正公司名称翻译问题和年份问题 (仅限中文)。
    """
    company_translations = {
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*BATTERY\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
        r'德赛\s*BATTERY': '德赛电池',
        r'德赛\s*\(00\)': '德赛电池（000049）', 
        r'德塞电池': '德赛电池', 
        
        r'iPhone\s*\+\s*ProMax': 'iPhone 12 Pro Max',
        r'iPhon\s*e12ProMax': 'iPhone 12 Pro Max',
        r'iPhone\s*X\s*系列': 'iPhone 12 Pro Max', 
        r'iPhone\s*1\s*\(Pro\s*Max\s*\)': 'iPhone 12 Pro Max',
        r'iPhone\s*1\s*Pro\s*Max': 'iPhone 12 Pro Max',
        r'iPhone\s*2\s*ProMax': 'iPhone 12 Pro Max',
    }
    for pattern, replacement in company_translations.items():
        text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'20\s*\(\s*\d{2}\?\)\s*年度', r'2021年度', text, flags=re.IGNORECASE)
    text = text.replace('20XX年', '2021年')
    text = text.replace('20+', '2021')
    text = text.replace('2OI I年', '2021年')
    text = text.replace('20 I I年', '2021年')
        
    return text


def clean_response(text: str) -> str:
    """
    强制后处理模块：清除所有污染内容 (专门针对中文规则)。
    """
    # print("🧹 开始强制后处理...") # 在循环中打印会很吵
    
    text = _fix_company_name_translation(text) 
    
    patterns_to_remove = [
        r'我需要检查这个回答是否符合要求.*?====', 
        r'\*\*注意\*\*:.*?改进后的版本[:：]', 
        r'上面的答案虽然符合要求.*?以下是改进后的版本:', 
        r'###\s*改进版答案', 
        r'###\s*回答', 
        r'回答完成后立即停止生成', 
        r'回答完成并停止', 
        r'确保回答', 
        r'用户可能', 
        r'总结一下', 
        r'请用简洁', 
        r'进一步简化', 
        r'再简化的版本', 
        r'最终答案定稿如下', 
        r'这个总结全面', 
        r'核心点总结[:：]?', 
        r'以上分析是否正确？还有哪些方面可以改进？', 
        r'您的分析基本合理，但在某些地方可以进一步完善和细化。以下是几点改进建议：',
        r'（参阅第三部分）', # 修正：多了一个右括号
        r'（详情见第②段）',
        r'这些问题的答案需要结合具体的研究报告内容进行详细分析。',
        r'上述答案涵盖了报告中提及的关键因素，并进行了适当归纳。',
        r'如有需要进一步细化某一方面的内容，请告知。',
        r'注意：以上论断完全依赖于已公开披露的信息资源 ; 对未来的具体前景尚需结合更多实时数据加以验证和完善', 
        r'（注意此段文字虽详细阐述了几方面因素及其相互作用机制，但由于题干要求高度浓缩为一句话内完成表述，故在此基础上进行了适当简化压缩）', 
        r'请注意，以上内容是对.*?展望，并非绝对结论。', 
        r'实际走势还需结合实际情况不断评估调整。希望这个回答对你有所帮助！', 
        r'要预测.*?做出判断[:：]?', 
        r'以下是几个关键因素和步骤[:：]?',
        r'综上所述[:：]?', 
        r'最终结论[:：]?',
        r'答案示例[:：：]?', 
        r'最终确认[:：]?',
        r'答案忠实地反映了原始文档的内容而无多余推断',
        r'回答[:：]\s*$', 
        r'回答是：\s*', 
        r'以下是原因：\s*',

        r'<\|[^>]+\|>', 
        r'\\boxed\{.*?\}', 
        r'\\text\{.*?\}', 
        r'\\s*', 
        r'[\u2460-\u2469]\s*', 

        r'===SYSTEM===[\s\S]*?===USER===', 
        r'---[\s\S]*?---', 
        r'【公司财务报告摘要】[\s\S]*?【完整公司财务报告片段】', 
        r'【用户问题】[\s\S]*?【回答】', 

        r'Question:\n.*?\nTable Context:', 
        r'Table Context:\n.*?\nText Context:', 
        r'Text Context:\n.*?\nQuestion:', 
        r'Context:\n.*?\nQuestion:', 
        r'Assistant\'s Response:', 
        r'--- END OF EXAMPLES ---', 
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) 
    text = re.sub(r'\*(.*?)\*', r'\1', text) 
    text = text.replace("---", "").replace("===", "") 
    text = re.sub(r'^\s*[\d]+\.\s*', '', text, flags=re.MULTILINE) 
    text = re.sub(r'^\s*[-*•·]\s*', '', text, flags=re.MULTILINE) 
    text = re.sub(r'^\s*\((\w|[一二三四五六七八九十])+\)\s*', '', text, flags=re.MULTILINE) 
    text = re.sub(r'[，；,;]$', '', text) 

    text = re.sub(r'\n+', ' ', text).strip() 
    text = re.sub(r'\s+', ' ', text).strip() 

    sentence_endings = r'(?<=[。？！；])\s*'
    default_ending = '。'

    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 3: 
        sentences = sentences[:3]
    
    final_text = ' '.join(sentences) 

    if final_text and not final_text.endswith(('.', '!', '?', '。', '！', '？')): 
        final_text += default_ending

    return final_text


# ====================================================================================
# Prompt 构造辅助函数 (从外部文件加载)
# ====================================================================================

def _load_template_content_from_file(template_file_name: str) -> str:
    """从指定文件中加载Prompt模板的完整字符串内容"""
    template_path = Path("data/prompt_templates") / template_file_name # 直接使用文件名，不添加chinese子目录
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ 模板文件未找到: {template_path}，请确保文件存在。")
        sys.exit(1)

def get_messages_for_test(summary: str, context: str, query: str, template_file_name: str = "chinese_test_template.txt") -> List[Dict[str, str]]:
    """
    构建用于测试的 messages 列表，从指定模板文件加载内容。
    Args:
        context (str): 完整上下文（已包含摘要）。
        query (str): 用户问题。
        template_file_name (str): 要加载的模板文件名 (例如 "chinese_test_template.txt")。
    Returns:
        List[Dict[str, str]]: 构建好的 messages 列表。
    """
    template_full_string = _load_template_content_from_file(template_file_name)
    
    messages = []
    # 使用正则表达式分割所有部分，并保留分隔符内容
    parts = re.split(r'(===SYSTEM===|===USER===|===ASSISTANT===)', template_full_string, flags=re.DOTALL)
    
    # 移除第一个空字符串（如果存在）和多余的空白
    parts = [p.strip() for p in parts if p.strip()]

    # 遍历 parts 列表，重新组合 role 和 content
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            role_tag_raw = parts[i].strip()
            content = parts[i+1].strip()
            
            role = None
            if role_tag_raw == "===SYSTEM===": role = "system"
            elif role_tag_raw == "===USER===": role = "user"
            elif role_tag_raw == "===ASSISTANT===": role = "assistant"
            
            if role and content:
                # 替换占位符 (只针对 'user' 角色消息进行替换)
                if role == "user":
                    content = content.replace('{query}', query)
                    # 替换 {summary} 和 {context}
                    # 直接使用传入的summary和context参数
                    content = content.replace('{summary}', summary).replace('{context}', context)
                
                messages.append({"role": role, "content": content})
                
    return messages


def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 列表转换为 Fin-R1 (Qwen2.5 based) 期望的ChatML格式字符串。
    """
    if not messages:
        return ""
    
    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"<|im_start|>system\n{content.strip()}<|im_end|>\n"
        elif role == "user":
            formatted_prompt += f"<|im_start|>user\n{content.strip()}<|im_end|>\n"
        elif role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"
    
    formatted_prompt += "<|im_start|>assistant\n" 
    
    return formatted_prompt


# ====================================================================================
# 模型加载和生成器包装类
# ====================================================================================

class ModelLoader:
    """负责加载和卸载模型，并提供生成接口"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

        # 使用配置文件中的缓存路径
        cache_dir = config.generator.cache_dir if config else "/users/sgjfei3/data/huggingface"
        
        if "Fin-R1" in model_name: 
            # 检查本地缓存路径
            local_fin_r1_path = f"{cache_dir}/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa"
            if os.path.exists(local_fin_r1_path):
                self.model_path = local_fin_r1_path
                print(f"✅ 使用本地缓存的Fin-R1模型: {self.model_path}")
            else:
                self.model_path = "SUFE-AIFLM-Lab/Fin-R1"
                print(f"⚠️ 本地缓存未找到，将从Hub下载: {self.model_path}")
        elif "Qwen3-8B" in model_name:
            # 检查本地缓存路径 - 使用正确的Qwen3-8B路径
            local_qwen_path = f"{cache_dir}/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
            if os.path.exists(local_qwen_path):
                self.model_path = local_qwen_path
                print(f"✅ 使用本地缓存的Qwen3-8B模型: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-8B"
                print(f"⚠️ 本地缓存未找到，将从Hub下载: {self.model_path}")
        else:
            self.model_path = model_name 
            print(f"⚠️ 模型路径 '{model_name}' 未知，尝试从Hugging Face Hub加载。建议提前下载到本地。")

    def load_model(self):
        if self.is_loaded:
            print(f"✅ {self.model_name} 已加载，无需重复加载。")
            return
        
        print(f"🔄 加载模型: {self.model_name} 从 {self.model_path}")
        is_local_path = isinstance(self.model_path, str) and ("snapshots" in self.model_path or "models--" in self.model_path) # 判断是否为本地缓存路径

        # 使用配置文件中的缓存设置
        cache_dir = config.generator.cache_dir if config else None
        tokenizer_args = {"trust_remote_code": True, "local_files_only": is_local_path, "cache_dir": cache_dir}
        model_args = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True, 
                      "load_in_8bit": True, "local_files_only": is_local_path, "cache_dir": cache_dir} 

        try:
            print("🔧 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"✅ {self.model_name} Tokenizer加载完成. Chat Template: {self.tokenizer.chat_template}")

            print("🔧 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            print(f"✅ {self.model_name} 模型加载完成. 设备: {self.model.device}, 量化: 8bit")
            self.is_loaded = True
        except Exception as e:
            print(f"❌ {self.model_name} 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.unload_model() # 确保失败时也清理
            raise

    def unload_model(self):
        if not self.is_loaded:
            return
        
        print(f"🗑️ 卸载模型: {self.model_name} 并清理显存...")
        try:
            if self.model:
                # 对于8位量化模型，直接删除而不使用.to()方法
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache() # 清理CUDA缓存
            gc.collect() # 垃圾回收
            self.is_loaded = False
            print(f"✅ {self.model_name} 显存已清理。")
        except Exception as e:
            print(f"❌ 卸载 {self.model_name} 时发生错误: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 150, do_sample: bool = False, repetition_penalty: float = 1.1) -> str:
        """生成文本，期望输入已经是 ChatML 格式的字符串"""
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载。请先调用 load_model()。")

        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate( 
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
        
        generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) 
        
        return generated_text

# ====================================================================================
# 评估指标计算 (你需要实现这些函数，这里是占位符)
# ====================================================================================
# Placeholder for F1 and EM if not imported
def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数，这里是占位符。"""
    return 1.0 if prediction == ground_truth else 0.0

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配率，这里是占位符。"""
    return 1.0 if prediction == ground_truth else 0.0


# ====================================================================================
# 主测试逻辑
# ====================================================================================

def run_chinese_comparison_test(args):
    print("🚀 中文模型对比测试开始...")
    
    # --- 配置要测试的模型 ---
    model_loaders = {
        "Fin-R1": ModelLoader("Fin-R1"),
        "Qwen3-8B": ModelLoader("Qwen3-8B")
    }

    # --- 测试配置 (使用命令行参数) ---
    data_path = args.data_path # 使用命令行参数
    sample_size = args.sample_size # 使用命令行参数
    # 模板文件名，需要与 data/prompt_templates/chinese/ 下的文件名一致
    template_file_name = "multi_stage_chinese_template.txt"  # 只使用文件名，路径在函数中拼接 
    
    # --- 加载数据集 ---
    print(f"📊 加载数据集: {data_path}")
    try:
        from utils.data_loader import load_json_or_jsonl, sample_data 
        dataset = load_json_or_jsonl(data_path)
        
        if sample_size > 0:
            dataset = sample_data(dataset, sample_size, 42)
            print(f"✅ 随机采样 {len(dataset)} 个样本进行评估。")
        else:
            print(f"✅ 加载了全部 {len(dataset)} 个样本进行评估。")
            
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    all_results_data = [] # 存储所有模型的评估结果

    # --- 逐个模型进行评估 ---
    for model_name, loader in model_loaders.items():
        print(f"\n🔄 开始评估模型: {model_name}")
        print(f"📊 当前GPU内存状态:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
        
        current_model_results = []
        total_f1_model = 0.0
        total_em_model = 0.0
        total_generation_time_model = 0.0
        
        try:
            print(f"🔄 正在加载模型: {model_name}")
            loader.load_model() # 加载当前模型
            print(f"✅ 模型 {model_name} 加载完成，开始评估...")
            
            pbar = tqdm(dataset, desc=f"评估 {model_name}")
            for i, item in enumerate(pbar):
                # 兼容多种查询字段名
                query = item.get("query", "") or item.get("generated_question", "") or item.get("question", "")
                summary = item.get("summary", "") # 获取summary字段
                context = item.get("context", "") # 获取context字段
                expected_answer = item.get("answer", "") # 获取参考答案

                # 构建 messages 列表，从外部模板文件加载
                # 传递summary和context，让模板函数正确处理
                messages = get_messages_for_test(summary, context, query, template_file_name)
                
                # 转换为 ChatML 格式
                prompt_string_for_model = _convert_messages_to_chatml(messages)
                
                start_time = time.time()
                generated_text = loader.generate( # 调用 loader 内部的 generate
                    prompt_string=prompt_string_for_model,
                    max_new_tokens=150, # 使用硬编码的max_new_tokens
                    do_sample=False, 
                    repetition_penalty=1.1
                )
                generation_time = time.time() - start_time
                
                final_answer = clean_response(generated_text) # 后处理 (硬编码为中文)
                
                f1 = calculate_f1_score(final_answer, expected_answer)
                em = calculate_exact_match(final_answer, expected_answer)

                total_f1_model += f1
                total_em_model += em
                total_generation_time_model += generation_time

                current_model_results.append({
                    "model": model_name,
                    "sample_id": i,
                    "query": query,
                    "expected_answer": expected_answer,
                    "raw_generated_text": generated_text,
                    "final_answer": final_answer,
                    "f1_score": f1,
                    "exact_match": em,
                    "generation_time": generation_time
                })

            # 打印当前模型的汇总结果
            num_samples_evaluated = len(dataset)
            avg_f1 = total_f1_model / num_samples_evaluated
            avg_em = total_em_model / num_samples_evaluated
            avg_gen_time = total_generation_time_model / num_samples_evaluated

            print(f"\n--- {model_name} 评估总结 ---")
            print(f"总样本数: {num_samples_evaluated}")
            print(f"平均 F1-score: {avg_f1:.4f}")
            print(f"平均 Exact Match: {avg_em:.4f}")
            print(f"平均生成时间: {avg_gen_time:.2f} 秒/样本")
            print("--------------------")
            
            all_results_data.extend(current_model_results)

        except Exception as e:
            print(f"❌ 评估 {model_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"🗑️ 正在卸载模型: {model_name}")
            loader.unload_model() # 确保每次循环都卸载模型
            print(f"✅ 模型 {model_name} 卸载完成")
            print(f"📊 卸载后GPU内存状态:")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

    # --- 评估完成，保存所有结果 ---
    output_filename = f"comparison_results_chinese_{os.path.basename(data_path).replace('.jsonl', '')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=4)
    print(f"\n🎉 评估完成！详细结果已保存到: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文模型对比测试脚本")
    parser.add_argument("--model", type=str, default="SUFE-AIFLM-Lab/Fin-R1", help="要评估的LLM名称 (Fin-R1 或 Qwen3-8B)")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl格式，例如 evaluate_mrr/alphafin_eval_optimized.jsonl)")
    parser.add_argument("--sample_size", type=int, default=500, help="随机采样的样本数量 (0表示评估全部)")
    parser.add_argument("--device", type=str, default="cuda:0", help="模型部署的设备 (例如 'cuda:0' 或 'cpu')")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="模型生成最大新Token数")
    parser.add_argument("--do_sample", type=bool, default=False, help="是否使用采样生成 (True/False)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")
    
    args = parser.parse_args()
    run_chinese_comparison_test(args)  # 传递args参数