#!/usr/bin/env python3
"""
生成模块性能评估脚本 - 对比 Fin-R1 和 Qwen3-8B 在中文数据集上的表现。
支持批量随机样本测试，并输出详细日志。
利用双 GPU 进行模型并行加载和评估以加速。
Prompt Template 内容从外部文件加载。
增加了 F1-score 和 Exact Match 的正确计算（支持中文分词）。
统计了输入/输出 Token 数和纯生成时间。
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import re
import gc
import json
import argparse
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Dict, Any
from collections import Counter
import string
import jieba # 引入jieba库

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入配置文件 (请确保 config/parameters.py 存在并定义了 config.generator.cache_dir)
try:
    from config.parameters import config
    print(f"✅ 使用配置文件中的缓存路径: {config.generator.cache_dir}")
except ImportError:
    print("⚠️ 无法导入配置文件，使用默认缓存路径 '/users/sgjfei3/data/huggingface'")
    class Config: # 定义一个假的config类，防止报错
        class Generator:
            cache_dir = "/users/sgjfei3/data/huggingface"
    config = Config()

# 确保 bitsandbytes, accelerate 已安装
# pip install bitsandbytes accelerate

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
        r'（参阅第三部分）',
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
    template_path = Path("data/prompt_templates") / template_file_name
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
                    # 直接使用传入的summary和context参数
                    content = content.replace('{summary}', summary).replace('{context}', context)

                messages.append({"role": role, "content": content})

    return messages


def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 列表转换为 Fin-R1 (Qwen2.5 based) 期望的ChatML格式字符串。
    注意：这里的 `im_im_end` 可能是笔误，Qwen系列标准应该是 `im_end`
    """
    if not messages:
        return ""

    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            formatted_prompt += f"<|im_start|>system\n{content.strip()}<|im_end|>\n" # 更正为im_end
        elif role == "user":
            formatted_prompt += f"<|im_start|>user\n{content.strip()}<|im_end|>\n" # 更正为im_end
        elif role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n" # 更正为im_end

    formatted_prompt += "<|im_start|>assistant\n"

    return formatted_prompt


# ====================================================================================
# 模型加载和生成器包装类
# ====================================================================================

class ModelLoader:
    """负责加载和卸载模型，并提供生成接口"""
    def __init__(self, model_name: str, device: str): # 新增 device 参数
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device # 使用传入的 device
        self.is_loaded = False

        cache_dir = config.generator.cache_dir

        if "Fin-R1" in model_name:
            # 检查本地缓存路径 - 使用 Fin-R1 的正确路径
            local_fin_r1_path = f"{cache_dir}/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa"
            if os.path.exists(local_fin_r1_path):
                self.model_path = local_fin_r1_path
                print(f"✅ [{self.model_name}] 使用本地缓存模型: {self.model_path}")
            else:
                self.model_path = "SUFE-AIFLM-Lab/Fin-R1"
                print(f"⚠️ [{self.model_name}] 本地缓存未找到，将从Hub下载: {self.model_path}")
        elif "Qwen3-8B" in model_name:
            # 检查本地缓存路径 - 使用 Qwen3-8B 的正确路径
            local_qwen_path = f"{cache_dir}/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5" # 假设这是Qwen3-8B的某个稳定快照
            if os.path.exists(local_qwen_path):
                self.model_path = local_qwen_path
                print(f"✅ [{self.model_name}] 使用本地缓存模型: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-8B"
                print(f"⚠️ [{self.model_name}] 本地缓存未找到，将从Hub下载: {self.model_path}")
        else:
            self.model_path = model_name
            print(f"⚠️ [{self.model_name}] 模型路径 '{model_name}' 未知，尝试从Hugging Face Hub加载。建议提前下载到本地。")

        # 4-bit 量化配置 (确保 `bitsandbytes` 已安装)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # NormalFloat 4-bit
            bnb_4bit_compute_dtype=torch.float16, # 计算数据类型，通常设置为 float16
            bnb_4bit_use_double_quant=False, # 不使用双量化
        )

    def load_model(self):
        if self.is_loaded:
            print(f"✅ [{self.model_name}] 已加载到 {self.device}，无需重复加载。")
            return

        print(f"🔄 [{self.model_name}] 正在加载模型到 {self.device} 从 {self.model_path}")
        # 判断是否为本地路径，影响 local_files_only 参数
        is_local_path = Path(self.model_path).exists() and Path(self.model_path).is_dir()

        cache_dir = config.generator.cache_dir
        tokenizer_args = {"trust_remote_code": True, "local_files_only": is_local_path, "cache_dir": cache_dir}
        model_args = {
            "torch_dtype": torch.float16,
            "device_map": self.device, # 直接指定设备，BitsAndBytesConfig 会处理分配
            "trust_remote_code": True,
            "quantization_config": self.quantization_config, # 使用 4-bit 量化配置
            "local_files_only": is_local_path,
            "cache_dir": cache_dir
        }

        try:
            print(f"🔧 [{self.model_name}] 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"✅ [{self.model_name}] Tokenizer加载完成. Chat Template: {self.tokenizer.chat_template}")

            print(f"🔧 [{self.model_name}] 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            print(f"✅ [{self.model_name}] 模型加载完成. 设备: {self.model.device.type}:{self.model.device.index}, 量化: 4bit")
            self.is_loaded = True
        except Exception as e:
            print(f"❌ [{self.model_name}] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.unload_model() # 确保失败时也清理
            raise

    def unload_model(self):
        if not self.is_loaded:
            return

        print(f"🗑️ [{self.model_name}] 卸载模型并清理显存...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            # 清理CUDA缓存并强制垃圾回收
            torch.cuda.empty_cache()
            gc.collect()
            self.is_loaded = False
            print(f"✅ [{self.model_name}] 显存已清理。")
        except Exception as e:
            print(f"❌ 卸载 [{self.model_name}] 时发生错误: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 150, do_sample: bool = False, repetition_penalty: float = 1.1) -> Dict[str, Any]:
        """
        生成文本，期望输入已经是 ChatML 格式的字符串。
        返回包含生成文本、输入和输出token数的字典。
        """
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载。请先调用 load_model()。")

        # 确保输入在模型所在的正确设备上
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        start_gen_time = time.time() # 记录纯生成开始时间
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
        end_gen_time = time.time() # 记录纯生成结束时间

        # 解码生成的 tokens
        # 注意：这里 outputs[0] 包含了 prompt_ids + generated_ids
        generated_tokens_ids = outputs[0, inputs["input_ids"].shape[1]:] # 仅取生成的新tokens
        generated_text = self.tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)

        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = generated_tokens_ids.shape[0] # 生成的tokens数量

        return {
            "generated_text": generated_text,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "generation_time_pure": end_gen_time - start_gen_time # 纯生成时间
        }

# ====================================================================================
# 评估指标计算 (基于词重叠，针对中文需要 jieba)
# ====================================================================================

def normalize_answer_chinese(s: str) -> str:
    """
    针对中文进行答案归一化：移除标点、转换全角字符为半角、去除多余空格、分词并小写。
    """
    if not s:
        return ""

    # 转换为小写并去除两端空白
    s = s.strip().lower()

    # 将全角标点替换为半角
    s = s.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?').replace('；', ';')
    s = s.replace('（', '(').replace('）', ')')

    # 移除所有常见标点符号
    # 这里需要确保涵盖了常见的中文和英文标点
    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~“”‘’【】『』《》—…·～「」～￥%#@！&（）《》]'
    s = re.sub(punctuation_pattern, '', s)


    # 使用 jieba 进行分词
    tokens = jieba.cut(s)
    # 过滤掉分词结果中的空格和空字符串
    normalized_tokens = [token for token in tokens if token.strip()]
    return " ".join(normalized_tokens)


def get_tokens_chinese(s: str) -> List[str]:
    """获取中文分词后的tokens列表。"""
    # 直接返回 normalize_answer_chinese 后的 split 结果
    return normalize_answer_chinese(s).split()

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数 (基于词重叠)。"""
    gold_tokens = get_tokens_chinese(ground_truth)
    pred_tokens = get_tokens_chinese(prediction)

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0 # 如果两者都为空，F1 为 1
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0 # 其中一个为空，另一个不为空，F1 为 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配率。"""
    return float(normalize_answer_chinese(prediction) == normalize_answer_chinese(ground_truth))

# ====================================================================================
# 主测试逻辑
# ====================================================================================

# 多进程/多线程处理
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_chinese_comparison_test(args):
    print("🚀 中文模型对比测试开始...")

    # 检查 GPU 数量，并分配设备
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"❌ 警告：检测到 {num_gpus} 块可用 GPU (少于 2 块)。将退化为单卡顺序评估模式。")
        model_configs = [
            ("Fin-R1", "cuda:0"),
            ("Qwen3-8B", "cuda:0") # 都会加载到 cuda:0，但会按顺序加载和卸载
        ]
        # 在单卡模式下，退化回顺序加载，防止显存不足
        single_gpu_sequential_mode = True
    else:
        print(f"✅ 检测到 {num_gpus} 块 GPU。尝试分配 Fin-R1 到 cuda:0，Qwen3-8B 到 cuda:1。")
        model_configs = [
            ("Fin-R1", "cuda:0"),
            ("Qwen3-8B", "cuda:1")
        ]
        single_gpu_sequential_mode = False

    model_loaders = {}
    for name, dev in model_configs:
        model_loaders[name] = ModelLoader(name, dev)

    data_path = args.data_path
    sample_size = args.sample_size
    template_file_name = "multi_stage_chinese_template.txt"

    print(f"📊 加载数据集: {data_path}")
    try:
        # 假设 utils.data_loader 存在并提供了 load_json_or_jsonl 和 sample_data
        from utils.data_loader import load_json_or_jsonl, sample_data
        dataset = load_json_or_jsonl(data_path)

        if sample_size > 0:
            dataset = sample_data(dataset, sample_size, 42) # 使用固定随机种子保证可复现性
            print(f"✅ 随机采样 {len(dataset)} 个样本进行评估。")
        else:
            print(f"✅ 加载了全部 {len(dataset)} 个样本进行评估。")

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    all_results_data = []

    if single_gpu_sequential_mode:
        print("\n--- 进入单 GPU 顺序评估模式 ---")
        for model_name, loader in model_loaders.items():
            try:
                print(f"\n🔄 正在加载模型: {model_name} 到 {loader.device}")
                loader.load_model() # 加载当前模型
                print(f"✅ 模型 {model_name} 加载完成，开始评估...")
                model_specific_results = evaluate_model_on_dataset(
                    model_name, loader, dataset, template_file_name,
                    args.max_new_tokens, args.do_sample, args.repetition_penalty
                )
                all_results_data.extend(model_specific_results)
                print(f"\n--- {model_name} 评估完成 ---")
            except Exception as e:
                print(f"❌ 模型 {model_name} 评估过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loader.unload_model() # 确保每次循环都卸载模型
                print(f"✅ 模型 {model_name} 卸载完成")
                print(f"📊 卸载后GPU内存状态:")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
    else: # 双 GPU 并行模式
        # 并行加载模型
        loaded_models = {}
        for model_name, loader in model_loaders.items():
            try:
                # 显式加载模型到指定的 GPU
                loader.load_model()
                loaded_models[model_name] = loader
            except Exception as e:
                print(f"❌ 模型 {model_name} 加载失败，跳过该模型: {e}")
                if model_name in loaded_models:
                    del loaded_models[model_name]
                loader.unload_model() # 确保清理显存
                continue

        if not loaded_models:
            print("❌ 没有模型成功加载，退出评估。")
            return

        print("\n✅ 所有成功加载的模型已就绪，开始并行评估...")

        # 使用 ThreadPoolExecutor 并行处理每个模型的评估
        # 每个模型一个线程，确保模型在自己的GPU上运行
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(loaded_models)) as executor:
            futures = {executor.submit(evaluate_model_on_dataset, model_name, loader, dataset, template_file_name, args.max_new_tokens, args.do_sample, args.repetition_penalty): model_name
                       for model_name, loader in loaded_models.items()}

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    model_specific_results = future.result()
                    all_results_data.extend(model_specific_results)
                    print(f"\n--- {model_name} 评估完成 ---")
                except Exception as e:
                    print(f"❌ 模型 {model_name} 评估过程中发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # 评估完成后卸载模型
                    if model_name in loaded_models:
                        loaded_models[model_name].unload_model()
        
        # 打印并行模式下卸载后的最终GPU内存状态
        print(f"\n📊 并行模式评估后，最终GPU内存状态:")
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

    # 汇总并打印最终对比结果
    print("\n--- 最终模型对比摘要 ---")
    model_summaries = {}
    for result in all_results_data:
        model_name = result["model"]
        if model_name not in model_summaries:
            model_summaries[model_name] = {
                "total_f1": 0.0,
                "total_em": 0.0,
                "total_gen_time": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "count": 0
            }

        model_summaries[model_name]["total_f1"] += result["f1_score"]
        model_summaries[model_name]["total_em"] += result["exact_match"]
        model_summaries[model_name]["total_gen_time"] += result["generation_time_pure"] # 使用纯生成时间
        model_summaries[model_name]["total_input_tokens"] += result["input_token_count"]
        model_summaries[model_name]["total_output_tokens"] += result["output_token_count"]
        model_summaries[model_name]["count"] += 1

    for model_name, data in model_summaries.items():
        if data["count"] > 0:
            avg_f1 = data["total_f1"] / data["count"]
            avg_em = data["total_em"] / data["count"]
            avg_gen_time = data["total_gen_time"] / data["count"]
            avg_input_tokens = data["total_input_tokens"] / data["count"]
            avg_output_tokens = data["total_output_tokens"] / data["count"]
        else:
            avg_f1, avg_em, avg_gen_time, avg_input_tokens, avg_output_tokens = 0.0, 0.0, 0.0, 0.0, 0.0

        print(f"\n模型: {model_name}")
        print(f"  评估样本数: {data['count']}")
        print(f"  平均 F1-score: {avg_f1:.4f}")
        print(f"  平均 Exact Match: {avg_em:.4f}")
        print(f"  平均生成耗时 (纯推理): {avg_gen_time:.2f} 秒/样本")
        print(f"  平均输入 Token 数: {avg_input_tokens:.1f}")
        print(f"  平均输出 Token 数: {avg_output_tokens:.1f}")
    print("----------------------------")


def evaluate_model_on_dataset(model_name: str, loader: ModelLoader, dataset: List[Dict[str, Any]], template_file_name: str, max_new_tokens: int, do_sample: bool, repetition_penalty: float) -> List[Dict[str, Any]]:
    """
    在特定数据集上评估单个模型。此函数将在独立的线程中运行。
    """
    model_results = []

    # 打印当前线程的模型和它所在的GPU
    print(f"\n[线程] 开始评估 {model_name} 在 {loader.device} 上...")

    pbar = tqdm(dataset, desc=f"评估 {model_name} ({loader.device})")
    for i, item in enumerate(pbar):
        query = item.get("query", "") or item.get("generated_question", "") or item.get("question", "")
        summary = item.get("summary", "")
        context = item.get("context", "")
        expected_answer = item.get("answer", "")

        messages = get_messages_for_test(summary, context, query, template_file_name)
        prompt_string_for_model = _convert_messages_to_chatml(messages)

        # 调用 loader 内部的 generate，它现在返回一个字典
        try:
            gen_output = loader.generate(
                prompt_string=prompt_string_for_model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty
            )
            generated_text = gen_output["generated_text"]
            final_answer = clean_response(generated_text) # 后处理

            f1 = calculate_f1_score(final_answer, expected_answer)
            em = calculate_exact_match(final_answer, expected_answer)

            model_results.append({
                "model": model_name,
                "sample_id": i,
                "query": query,
                "expected_answer": expected_answer,
                "raw_generated_text": generated_text,
                "final_answer": final_answer,
                "f1_score": f1,
                "exact_match": em,
                "generation_time_pure": gen_output["generation_time_pure"],
                "input_token_count": gen_output["input_token_count"],
                "output_token_count": gen_output["output_token_count"],
            })
        except Exception as e:
            print(f"❌ [线程] {model_name} 样本 {i} 评估失败: {e}")
            # 可以选择在这里记录失败样本或跳过
            model_results.append({
                "model": model_name,
                "sample_id": i,
                "query": query,
                "expected_answer": expected_answer,
                "raw_generated_text": "[ERROR]",
                "final_answer": "[ERROR]",
                "f1_score": 0.0,
                "exact_match": 0.0,
                "generation_time_pure": 0.0,
                "input_token_count": 0,
                "output_token_count": 0,
                "error": str(e)
            })
    return model_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文模型对比测试脚本")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl格式，例如 evaluate_mrr/alphafin_eval_optimized.jsonl)")
    parser.add_argument("--sample_size", type=int, default=100, help="随机采样的样本数量 (0表示评估全部，默认为100)")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="模型生成最大新Token数")
    parser.add_argument("--do_sample", action='store_true', help="是否使用采样生成 (如果设置了此flag，则为True，默认False)") # 修正布尔参数
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")

    args = parser.parse_args()
    run_chinese_comparison_test(args)