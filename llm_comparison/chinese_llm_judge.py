import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
import time
import re
import gc
import json
import argparse
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Dict, Any
import logging

# 添加项目根目录到Python路径 (如果需要)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- 日志配置 (与你的评估脚本保持一致) ---
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"llm_judge_{time.strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 导入配置文件 (确保存在)
try:
    from config.parameters import config
    logger.info(f"✅ 使用配置文件中的缓存路径: {config.generator.cache_dir}")
except ImportError:
    logger.warning("⚠️ 无法导入配置文件，使用默认缓存路径 '/users/sgjfei3/data/huggingface'")
    class Config:
        class Generator:
            cache_dir = "/users/sgjfei3/data/huggingface"
        generator = Generator()
    config = Config()

# --- 模型加载器 (与你评估脚本中的 ModelLoader 保持一致) ---
class ModelLoader:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device
        self.is_loaded = False
        self.cache_dir = config.generator.cache_dir # 使用配置的缓存目录

        # 根据模型名称设置路径
        if "Qwen3-8B" in model_name:
            local_qwen_path = f"{self.cache_dir}/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
            if os.path.exists(local_qwen_path):
                self.model_path = local_qwen_path
                logger.info(f"✅ [{self.model_name}] 使用本地缓存模型: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-8B"
                logger.warning(f"⚠️ [{self.model_name}] 本地缓存未找到，将从Hub下载: {self.model_path}")
        else:
            self.model_path = model_name
            logger.warning(f"⚠️ [{self.model_name}] 模型路径 '{model_name}' 未知，尝试从Hugging Face Hub加载。")

        # 4-bit 量化配置
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    def load_model(self):
        if self.is_loaded:
            logger.info(f"✅ [{self.model_name}] 已加载到 {self.device}，无需重复加载。")
            return

        logger.info(f"🔄 [{self.model_name}] 正在加载模型到 {self.device} 从 {self.model_path}")
        is_local_path = Path(self.model_path).exists() and Path(self.model_path).is_dir()

        tokenizer_args = {"trust_remote_code": True, "local_files_only": is_local_path, "cache_dir": self.cache_dir}
        model_args = {
            "torch_dtype": torch.float16,
            "device_map": self.device,
            "trust_remote_code": True,
            "quantization_config": self.quantization_config,
            "local_files_only": is_local_path,
            "cache_dir": self.cache_dir
        }

        try:
            logger.info(f"🔧 [{self.model_name}] 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"✅ [{self.model_name}] Tokenizer加载完成. Chat Template: {self.tokenizer.chat_template}")

            logger.info(f"🔧 [{self.model_name}] 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            logger.info(f"✅ [{self.model_name}] 模型加载完成. 设备: {self.model.device.type}:{self.model.device.index}, 量化: 4bit")
            self.is_loaded = True
        except Exception as e:
            logger.exception(f"❌ [{self.model_name}] 模型加载失败: {e}")
            self.unload_model()
            raise

    def unload_model(self):
        if not self.is_loaded:
            return

        logger.info(f"🗑️ [{self.model_name}] 卸载模型并清理显存...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            self.is_loaded = False
            logger.info(f"✅ [{self.model_name}] 显存已清理。")
        except Exception as e:
            logger.error(f"❌ 卸载 [{self.model_name}] 时发生错误: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 512, do_sample: bool = False, repetition_penalty: float = 1.1) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载。请先调用 load_model()。")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(f"模型 {self.model_name} 的tokenizer或model为None。请重新加载模型。")

        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        start_gen_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
        end_gen_time = time.time()

        generated_tokens_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)

        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = generated_tokens_ids.shape[0]

        logger.debug(f"[{self.model_name}] 输入tokens: {input_token_count}, 输出tokens: {output_token_count}, 生成时间: {end_gen_time - start_gen_time:.2f}s")
        return {
            "generated_text": generated_text,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "generation_time_pure": end_gen_time - start_gen_time
        }

# --- Prompt 构建辅助函数 (用于 LLM-Judge) ---
def _load_judge_template(template_file_name: str) -> str:
    template_path = Path("data/prompt_templates") / template_file_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"❌ LLM-Judge 模板文件未找到: {template_path}，请确保文件存在。")
        sys.exit(1)

def get_judge_messages(query: str, expected_answer: str, model_final_answer: str, template_file_name: str) -> List[Dict[str, str]]:
    template_full_string = _load_judge_template(template_file_name)

    # 替换 Prompt 模板中的占位符
    # 使用安全的替换方式，确保只替换一次
    system_part = re.search(r'===SYSTEM===(.*?)===USER===', template_full_string, re.DOTALL)
    user_part = re.search(r'===USER===(.*?)===ASSISTANT===', template_full_string, re.DOTALL)

    messages = []
    if system_part:
        messages.append({"role": "system", "content": system_part.group(1).strip()})
    
    if user_part:
        user_content = user_part.group(1).strip()
        user_content = user_content.replace('{query}', query)
        user_content = user_content.replace('{expected_answer}', expected_answer)
        user_content = user_content.replace('{model_final_answer}', model_final_answer)
        messages.append({"role": "user", "content": user_content})

    # Judge 的 Prompt 通常以 ASSISTANT 的开始标签结束，等待其生成评分JSON
    messages.append({"role": "assistant", "content": ""}) # Judge 预期生成内容，所以content为空字符串

    # 将 messages 列表转换为 ChatML 格式
    formatted_prompt = ""
    for msg in messages[:-1]: # 最后一个是 ASSISTANT 的开始标签，不需要 im_end
        formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    formatted_prompt += f"<|im_start|>{messages[-1]['role']}\n" # 添加最后一个 ASSISTANT 的开始标签

    logger.debug(f"Judge Prompt (前500字符):\n{formatted_prompt[:500]}...")
    return formatted_prompt

# --- LLM-Judge 主逻辑 ---
def run_llm_judge_evaluation(args):
    logger.info("🤖 LLM-Judge 评估开始...")

    # 加载 Judge 模型 (Qwen3-8B，通常放在 cuda:0)
    judge_model_name = "Qwen3-8B"
    judge_loader = ModelLoader(judge_model_name, "cuda:0")
    try:
        judge_loader.load_model()
    except Exception as e:
        logger.error(f"❌ LLM-Judge 模型 {judge_model_name} 加载失败，无法进行评估: {e}")
        return

    # 加载生成模型的评估结果文件
    if not os.path.exists(args.results_file):
        logger.error(f"❌ 未找到生成模型的评估结果文件: {args.results_file}")
        judge_loader.unload_model()
        return

    with open(args.results_file, 'r', encoding='utf-8') as f:
        generated_results = json.load(f)

    judge_template_file = args.judge_template_file # 例如 "qwen_judge_template.txt"
    judged_results = []

    logger.info(f"📊 开始用 {judge_model_name} 评估 {len(generated_results)} 个生成结果...")
    
    # 对每个生成结果进行 Judge 评估
    pbar = tqdm(generated_results, desc=f"LLM-Judge 评估中 ({judge_model_name})")
    for item in pbar:
        query = item["query"]
        expected_answer = item["expected_answer"]
        model_final_answer = item["final_answer"]
        model_name = item["model"]
        sample_id = item["sample_id"]

        judge_prompt = get_judge_messages(query, expected_answer, model_final_answer, judge_template_file)

        try:
            judge_output = judge_loader.generate(
                prompt_string=judge_prompt,
                max_new_tokens=args.judge_max_new_tokens, # Judge 生成通常也需要一些token
                do_sample=False, # Judge 评分通常不使用采样，追求确定性
                repetition_penalty=1.0 # Judge 评分通常没有重复惩罚
            )
            judge_raw_text = judge_output["generated_text"]
            
            # 尝试解析 Judge 的 JSON 输出
            judge_score_data = {}
            try:
                # 提取第一个JSON块，因为LLM可能会在JSON前后说废话
                json_match = re.search(r'\{[\s\S]*\}', judge_raw_text)
                if json_match:
                    json_string = json_match.group(0)
                    judge_score_data = json.loads(json_string)
                else:
                    logger.warning(f"⚠️ Judge Output for Sample {sample_id} ({model_name}) 无JSON输出: {judge_raw_text[:100]}...")
                    judge_score_data = {"Accuracy_Score": 0, "Conciseness_Score": 0, "Professionalism_Score": 0, "Reasoning": "Judge输出不含JSON"}

            except json.JSONDecodeError as json_e:
                logger.error(f"❌ Judge Output for Sample {sample_id} ({model_name}) JSON解析失败: {json_e} - Raw: {judge_raw_text[:200]}...")
                judge_score_data = {"Accuracy_Score": 0, "Conciseness_Score": 0, "Professionalism_Score": 0, "Reasoning": f"JSON解析失败: {judge_raw_text[:50]}"}
            
            judged_results.append({
                "generator_model": model_name,
                "sample_id": sample_id,
                "query": query,
                "expected_answer": expected_answer,
                "generator_final_answer": model_final_answer,
                "judge_raw_output": judge_raw_text,
                "accuracy_score": judge_score_data.get("Accuracy_Score", 0),
                "conciseness_score": judge_score_data.get("Conciseness_Score", 0),
                "professionalism_score": judge_score_data.get("Professionalism_Score", 0),
                "judge_reasoning": judge_score_data.get("Reasoning", "")
            })

        except Exception as e:
            logger.exception(f"❌ 调用 Judge 模型评估 Sample {sample_id} ({model_name}) 失败: {e}")
            judged_results.append({
                "generator_model": model_name,
                "sample_id": sample_id,
                "query": query,
                "expected_answer": expected_answer,
                "generator_final_answer": model_final_answer,
                "judge_raw_output": "[ERROR]",
                "accuracy_score": 0, "conciseness_score": 0, "professionalism_score": 0,
                "judge_reasoning": str(e)
            })
    
    # 保存 LLM-Judge 评估结果
    output_filename = f"llm_judge_results_{Path(args.results_file).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(judged_results, f, ensure_ascii=False, indent=4)
    logger.info(f"🎉 LLM-Judge 评估完成！结果已保存到: {output_filename}")

    # 汇总并打印最终 Judge 评估摘要
    logger.info("\n--- LLM-Judge 评估摘要 ---")
    model_summaries = {}
    for result in judged_results:
        model_name = result["generator_model"]
        if model_name not in model_summaries:
            model_summaries[model_name] = {
                "total_accuracy": 0,
                "total_conciseness": 0,
                "total_professionalism": 0,
                "count": 0
            }
        model_summaries[model_name]["total_accuracy"] += result["accuracy_score"]
        model_summaries[model_name]["total_conciseness"] += result["conciseness_score"]
        model_summaries[model_name]["total_professionalism"] += result["professionalism_score"]
        model_summaries[model_name]["count"] += 1

    for model_name, data in model_summaries.items():
        if data["count"] > 0:
            avg_accuracy = data["total_accuracy"] / data["count"]
            avg_conciseness = data["total_conciseness"] / data["count"]
            avg_professionalism = data["total_professionalism"] / data["count"]
        else:
            avg_accuracy, avg_conciseness, avg_professionalism = 0.0, 0.0, 0.0

        logger.info(f"\n生成模型: {model_name}")
        logger.info(f"  评估样本数: {data['count']}")
        logger.info(f"  平均准确性分数: {avg_accuracy:.2f}")
        logger.info(f"  平均简洁性分数: {avg_conciseness:.2f}")
        logger.info(f"  平均专业性分数: {avg_professionalism:.2f}")
    logger.info("----------------------------")
    
    judge_loader.unload_model() # 卸载 Judge 模型

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Judge 评估脚本")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="包含生成模型输出的JSON文件路径 (例如 comparison_results_chinese_*.json)")
    parser.add_argument("--judge_template_file", type=str, default="qwen_judge_template.txt",
                        help="LLM Judge 的 Prompt 模板文件 (例如 qwen_judge_template.txt)")
    parser.add_argument("--judge_max_new_tokens", type=int, default=256,
                        help="LLM Judge 生成的最大新 Token 数")

    args = parser.parse_args()
    run_llm_judge_evaluation(args)