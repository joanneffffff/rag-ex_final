#!/usr/bin/env python3
"""
英文模型性能评估脚本 - 对比 Fin-R1 和 Qwen3-8B 在英文数据集上的表现。
支持批量随机样本测试，并输出详细日志。
使用单GPU顺序加载和评估模型。
Prompt Template 内容从外部文件加载。
增加了 F1-score 和 Exact Match 的正确计算（针对英文）。
统计了输入/输出 Token 数和纯生成时间。
优化了答案提取逻辑以匹配英文Prompt中<answer>标签的严格要求。
"""

import warnings
import logging
import os
import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import argparse
from collections import Counter
from difflib import SequenceMatcher
import sys
import gc
import signal
import atexit

# 环境设置
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# --- 日志配置 ---
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f"english_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.log"

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

try:
    from tqdm import tqdm
except ImportError:
    logger.error("❌ tqdm is not installed. Please run: pip install tqdm")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig


# 导入配置文件 (请确保 config/parameters.py 存在并定义了 config.generator.cache_dir)
try:
    from config.parameters import config
    logger.info(f"✅ Using model cache path from config: {config.generator.cache_dir}")
except ImportError:
    logger.warning("⚠️ Config file not found. Using default model cache path '/users/sgjfei3/data/huggingface'")
    class Config:
        class Generator:
            cache_dir = "/users/sgjfei3/data/huggingface"
        generator = Generator()
    config = Config()

# ===================================================================
# 资源清理机制
# ===================================================================

class ResourceManager:
    """Manages resources to ensure proper cleanup upon script exit."""
    
    def __init__(self):
        self.model_loaders = {}
        self.cleanup_registered = False
        self._register_cleanup()
    
    def _register_cleanup(self):
        """Registers cleanup function for script exit and signals."""
        if not self.cleanup_registered:
            atexit.register(self.cleanup_resources)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.cleanup_registered = True
    
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful exit."""
        logger.info(f"\n🛑 Received signal {signum}, starting resource cleanup...")
        self.cleanup_resources()
        sys.exit(0)
    
    def add_model_loader(self, name: str, loader):
        """Adds a model loader reference to manage."""
        self.model_loaders[name] = loader
    
    def cleanup_resources(self):
        """Unloads all loaded models and clears GPU memory."""
        logger.info("🧹 Starting resource cleanup...")
        
        try:
            for name, loader in self.model_loaders.items():
                logger.info(f"🗑️ Cleaning model {name}...")
                loader.unload_model()
            self.model_loaders.clear()
            
            if torch.cuda.is_available():
                logger.info("🗑️ Clearing GPU memory...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"   GPU {i}: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")
            
            logger.info("🗑️ Forcing garbage collection...")
            gc.collect()
            
            logger.info("✅ Resource cleanup complete")
            
        except Exception as e:
            logger.error(f"⚠️ Error during resource cleanup: {e}")

resource_manager = ResourceManager()

# ===================================================================
# Answer Extraction and Normalization for English (Strict <answer> tag)
# ===================================================================

NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."

def _shared_text_standardizer_english(text: str) -> str:
    """
    Helper function to standardize English text for both answer extraction and F1 score calculation.
    Strictly follows the rules from the English Prompt Template.
    """
    text = text.strip()
    
    # Lowercase all text
    text = text.lower()

    # 递归替换所有 \text{...} 为 ...（保留内容）
    while True:
        new_text = re.sub(r'\\text\{([^}]*)\}', r'\1', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    # 其余 LaTeX 格式直接去掉
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove currency symbols and common unit words based on prompt rule
    text = re.sub(r'\b(million|billion|thousand|trillion|usd|eur|gbp|m|b)\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[\$£€]', '', text).strip()

    # Remove commas from numbers
    text = text.replace(',', '')

    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Normalize percentages
    text = text.replace(' percent', '%').replace('pct', '%')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    
    # Remove common introductory phrases
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation
    if text.endswith('%'):
        text = re.sub(r'[\.,;]$', '', text).strip()
    else:
        text = re.sub(r'[\.,;%]$', '', text).strip() 
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()

    return text

def extract_final_answer_from_tag(raw_output: str) -> str:
    """
    Extracts the final answer from the model's raw output by looking for the <answer> tag.
    Returns NOT_FOUND_REPLY_ENGLISH if no valid answer found or tag is empty.
    """
    NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."
    
    # First, try to find <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        # Ensure extracted content is not empty or an empty tag itself (e.g., <answer></answer>)
        if content and content.lower() not in ['<final></final>', '<answer></answer>', '<final-answer></final-answer>']:
            
            # Try to extract the most concise answer from the content
            # Look for patterns that might contain the actual answer
            
            # 1. Look for boxed answers: \boxed{...}
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
            if boxed_match:
                return _shared_text_standardizer_english(boxed_match.group(1))
            
            # 2. Look for percentage patterns: 12.82%
            percentage_match = re.search(r'(\d+\.?\d*)\s*%', content)
            if percentage_match:
                return _shared_text_standardizer_english(percentage_match.group(0))
            
            # 3. Look for numerical answers at the end of sentences
            # This is for cases like "Thus, the answer is 12.82%"
            final_number_match = re.search(r'(?:thus|therefore|answer is|result is)\s+(?:approximately\s+)?(\d+\.?\d*)', content, re.IGNORECASE)
            if final_number_match:
                return _shared_text_standardizer_english(final_number_match.group(1))
            
            # 4. Look for the largest numerical value (likely the answer)
            # This helps when there are multiple numbers in the text
            numbers = re.findall(r'\b(\d+(?:,\d+)*)\b', content)
            if numbers:
                # Convert to integers for comparison, removing commas
                number_values = [int(num.replace(',', '')) for num in numbers]
                largest_number = max(number_values)
                return _shared_text_standardizer_english(str(largest_number))
            
            # 5. If no specific pattern found, return the original content
            return _shared_text_standardizer_english(content)
    
    # If no <answer> tags found, look for boxed answers in the entire text
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', raw_output)
    if boxed_match:
        return _shared_text_standardizer_english(boxed_match.group(1))
    
    # If no valid <answer> structure is found or content is invalid,
    # return the specific "not found" phrase.
    return NOT_FOUND_REPLY_ENGLISH

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculates F1-score based on token overlap for English."""
    
    normalized_prediction = _shared_text_standardizer_english(prediction).lower()
    normalized_ground_truth = _shared_text_standardizer_english(ground_truth).lower()

    # Handle cases where the model explicitly states "I cannot find the answer..."
    if normalized_prediction == NOT_FOUND_REPLY_ENGLISH.lower():
        return 1.0 if normalized_ground_truth == NOT_FOUND_REPLY_ENGLISH.lower() else 0.0
    
    # Handle cases where the ground truth is "I cannot find the answer...", but the model gave a factual answer (which is an error)
    if normalized_ground_truth == NOT_FOUND_REPLY_ENGLISH.lower():
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if not ground_truth_tokens: 
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens: 
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: 
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculates Exact Match score for English."""
    return 1.0 if _shared_text_standardizer_english(prediction).lower() == _shared_text_standardizer_english(ground_truth).lower() else 0.0

# ===================================================================
# Prompt Formatting for English (Adapted from your base template)
# ===================================================================

def _load_template_content_from_file_english(template_file_name: str) -> str:
    """Loads the full string content of an English Prompt template from a specified file."""
    template_path = Path("data/prompt_templates") / template_file_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"❌ English Template file not found: {template_path}. Please ensure the file exists.")
        sys.exit(1)

def get_final_prompt_messages_english(context: str, query: str) -> List[Dict[str, str]]:
    """
    Constructs the messages list for English evaluation, using the specified template file.
    """
    template_file_name = "unified_english_template_no_think.txt"
    template_full_string = _load_template_content_from_file_english(template_file_name)

    messages = []

    # 1. Extract SYSTEM message (everything from ===SYSTEM=== to ===USER===)
    system_match = re.search(r'===SYSTEM===(.*?)===USER===', template_full_string, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
        # Clean up unwanted parts from SYSTEM content
        system_content = re.sub(r'---CRITICAL RULES for the <answer> tag[\s\S]*', '', system_content).strip()
        system_content = re.sub(r'---[\s\S]*', '', system_content).strip()
        messages.append({"role": "system", "content": system_content})
    
    # 2. Create USER message with the actual query and context
    # The template has examples, but we'll create a simple format for the actual query
    user_content = f"Q: {query}\nTable Context: {context}\nText Context: {context}\n<answer>"
    messages.append({"role": "user", "content": user_content})

    logger.debug(f"Constructed messages for prompt:\n{messages}")
    return messages


def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Converts messages list to ChatML format string expected by Fin-R1 (Qwen2.5 based).
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
            # Assistant role is usually part of few-shot examples
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"

    # Append assistant start tag to indicate model should start generating new assistant response
    formatted_prompt += "<|im_start|>assistant\n"

    logger.debug(f"Converted ChatML Prompt (first 500 chars):\n{formatted_prompt[:500]}...")
    return formatted_prompt


# ===================================================================
# Model Loader (No Changes Needed)
# ===================================================================

class ModelLoader:
    """Manages loading, unloading, and generating text from HuggingFace models."""
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device
        self.is_loaded = False

        self.cache_dir = config.generator.cache_dir # Use cache dir from config

        # Model path configurations
        if "Fin-R1" in model_name:
            local_fin_r1_path = f"{self.cache_dir}/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa"
            if os.path.exists(local_fin_r1_path):
                self.model_path = local_fin_r1_path
                logger.info(f"✅ [{self.model_name}] Using local cached model: {self.model_path}")
            else:
                self.model_path = "SUFE-AIFLM-Lab/Fin-R1"
                logger.warning(f"⚠️ [{self.model_name}] Local cache not found, will download from Hub: {self.model_path}")
        elif "Qwen3-8B" in model_name:
            local_qwen_path = f"{self.cache_dir}/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
            if os.path.exists(local_qwen_path):
                self.model_path = local_qwen_path
                logger.info(f"✅ [{self.model_name}] Using local cached model: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-8B"
                logger.warning(f"⚠️ [{self.model_name}] Local cache not found, will download from Hub: {self.model_path}")
        else:
            self.model_path = model_name
            logger.warning(f"⚠️ [{self.model_name}] Unknown model path '{model_name}', attempting to load from Hugging Face Hub.")

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    def load_model(self):
        if self.is_loaded:
            logger.info(f"✅ [{self.model_name}] Already loaded to {self.device}, no need to reload.")
            return

        logger.info(f"🔄 [{self.model_name}] Loading model to {self.device} from {self.model_path}")
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
            logger.info(f"🔧 [{self.model_name}] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"✅ [{self.model_name}] Tokenizer loaded. Chat Template: {self.tokenizer.chat_template}")

            logger.info(f"🔧 [{self.model_name}] Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            logger.info(f"✅ [{self.model_name}] Model loaded. Device: {self.model.device.type}:{self.model.device.index}, Quantization: 4bit")
            self.is_loaded = True
        except Exception as e:
            logger.exception(f"❌ [{self.model_name}] Model loading failed: {e}")
            self.unload_model()
            raise

    def unload_model(self):
        if not self.is_loaded:
            return

        logger.info(f"🗑️ [{self.model_name}] Unloading model and clearing GPU memory...")
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
            logger.info(f"✅ [{self.model_name}] GPU memory cleared.")
        except Exception as e:
            logger.error(f"❌ Error unloading [{self.model_name}]: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 512, do_sample: bool = False, repetition_penalty: float = 1.1) -> Dict[str, Any]:
        """
        Generates text given a ChatML formatted prompt string.
        Returns a dictionary with generated text, input/output token counts, and pure generation time.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded. Call load_model() first.")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(f"Model {self.model_name}'s tokenizer or model is None. Reload model.")

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

        logger.debug(f"[{self.model_name}] Input tokens: {input_token_count}, Output tokens: {output_token_count}, Gen Time: {end_gen_time - start_gen_time:.2f}s")
        return {
            "generated_text": generated_text,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "generation_time_pure": end_gen_time - start_gen_time
        }

# ===================================================================
# Main Evaluation Logic
# ===================================================================

def run_english_comparison_test(args):
    logger.info("🚀 English Model Comparison Test Started...")

    # 强制使用单GPU顺序模式
    device = args.device if hasattr(args, 'device') and args.device else "cuda:0"
    logger.info(f"✅ Using single GPU sequential mode on device: {device}")
    
    model_configs = [
        ("Fin-R1", device),
        ("Qwen3-8B", device)
    ]
    single_gpu_sequential_mode = True

    model_loaders = {}
    for name, dev in model_configs:
        model_loaders[name] = ModelLoader(name, dev)
        resource_manager.add_model_loader(name, model_loaders[name])

    data_path = args.data_path
    sample_size = args.sample_size
    # English evaluation requires a specific English Prompt Template
    template_file_name = "unified_english_template_no_think.txt" # Use existing template

    logger.info(f"📊 Loading dataset: {data_path}")
    try:
        # Load data for English evaluation (TatQA format usually)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        if sample_size > 0 and sample_size < len(dataset):
            import random
            random.seed(42)
            dataset = random.sample(dataset, sample_size)
            logger.info(f"✅ Randomly sampled {len(dataset)} samples for evaluation.")
        else:
            logger.info(f"✅ Loaded all {len(dataset)} samples for evaluation.")
            
        if not dataset:
            logger.error("❌ No samples loaded for evaluation. Please check dataset path and content.")
            return

    except Exception as e:
        logger.exception(f"❌ Dataset loading failed: {e}")
        return

    all_results_data = []

    if single_gpu_sequential_mode:
        logger.info("\n--- Entering Single GPU Sequential Evaluation Mode ---")
        for model_name, loader in model_loaders.items():
            try:
                logger.info(f"\n🔄 Loading model: {model_name} to {loader.device}")
                loader.load_model()
                logger.info(f"✅ Model {model_name} loaded. Starting evaluation...")
                
                model_specific_results = evaluate_model_on_dataset(
                    model_name, loader, dataset, template_file_name, # Pass template_file_name
                    args.max_new_tokens, args.do_sample, args.repetition_penalty,
                    save_frequency=args.save_frequency # Pass save_frequency
                )
                all_results_data.extend(model_specific_results)
                logger.info(f"\n--- {model_name} Evaluation Complete ---")
            except Exception as e:
                logger.exception(f"❌ Error during model {model_name} evaluation: {e}")
            finally:
                loader.unload_model()
                logger.info(f"✅ Model {model_name} Unloaded")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        logger.info(f"   GPU {i}: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")


    # --- Evaluation Complete, Save All Results ---
    output_filename = f"tatqa_comparison_results_{Path(args.data_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=4)
    logger.info(f"\n🎉 Evaluation Complete! Detailed results saved to: {output_filename}")

    # Summarize and print final comparison results
    logger.info("\n--- Final Model Comparison Summary ---")
    model_summaries = {}
    for result in all_results_data:
        model_name = result["model"]
        if model_name not in model_summaries:
            model_summaries[model_name] = {
                "total_f1": 0.0, "total_em": 0.0, "total_gen_time": 0.0,
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_smart_context_length": 0, "total_summary_length": 0, "total_original_context_length": 0,
                "count": 0
            }

        model_summaries[model_name]["total_f1"] += result["f1_score"]
        model_summaries[model_name]["total_em"] += result["exact_match"]
        model_summaries[model_name]["total_gen_time"] += result["generation_time_pure"]
        model_summaries[model_name]["total_input_tokens"] += result["input_token_count"]
        model_summaries[model_name]["total_output_tokens"] += result["output_token_count"]
        # English evaluation might not have these specific fields for every item, so use .get with default 0
        model_summaries[model_name]["total_smart_context_length"] += result.get("smart_context_length", 0)
        # Handle summary_length which might be string or int
        summary_length = result.get("summary_length", 0)
        if isinstance(summary_length, str):
            summary_length = len(summary_length)  # Convert string length to int
        model_summaries[model_name]["total_summary_length"] += summary_length
        model_summaries[model_name]["total_original_context_length"] += result.get("original_context_length", 0)
        model_summaries[model_name]["count"] += 1

    for model_name, data in model_summaries.items():
        if data["count"] > 0:
            avg_f1 = data["total_f1"] / data["count"]
            avg_em = data["total_em"] / data["count"]
            avg_gen_time = data["total_gen_time"] / data["count"]
            avg_input_tokens = data["total_input_tokens"] / data["count"]
            avg_output_tokens = data["total_output_tokens"] / data["count"]
            avg_smart_context_length = data["total_smart_context_length"] / data["count"]
            avg_summary_length = data["total_summary_length"] / data["count"]
            avg_original_context_length = data["total_original_context_length"] / data["count"]
        else:
            avg_f1, avg_em, avg_gen_time, avg_input_tokens, avg_output_tokens = 0.0, 0.0, 0.0, 0.0, 0.0
            avg_smart_context_length, avg_summary_length, avg_original_context_length = 0.0, 0.0, 0.0

        logger.info(f"\nModel: {model_name}")
        logger.info(f"  Evaluated Samples: {data['count']}")
        logger.info(f"  Average F1-score: {avg_f1:.4f}")
        logger.info(f"  Average Exact Match: {avg_em:.4f}")
        logger.info(f"  Average Generation Time (pure inference): {avg_gen_time:.2f} s/sample")
        logger.info(f"  Average Input Tokens: {avg_input_tokens:.1f}")
        logger.info(f"  Average Output Tokens: {avg_output_tokens:.1f}")
        logger.info(f"  Average Smart Context Length: {avg_smart_context_length:.1f} chars")
        logger.info(f"  Average Summary Length: {avg_summary_length:.1f} chars")
        logger.info(f"  Average Original Context Length: {avg_original_context_length:.1f} chars")
    logger.info("----------------------------")


def evaluate_model_on_dataset(model_name: str, loader: ModelLoader, dataset: List[Dict[str, Any]], template_file_name: str, max_new_tokens: int, do_sample: bool, repetition_penalty: float, save_frequency: int = 10, tqdm_position: int = 0) -> List[Dict[str, Any]]:
    """
    Evaluates a single model on a given dataset. This function will run in a separate thread.
    """
    model_results = []
    
    logger.info(f"\n[Thread] Starting evaluation for {model_name} on {loader.device}...")

    pbar = tqdm(dataset, desc=f"Evaluating {model_name} ({loader.device})", position=tqdm_position)
    for i, item in enumerate(pbar):
        # Extract fields from item, handling potential missing fields in English dataset
        query = item.get("query", "") # TatQA uses 'query'
        
        # TatQA context structure is a string containing the context information
        context_data_raw = item.get("context", "") 
        context_data = str(context_data_raw) if context_data_raw is not None else ""

        expected_answer = str(item.get("answer", "")) # Ensure answer is string
        doc_id = item.get("doc_id", f"sample_{i}") # Use doc_id if available

        # For English evaluation, typically no 'instruction' or 'summary' fields as in your Chinese dataset
        # However, for consistency with Chinese logging, we'll keep placeholders
        item_instruction = "" # Default to empty for English evaluation
        summary_content = "" # Default to empty for English evaluation

        # Calculate context lengths based on processed string
        # For TatQA, original_context_length might be based on list of dicts/string length
        # smart_context_length is the length of the string passed to the prompt
        original_context_length = len(str(context_data_raw)) # Rough length of original context representation
        smart_context_length = len(context_data) # Length after joining/converting to string

        # Get the Prompt messages using the English-adapted function
        messages = get_final_prompt_messages_english(context_data, query) # No instruction field in TatQA template
        
        # Convert messages to ChatML format for model generation
        prompt_string_for_model = _convert_messages_to_chatml(messages)

        try:
            gen_output = loader.generate(
                prompt_string=prompt_string_for_model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty
            )
            raw_generated_text = gen_output["generated_text"]
            
            # Use the English-specific answer extraction logic
            final_answer = extract_final_answer_from_tag(raw_generated_text) 
            
            f1 = calculate_f1_score(final_answer, expected_answer)
            em = calculate_exact_match(final_answer, expected_answer)

            model_results.append({
                "model": model_name,
                "sample_id": i,
                "doc_id": doc_id,
                "query": query,
                "expected_answer": expected_answer,
                "raw_generated_text": raw_generated_text,
                "final_answer": final_answer,
                "f1_score": f1,
                "exact_match": em,
                "generation_time_pure": gen_output["generation_time_pure"],
                "input_token_count": gen_output["input_token_count"],
                "output_token_count": gen_output["output_token_count"],
                "smart_context_length": smart_context_length, 
                "summary_length": summary_content, # Should be 0 for English, for log consistency
                "original_context_length": original_context_length 
            })
        except Exception as e:
            logger.exception(f"❌ [Thread] Error evaluating sample {i} for {model_name}: {e}")
            model_results.append({
                "model": model_name,
                "sample_id": i,
                "doc_id": doc_id,
                "query": query,
                "expected_answer": expected_answer,
                "raw_generated_text": "[ERROR]",
                "final_answer": "[ERROR]",
                "f1_score": 0.0,
                "exact_match": 0.0,
                "generation_time_pure": 0.0,
                "input_token_count": 0,
                "output_token_count": 0,
                "smart_context_length": smart_context_length,
                "summary_length": summary_content,
                "original_context_length": original_context_length,
                "error": str(e)
            })
        
        # Save partial results
        if (i + 1) % save_frequency == 0 or (i + 1) == len(dataset):
            partial_file = f"partial_results_{model_name}.json"
            try:
                with open(partial_file, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=4)
                logger.info(f"✅ Saved {len(model_results)} partial results to {partial_file}")
            except Exception as e:
                logger.error(f"⚠️ Failed to save partial results: {e}")
    return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TatQA English Model Comparison Script")
    parser.add_argument("--data_path", type=str, default="evaluate_mrr/tatqa_eval_balanced_100.jsonl", help="Path to the evaluation dataset file (JSONL format)")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of random samples to evaluate (0 for full dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action='store_true', help="Enable sampling during generation (True if flag is present, default False)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation")
    parser.add_argument("--save_frequency", type=int, default=10, help="Frequency for saving partial results (every N samples)")
    parser.add_argument("--device", type=str, default="", help="Specify GPU device (e.g., cuda:0, cuda:1). If not specified, will use automatic GPU detection.")
    
    args = parser.parse_args()
    run_english_comparison_test(args)