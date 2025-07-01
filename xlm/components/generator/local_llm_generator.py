import os
import json
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from xlm.components.generator.generator import Generator
from config.parameters import Config


class LocalLLMGenerator(Generator):
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_quantization: Optional[bool] = None,
        quantization_type: Optional[str] = None,
        use_flash_attention: bool = False
    ):
        # 使用config中的平台感知配置
        self.config = Config()
        
        # 如果没有提供model_name，从config读取
        if model_name is None:
            model_name = self.config.generator.model_name
        
        # 如果没有提供device，从config读取
        if device is None:
            device = self.config.generator.device
        
        # 如果没有提供量化参数，从config读取
        if use_quantization is None:
            use_quantization = self.config.generator.use_quantization
        if quantization_type is None:
            quantization_type = self.config.generator.quantization_type
        
        super().__init__(model_name=model_name)
        self.device = device
        self.temperature = self.config.generator.temperature
        self.max_new_tokens = self.config.generator.max_new_tokens
        self.top_p = self.config.generator.top_p
        
        # 使用config中的平台感知配置
        if cache_dir is None:
            cache_dir = self.config.generator.cache_dir  # 使用generator的缓存目录
        
        self.cache_dir = cache_dir  # 关键修正，确保属性存在
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.use_flash_attention = use_flash_attention
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置环境变量
        # os.environ['HF_HOME'] = 'D:/AI/huggingface'
        os.environ['HF_HOME'] = self.cache_dir
        # os.environ['TRANSFORMERS_CACHE'] = os.path.join('D:/AI/huggingface', 'transformers')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        print(f"LocalLLMGenerator '{model_name}' loaded on {self.device} with quantization: {self.use_quantization} ({self.quantization_type}).")
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare model loading arguments
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": True,
            "use_cache": True,
        }

        # 根据配置应用量化
        if self.use_quantization and self.device and self.device.startswith('cuda'):
            print(f"CUDA device detected. Applying {self.quantization_type} quantization...")
            
            if self.quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
            elif self.quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                print(f"Unknown quantization type: {self.quantization_type}, falling back to 4bit")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
            
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.device  # 明确指定设备
        else:
            if self.device and self.device.startswith('cuda'):
                print("CUDA device detected but quantization disabled. Loading model without quantization.")
                model_kwargs["device_map"] = self.device  # 明确指定设备
                model_kwargs["torch_dtype"] = torch.float16
            else:
                print("CPU device detected. Loading model without quantization.")
                model_kwargs["device_map"] = "cpu"
                model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
    
    def convert_to_json_chat_format(self, text):
        """将单一字符串转换为JSON聊天格式"""
        
        # 检测是否包含系统指令
        if "你是一位专业的金融分析师" in text:
            # 提取system部分 - 查找系统指令的开始和结束
            system_start = text.find("你是一位专业的金融分析师")
            
            # 查找系统指令的结束位置（通常是"【公司财务报告摘要】"或"【公司财务报告片段】"之前）
            context_markers = ["【公司财务报告摘要】", "【公司财务报告片段】", "【完整公司财务报告片段】"]
            context_start = -1
            for marker in context_markers:
                pos = text.find(marker)
                if pos != -1:
                    context_start = pos
                    break
            
            if system_start != -1 and context_start != -1:
                system_content = text[system_start:context_start].strip()
                user_content = text[context_start:].strip()
                
                # 构造JSON格式
                json_chat = [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ]
                
                return json.dumps(json_chat, ensure_ascii=False)
        
        return text

    def convert_json_to_fin_r1_format(self, json_chat):
        """将JSON格式转换为Fin-R1聊天格式"""
        
        try:
            chat_data = json.loads(json_chat)
            fin_r1_format = ""
            
            for message in chat_data:
                if message["role"] == "system":
                    fin_r1_format += f'<|im_start|>system\n{message["content"]}<|im_end|>\n'
                elif message["role"] == "user":
                    fin_r1_format += f'<|im_start|>user\n{message["content"]}<|im_end|>\n'
            
            fin_r1_format += '<|im_start|>assistant\n'
            return fin_r1_format
            
        except json.JSONDecodeError:
            return json_chat
        
    def generate(self, texts: List[str]) -> List[str]:
        responses = []
        for text in texts:
            # 调试信息：打印设备状态
            print(f"Generator device: {self.device}")
            print(f"Model device: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'Unknown'}")
            
            # 确保tokenizer在正确的设备上
            if hasattr(self.tokenizer, 'device') and self.tokenizer.device != self.device:
                print(f"Warning: Tokenizer device mismatch. Moving to {self.device}")
            
            # 检查输入长度
            print(f"Input text length: {len(text)} characters")
            
            # 检查Fin-R1是否支持聊天格式
            if "Fin-R1" in self.model_name:
                print("Fin-R1 detected, converting to JSON chat format...")
                # 转换为JSON格式
                json_chat = self.convert_to_json_chat_format(text)
                print(f"JSON chat format length: {len(json_chat)} characters")
                
                # 转换为Fin-R1格式
                text = self.convert_json_to_fin_r1_format(json_chat)
                print(f"Converted to Fin-R1 format, length: {len(text)} characters")
            else:
                print("Non-Fin-R1 model detected, using original format...")
            
            # 优化：直接用tokenizer.__call__处理padding和truncation
            # 移除max_length限制，完全避免截断
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,  # 完全禁用截断
                padding=False,     # 改为False，避免不必要的padding
                add_special_tokens=True
            )
            
            print(f"Tokenized input length: {inputs['input_ids'].shape[1]} tokens")
            
            # 确保所有输入都在正确的设备上
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}")
            
            if model_device.type == 'cuda':
                input_ids = inputs["input_ids"].to(model_device)
                attention_mask = inputs["attention_mask"].to(model_device)
                print(f"Input tensors moved to: {input_ids.device}")
            else:
                input_ids = inputs["input_ids"].cpu()
                attention_mask = inputs["attention_mask"].cpu()
                print(f"Input tensors moved to: {input_ids.device}")
            
            # 根据配置决定是否使用句子完整性检测
            enable_completion = getattr(self.config.generator, 'enable_sentence_completion', True)
            
            if enable_completion:
                # Generate with sentence completion check
                response = self._generate_with_completion_check(input_ids, attention_mask)
            else:
                # 使用原始生成方法
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.3,
                        length_penalty=0.8,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 清理答案，移除可能的prompt注入
            cleaned_response = self._clean_response(response)
            responses.append(cleaned_response.strip())
            
        return responses
    
    def _clean_response(self, response: str) -> str:
        """清理LLM生成的答案，移除可能的prompt注入和格式标记"""
        
        # 移除常见的prompt注入标记和格式
        injection_patterns = [
            # 中文标记
            r'【回答】.*',  # 移除"【回答】"及其后面的内容
            r'回答：.*',   # 移除"回答："及其后面的内容
            r'回答\s*[:：].*',  # 移除"回答:"及其后面的内容
            
            # 英文标记
            r'Answer:.*',  # 移除"Answer:"及其后面的内容
            r'Answer\s*[:：].*',  # 移除"Answer:"及其后面的内容
            
            # 分隔线和格式标记
            r'---.*',      # 移除分隔线
            r'===.*',      # 移除分隔线
            r'___.*',      # 移除下划线分隔线
            r'\*\*\*.*',   # 移除星号分隔线
            
            # 格式标记
            r'boxed\{.*?\}',  # 移除boxed格式
            r'\\boxed\{.*?\}',  # 移除LaTeX boxed格式
            r'\\text\{.*?\}',  # 移除LaTeX text格式
            
            # 其他常见注入
            r'根据.*?信息.*?无法.*?提供.*?信息.*',  # 移除重复的"无法提供信息"表述
            r'根据现有信息.*?无法提供此项信息.*',  # 移除重复的"无法提供此项信息"表述
        ]
        
        cleaned = response
        for pattern in injection_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除多余的空白字符和换行
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 移除开头和结尾的标点符号
        cleaned = re.sub(r'^[，。！？、；：,.!?;:]+', '', cleaned)
        cleaned = re.sub(r'[，。！？、；：,.!?;:]+$', '', cleaned)
        
        # 移除重复的句子
        sentences = cleaned.split('。')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        cleaned = '。'.join(unique_sentences)
        
        # 如果清理后为空，返回原始响应的前100个字符
        if not cleaned.strip():
            return response[:100].strip()
        
        return cleaned
    
    def _is_sentence_complete(self, text: str) -> bool:
        """检测句子是否完整"""
        if not text.strip():
            return True
        
        # 中文句子完整性检测
        chinese_endings = ['。', '！', '？', '；', '：', '…', '...']
        # 英文句子完整性检测
        english_endings = ['.', '!', '?', ';', ':', '...']
        
        # 检查是否以完整句子结尾
        text = text.strip()
        for ending in chinese_endings + english_endings:
            if text.endswith(ending):
                return True
        
        # 检查是否包含完整的句子结构（有主语和谓语）
        # 简单的中文句子结构检测
        if '。' in text:
            sentences = text.split('。')
            if sentences and sentences[-1].strip():  # 最后一句不为空
                return False  # 最后一句不完整
            return True
        
        # 英文句子结构检测
        if '.' in text:
            sentences = text.split('.')
            if sentences and sentences[-1].strip():  # 最后一句不为空
                return False  # 最后一句不完整
            return True
        
        return False
    
    def _generate_with_completion_check(self, input_ids, attention_mask):
        """带完整性检查的生成，如果句子不完整则重试"""
        
        # 从配置获取参数
        max_attempts = getattr(self.config.generator, 'max_completion_attempts', 3)
        token_increment = getattr(self.config.generator, 'token_increment', 50)
        max_total_tokens = getattr(self.config.generator, 'max_total_tokens', 400)
        
        for attempt in range(max_attempts):
            # 计算当前尝试的token数量
            current_max_tokens = min(
                self.max_new_tokens + (attempt * token_increment),
                max_total_tokens
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=current_max_tokens,
                    do_sample=True,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    length_penalty=0.8,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 检查句子完整性
            if self._is_sentence_complete(response):
                return response
            
            print(f"⚠️  第{attempt+1}次生成句子不完整，增加token数量重试...")
        
        # 如果所有尝试都失败，返回最后一次的结果
        print("⚠️  达到最大重试次数，返回当前结果")
        return response