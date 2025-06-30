import os
import json
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
        config = Config()
        
        # 如果没有提供model_name，从config读取
        if model_name is None:
            model_name = config.generator.model_name
        
        # 如果没有提供device，从config读取
        if device is None:
            device = config.generator.device
        
        # 如果没有提供量化参数，从config读取
        if use_quantization is None:
            use_quantization = config.generator.use_quantization
        if quantization_type is None:
            quantization_type = config.generator.quantization_type
        
        super().__init__(model_name=model_name)
        self.device = device
        self.temperature = config.generator.temperature
        self.max_new_tokens = config.generator.max_new_tokens
        self.top_p = config.generator.top_p
        
        # 使用config中的平台感知配置
        if cache_dir is None:
            cache_dir = config.generator.cache_dir  # 使用generator的缓存目录
        
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
            
            # Generate with increased length
            with torch.no_grad():  # 添加no_grad来节省内存
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,  # 使用配置文件中的值
                    do_sample=True,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,  # Use eos_token_id for padding
                    repetition_penalty=1.1,  # 添加重复惩罚
                    length_penalty=1.0,  # 添加长度惩罚
                    eos_token_id=self.tokenizer.eos_token_id  # 明确指定结束token
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response.strip())
            
        return responses 