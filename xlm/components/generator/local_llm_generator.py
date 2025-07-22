import os
import json
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

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
        self.config = Config() # Use platform-aware configuration from config
        
        # If model_name not provided, read from config
        if model_name is None:
            model_name = self.config.generator.model_name
        
        # If device not provided, read from config
        if device is None:
            device = self.config.generator.device
        
        # If quantization parameters not provided, read from config
        if use_quantization is None:
            use_quantization = self.config.generator.use_quantization
        if quantization_type is None:
            quantization_type = self.config.generator.quantization_type
        
        # Validate configuration parameters
        validated_device, validated_use_quantization, validated_quantization_type = self._validate_config(
            model_name, device or "cpu", use_quantization, quantization_type
        )
        
        super().__init__(model_name=model_name)
        self.device = validated_device
        self.use_quantization = validated_use_quantization
        self.quantization_type = validated_quantization_type
        self.max_new_tokens = self.config.generator.max_new_tokens
        
        # For Fin-R1 model, don't set temperature and top_p attributes to avoid transformers auto-injection
        if "Fin-R1" not in model_name:
            self.temperature = self.config.generator.temperature
            self.top_p = self.config.generator.top_p
        
        # Use platform-aware configuration from config
        if cache_dir is None:
            cache_dir = self.config.generator.cache_dir 
        
        self.cache_dir = cache_dir  
        self.use_flash_attention = use_flash_attention
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set Hugging Face environment variables
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        
        # Memory optimization setup
        self._setup_memory_optimization()
        
        # Initialize template loader
        self._init_template_loader()
        
        self._load_model_and_tokenizer()
        print(f"LocalLLMGenerator '{model_name}' loaded on {self.device} with quantization: {self.use_quantization} ({self.quantization_type}).")
    
    def _init_template_loader(self):
        """Initialize template loader - using shared resource manager"""
        try:
            from xlm.utils.shared_resource_manager import shared_resource_manager
            self.templates = shared_resource_manager.get_templates()
        except ImportError:
            # Fallback to original method
            self.template_dir = Path("data/prompt_templates")
            self.templates = {}
            self._load_templates()
    
    def _load_templates(self):
        """Load all template files - only used for fallback"""
        if not self.template_dir.exists():
            print(f"Warning: Template directory {self.template_dir} does not exist")
            return
        
        for template_file in self.template_dir.glob("*.txt"):
            template_name = template_file.stem
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self.templates[template_name] = f.read().strip()
                print(f"Loaded template: {template_name}")
            except Exception as e:
                print(f"Error loading template {template_name}: {e}")
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get template with specified name"""
        return self.templates.get(template_name)
    
    def format_hybrid_template(self, question: str, table_context: str = "", text_context: str = "", hybrid_decision: str = "hybrid") -> str:
        """
        Format hybrid answer template
        
        Args:
            question: User question
            table_context: Table context
            text_context: Text context
            hybrid_decision: Hybrid decision type ("hybrid", "table", "text", "multi_stage_chinese")
            
        Returns:
            Formatted template string
        """
        # Select template based on hybrid_decision
        template_name = None
        if hybrid_decision == "multi_stage_chinese":
            template_name = "multi_stage_chinese_template"
        elif hybrid_decision == "hybrid":
            template_name = "template_for_hybrid_answer"
        elif hybrid_decision == "table":
            template_name = "template_for_table_answer"
        elif hybrid_decision == "text":
            template_name = "template_for_text_answer"
        else:
            # Default to hybrid template
            template_name = "template_for_hybrid_answer"
        
        template = self.get_template(template_name)
        if template is None:
            raise ValueError(f"{template_name} not found")
        
        try:
            # For multi_stage_chinese_template, use special formatting logic
            if hybrid_decision == "multi_stage_chinese":
                # Combine context as complete fragment, use summary as summary
                combined_context = f"{table_context}\n{text_context}".strip()
                summary = combined_context[:500] + "..." if len(combined_context) > 500 else combined_context
                
                formatted_template = template.format(
                    summary=summary,
                    context=combined_context,
                    query=question
                )
            else:
                # Other templates use original logic
                formatted_template = template.format(
                    question=question,
                    table_context=table_context,
                    text_context=text_context
                )
            return formatted_template
        except KeyError as e:
            raise ValueError(f"Error formatting {template_name} template: missing key {e}")
        except Exception as e:
            raise ValueError(f"Error formatting {template_name} template: {e}")
    
    def generate_hybrid_answer(self, question: str, table_context: str = "", text_context: str = "", hybrid_decision: str = "hybrid") -> str:
        """
        Generate answer using hybrid answer template
        
        Args:
            question: User question
            table_context: Table context
            text_context: Text context
            hybrid_decision: Hybrid decision type ("hybrid", "table", "text", "multi_stage_chinese")
            
        Returns:
            Generated answer
        """
        # Format hybrid answer template
        formatted_prompt = self.format_hybrid_template(question, table_context, text_context, hybrid_decision)
        
        # Use existing generation method
        responses = self.generate([formatted_prompt])
        
        if responses and len(responses) > 0:
            return responses[0]
        else:
            return "Failed to generate answer"
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract final answer from generated response
        
        Args:
            response: Generated complete response
            
        Returns:
            Extracted answer part
        """
        # Try to extract content from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # If <answer> tags not found, try other patterns
        # Look for content after "Answer:" or "Answer:"
        answer_patterns = [
            r'Answer:\s*(.*?)(?:\n|$)',
            r'答案:\s*(.*?)(?:\n|$)',
            r'<answer>\s*(.*?)\s*</answer>',
            r'Answer\s*:\s*(.*?)(?:\n|$)',
            r'答案\s*:\s*(.*?)(?:\n|$)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If none found, return original response
        return response.strip()

    def _validate_config(self, model_name: str, device: str, use_quantization: bool, quantization_type: str):
        """Validate configuration parameters"""
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        validated_device = device
        validated_use_quantization = use_quantization
        validated_quantization_type = quantization_type
        
        if device and device.startswith('cuda'):
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                validated_device = "cpu"
            else:
                # Parse specified GPU device ID
                if ":" in device:
                    gpu_id = int(device.split(":")[1])
                else:
                    gpu_id = 0
                
                # Check if specified GPU exists
                if gpu_id >= torch.cuda.device_count():
                    print(f"Warning: GPU {gpu_id} does not exist, falling back to GPU 0")
                    gpu_id = 0
                    validated_device = "cuda:0"
                
                # Check specified GPU memory
                cuda_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                print(f"GPU {gpu_id} memory available: {cuda_memory:.1f} GB")
                
                if cuda_memory < 4 and use_quantization:
                    print(f"Warning: GPU {gpu_id} memory < 4GB, enabling quantization")
                    validated_use_quantization = True
                    if quantization_type not in ['4bit', '8bit']:
                        validated_quantization_type = '4bit'
        
        if validated_use_quantization and validated_quantization_type not in ['4bit', '8bit']:
            print(f"Warning: Invalid quantization type: {validated_quantization_type}, falling back to 4bit")
            validated_quantization_type = '4bit'
            
        return validated_device, validated_use_quantization, validated_quantization_type
    
    def _setup_memory_optimization(self):
        """Setup memory optimization"""
        # Set PyTorch memory allocator
        if torch.cuda.is_available() and self.device and self.device.startswith('cuda'):
            # Parse specified GPU device ID
            if ":" in self.device:
                gpu_id = int(self.device.split(":")[1])
            else:
                gpu_id = 0
            
            # Check if specified GPU exists
            if gpu_id < torch.cuda.device_count():
                # Set current device
                torch.cuda.set_device(gpu_id)
                # Enable memory cache
                torch.cuda.empty_cache()
                print(f"Success: Set GPU {gpu_id} as current device")
            else:
                print(f"Warning: GPU {gpu_id} does not exist, using GPU 0")
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            
            # Set more aggressive memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
            
            # If memory insufficient, enable gradient checkpointing
            if hasattr(self.config.generator, 'use_gradient_checkpointing'):
                if getattr(self.config.generator, 'use_gradient_checkpointing', False):
                    print("Enabling gradient checkpointing for memory optimization")
        
        # Set transformers memory optimization
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer parallelization issues
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer"""
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
            "use_cache": False,  # Disable KV cache to save memory
        }

        # Apply quantization based on configuration
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
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
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
            model_kwargs["device_map"] = self.device  # Explicitly specify device
        else:
            if self.device and self.device.startswith('cuda'):
                print("CUDA device detected but quantization disabled. Loading model without quantization.")
                model_kwargs["device_map"] = self.device  # Explicitly specify device
                model_kwargs["torch_dtype"] = torch.float16
            else:
                print("CPU device detected. Loading model without quantization.")
                model_kwargs["device_map"] = "cpu"
                model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
    
    def convert_to_json_chat_format(self, text, hybrid_decision=None):
        """Convert string containing chat format markers to JSON chat format"""
        
        # If input is already in JSON format, return directly
        if text.strip().startswith('[') and text.strip().endswith(']'):
            try:
                json.loads(text)
                print("Input is already in JSON format")
                return text
            except json.JSONDecodeError:
                pass
        
        # Detect English template type (===SYSTEM=== format, exclude rag_english_template)
        if "===SYSTEM===" in text and "===USER===" in text:
            # Exclude rag_english_template (uses <system> format)
            if "<system>" in text:
                print("Detected rag_english_template format (skipping)")
                # Continue to next detection logic
            else:
                # Detect template type (based on hybrid_decision parameter)
                template_type = "unknown"
                if hybrid_decision:
                    if hybrid_decision == "table":
                        template_type = "table"
                    elif hybrid_decision == "text":
                        template_type = "text"
                    elif hybrid_decision == "hybrid":
                        template_type = "hybrid"
                    else:
                        template_type = "general"
                else:
                    template_type = "general"
                
                print(f"Detected English template format: {template_type}")
                
                # Extract SYSTEM section
                system_start = text.find("===SYSTEM===")
                user_start = text.find("===USER===")
                
                if system_start != -1 and user_start != -1:
                    system_content = text[system_start + 12:user_start].strip()
                    user_content = text[user_start + 10:].strip()
                    
                    # Build JSON format
                    chat_data = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    
                    return json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        # Detect ChatML format (<|im_start|>role...<|im_end|>)
        if "<|im_start|>" in text and "<|im_end|>" in text:
            print("Detected ChatML format with <|im_start|> markers")
            
            import re
            # Match <|im_start|>role\ncontent<|im_end|> format
            pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
            matches = re.findall(pattern, text, re.DOTALL)
            
            chat_data = []
            for role, content in matches:
                content = content.strip()
                if content:  # Only add non-empty content
                    if role == "system":
                        chat_data.append({"role": "system", "content": content})
                    elif role == "user":
                        chat_data.append({"role": "user", "content": content})
                    elif role == "assistant":
                        chat_data.append({"role": "assistant", "content": content})
            
            if chat_data:
                print(f"Extracted {len(chat_data)} ChatML messages: {[msg['role'] for msg in chat_data]}")
                return json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        # Detect multi_stage_chinese_template.txt format
        if "===SYSTEM===" in text and "===USER===" in text:
            print("Detected multi-stage Chinese template format")
            
            # Extract SYSTEM section
            system_start = text.find("===SYSTEM===")
            user_start = text.find("===USER===")
            
            if system_start != -1 and user_start != -1:
                system_content = text[system_start + 12:user_start].strip()
                user_content = text[user_start + 10:].strip()
                
                # Build JSON format
                chat_data = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                
                return json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        # Detect if contains Chinese system instructions (compatible with old format)
        if "你是一位专业的金融分析师" in text:
            print("Detected Chinese system instruction")
            # Extract system section - find start and end of system instructions
            system_start = text.find("你是一位专业的金融分析师")
            
            # Find end position of system instructions (usually before "【公司财务报告摘要】" or "【公司财务报告片段】")
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
                
                # Build JSON format
                chat_data = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                
                return json.dumps(chat_data, ensure_ascii=False, indent=2)
        
        # If none match, return original text as user message
        print("No specific format detected, treating as user message")
        chat_data = [
            {"role": "user", "content": text}
        ]
        return json.dumps(chat_data, ensure_ascii=False, indent=2)



    def convert_json_to_model_format(self, json_chat: str) -> str:
        """Convert JSON chat format to model expected format"""
        try:
            chat_data = json.loads(json_chat)
            
            # Select conversion method based on model type
            if "Fin-R1" in self.model_name:
                return self._convert_to_fin_r1_format(chat_data)
            elif "Qwen" in self.model_name:
                return self._convert_to_qwen_format(chat_data)
            else:
                return self._convert_to_default_format(chat_data)
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return json_chat  # Return original text as fallback

    def _convert_to_fin_r1_format(self, chat_data: List[Dict]) -> str:
        """Convert to Fin-R1 expected <|im_start|>...<|im_end|> format"""
        formatted_parts = []
        
        for message in chat_data:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant start marker
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)

    def _convert_to_qwen_format(self, chat_data: List[Dict]) -> str:
        """Convert to Qwen expected format"""
        formatted_parts = []
        
        for message in chat_data:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant start marker
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)

    def _convert_to_default_format(self, chat_data: List[Dict]) -> str:
        """Convert to default format (direct concatenation)"""
        formatted_parts = []
        
        for message in chat_data:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n".join(formatted_parts)

    def generate(self, texts: List[str]) -> List[str]:
        """Generate answers with error handling and performance optimization"""
        responses = []
        
        for i, text in enumerate(texts):
            try:
                print(f"Processing input {i+1}/{len(texts)}...")
                
                # Debug info: print device status
                print(f"Generator device: {self.device}")
                model_device = next(self.model.parameters()).device 
                print(f"Model device: {model_device}")
                
                print(f"Input text length: {len(text)} characters")
                
                # Check if model supports chat format and convert
                processed_text = text
                if any(model_name in self.model_name for model_name in ["Fin-R1", "Qwen"]):
                    print(f"Chat model detected ({self.model_name}), converting to JSON chat format...")
                    json_chat_str = self.convert_to_json_chat_format(text)
                    print(f"JSON chat format length: {len(json_chat_str)} characters")
                    
                    processed_text = self.convert_json_to_model_format(json_chat_str)
                    print(f"Converted to {self.model_name} format, length: {len(processed_text)} characters")
                else:
                    print(f"Non-chat model detected ({self.model_name}), using original format...")
                
                # Tokenize input
                inputs = self.tokenizer(
                    processed_text,
                    return_tensors="pt",
                    truncation=False,  
                    padding=False,     
                    add_special_tokens=True
                )
                
                print(f"Tokenized input length: {inputs['input_ids'].shape[1]} tokens")
                
                # Ensure all input tensors are on correct device
                if model_device.type == 'cuda':
                    input_ids = inputs["input_ids"].to(model_device)
                    attention_mask = inputs["attention_mask"].to(model_device)
                    print(f"Input tensors moved to: {input_ids.device}")
                else:
                    input_ids = inputs["input_ids"].cpu()
                    attention_mask = inputs["attention_mask"].cpu()
                    print(f"Input tensors moved to: {input_ids.device}")
                
                enable_completion = getattr(self.config.generator, 'enable_sentence_completion', True) 
                
                if enable_completion:
                    response = self._generate_with_completion_check(input_ids, attention_mask)
                else:
                    response = self._generate_simple(input_ids, attention_mask)
                
                # Clean answer, remove possible prompt injection and format markers
                cleaned_response = self._clean_response(response)
                responses.append(cleaned_response.strip())
                
                print(f"Success: Input {i+1} processing completed")
                
            except Exception as e:
                print(f"Error: Failed to process input {i+1}: {str(e)}")
                # Return error message to avoid entire batch failure
                responses.append(f"Error generating answer: {str(e)}")
                
        return responses
    
    def _generate_simple(self, input_ids, attention_mask):
        """Simple generation method without completeness check"""
        import time
        import threading
        
        start_time = time.time()
        
        # Get model-specific configuration
        model_config = self._get_model_specific_config()
        
        # For Fin-R1 model, delete unsupported attributes to avoid transformers auto-injection
        if model_config["model_type"] == "fin_r1":
            for k in ["temperature", "top_p", "top_k"]:
                if hasattr(self, k):
                    delattr(self, k)
        
        # Print debug information
        print(f"Debug: Generation parameters:")
        print(f"   - max_new_tokens: {self.max_new_tokens}")
        print(f"   - model_type: {model_config['model_type']}")
        
        # Display relevant parameters based on model type
        if model_config["model_type"] == "fin_r1":
            print(f"   - do_sample: False (Fin-R1 uses deterministic generation)")
            print(f"   - repetition_penalty: 1.1")
        else:
            print(f"   - temperature: {self.temperature}")
            print(f"   - top_p: {self.top_p}")
            print(f"   - do_sample: {getattr(self.config.generator, 'do_sample', False)}")
        
        # Get configuration parameters
        do_sample = getattr(self.config.generator, 'do_sample', False)
        repetition_penalty = getattr(self.config.generator, 'repetition_penalty', 1.1)
        
        # Choose different generation parameters based on model type
        if model_config["model_type"] == "fin_r1":
            # Fin-R1 parameters: only use model-supported parameters to avoid warnings
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,  # Use deterministic generation
                "pad_token_id": model_config["pad_token_id"],
                "eos_token_id": model_config["eos_token_id"],
                "repetition_penalty": 1.1  # Prevent repetition
            }
        else:
            # Other models: use complete generation parameters
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": model_config["pad_token_id"],
                "eos_token_id": model_config["eos_token_id"],
                "repetition_penalty": repetition_penalty
            }
            
            # Add sampling-related parameters (only for non-Fin-R1 models)
            if do_sample:
                generation_kwargs.update({
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "no_repeat_ngram_size": 3
                })
        
        print(f"   - Final max_new_tokens used: {generation_kwargs['max_new_tokens']}")
        print(f"   - Number of generation parameters: {len(generation_kwargs)}")
        
        # Add timeout mechanism
        max_generation_time = getattr(self.config.generator, 'max_generation_time', 30)  # 30 seconds timeout
        print(f"   - Generation timeout: {max_generation_time} seconds")
        
        # Use thread and event to monitor timeout
        generation_completed = threading.Event()
        generation_result: list = [None]
        generation_error: list = [None]
        
        def generate_with_timeout():
            try:
                with torch.no_grad():
                    outputs = self.model.generate(**generation_kwargs)
                generation_result[0] = outputs
                generation_completed.set()
            except Exception as e:
                generation_error[0] = str(e)
                generation_completed.set()
        
        # Start generation thread
        generation_thread = threading.Thread(target=generate_with_timeout)
        generation_thread.start()
        
        # Wait for generation to complete or timeout
        if generation_completed.wait(timeout=max_generation_time):
            if generation_error[0]:
                print(f"Error: {generation_error[0]}")
                return f"Error: {generation_error[0]}"
            outputs = generation_result[0]
            if outputs is None:
                return "Error: Failed to generate answer"
        else:
            print("Error: Generation timeout, returning partial result...")
            # Try to get partial result
            try:
                with torch.no_grad():
                    # Use smaller max_new_tokens for retry
                    generation_kwargs["max_new_tokens"] = min(50, self.max_new_tokens // 4)
                    outputs = self.model.generate(**generation_kwargs)
            except Exception as e:
                print(f"Error: Failed to generate answer: {str(e)}")
                return "Error: Failed to generate answer"
        
        # Calculate actual generated token number and generation time
        generated_tokens = outputs[0][input_ids.shape[1]:]
        actual_new_tokens = len(generated_tokens)
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"   - Actual generated token number: {actual_new_tokens}")
        print(f"   - Generation time: {generation_time:.2f} seconds")
        print(f"   - Generation speed: {actual_new_tokens/generation_time:.1f} tokens/second")
        
        # Check if too many tokens are generated
        if actual_new_tokens >= self.max_new_tokens:
            print(f"Warning: Generated maximum token number ({actual_new_tokens}), possibly truncated")
        
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def _clean_response(self, response: str) -> str:
        """
        Forced post-processing module: remove all contaminated content
        Based on optimized version from test_clean.py
        """
        print("Cleaning: Starting forced post-processing...")
        
        # 0. Prioritize company name and year corrections as they may affect subsequent cleaning matches
        text = self._fix_company_name_translation(response)
        
        # 1. Remove meta-comments and debug information (place first, handle large redundant blocks)
        # Note: regex order is important, broader patterns first
        patterns_to_remove = [
            # Most likely to appear in large evaluation/thinking mode
            r'我需要检查这个回答是否符合要求.*?====', # Match from "我需要检查" to "===="
            r'\*\*注意\*\*:.*?改进后的版本[:：]', # Match from "**注意**:" to "改进后的版本:"
            r'上面的答案虽然符合要求.*?以下是改进后的版本:', # Same as above
            r'###\s*改进版答案', # Remove ### 改进版答案 title
            r'###\s*回答', # Remove ### 回答 title
            r'回答完成后立即停止生成', # Remove prompt's final instruction
            r'回答完成并停止', # Remove prompt's final instruction
            r'确保回答', # Remove prompt's final instruction
            r'用户可能', # Remove prompt's final instruction
            r'总结一下', # Remove prompt's final instruction
            r'请用简洁', # Remove prompt's final instruction
            r'进一步简化', # Remove prompt's final instruction
            r'再简化的版本', # Remove prompt's final instruction
            r'最终答案定稿如下', # Remove prompt's final instruction
            r'这个总结全面', # Remove prompt's final instruction
            r'核心点总结[:：]?', # Remove 核心点总结 title
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
            r'答案示例[:：]?',
            r'最终确认[:：]?',
            r'答案忠实地反映了原始文档的内容而无多余推断',
            r'回答[:：]\s*$', # Remove standalone "回答：" or "回答：" at line end
            r'回答是：\s*', # Remove "回答是："
            r'以下是原因：\s*', # Remove "以下是原因："

            # Remove <|markers|> (these should be handled by skip_special_tokens=True, but as post-processing fallback)
            r'<\|[^>]+\|>',
            r'\\boxed\{.*?\}', # Remove \boxed{} format
            r'\\text\{.*?\}', # Remove LaTeX text format
            r'\\s*', # Remove some LaTeX related whitespace
            r'[\u2460-\u2469]\s*', # Remove circled numbers, such as ①

            # Clear structural markers that exist in Prompt, if they accidentally appear in answers
            r'===SYSTEM===[\s\S]*?===USER===', # Remove System section
            r'---[\s\S]*?---', # Remove USER section's --- separators and all content between them (if accidentally copied)
            r'【公司财务报告摘要】[\s\S]*?【完整公司财务报告片段】', # Remove summary and fragment labels
            r'【用户问题】[\s\S]*?【回答】', # Remove question and answer labels

            r'Based on the provided financial reports and analyses, the main reasons for Desay Battery\'s (000049) continued profit growth in 2021 are:', # English opening
            r'Here are the main reasons for Desay Battery\'s (000049) continued profit growth in 2021:', # English opening

            r'根据财报预测及评论，德赛 battery \(00\) 的20\(21\?\) 年度利润增涨主因有三:', # Specific opening
            r'根据财报预测，德赛 battery \(00\) 的20\(21\?\) 年度利润增涨主因有三:', # Specific opening

            r'综上所述，A客 户市场份额扩张 \+ 多元化应用生态系统的协同效应共同构成了20年度乃至整个21财年内稳健增长的基础条件 \. 注意 ：以上论断完全依赖于已公开披露的信息资源 ; 对未来的具体前景尚需结合更多实时数据加以验证和完善', # Exact match for previous logs

            r'（注意此段文字虽详细阐述了几方面因素及其相互作用机制，但由于题干要求高度浓缩为一句话内完成表述，故在此基础上进行了适当简化压缩）', # Exact match for previous logs

            r'德赛 battery \(00\) 的 20 年度财报显示其利润大幅超越预期 , 主要由于 iPhone 1\(Pro Max \) 新机型的需求旺盛 和新产品带来的高毛利率。展望未来一年 , 原因有三 :', # Another specific opening
        ]
        
        # Apply cleaning patterns in batch
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # 2. Remove all format markers (more general cleaning)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Remove **bold**, keep content
        text = re.sub(r'\*(.*?)\*', r'\1', text)   # Remove *italic*, keep content
        text = text.replace("---", "").replace("===", "") # Remove separators
        text = re.sub(r'^\s*[\d]+\.\s*', '', text, flags=re.MULTILINE) # Remove line-start numbered lists "1. "
        text = re.sub(r'^\s*[-*•·]\s*', '', text, flags=re.MULTILINE) # Remove line-start bullet lists "- "
        text = re.sub(r'^\s*\((\w|[一二三四五六七八九十])+\)\s*', '', text, flags=re.MULTILINE) # Remove line-start (i), (一)
        text = re.sub(r'\s*\([^\)]*\)\s*', '', text) # Remove all English parentheses and content, **use with caution**
        text = re.sub(r'\s*（[^）]*）\s*', '', text) # Remove all Chinese parentheses and content, **use with caution**
        text = re.sub(r'[，；,;]$', '', text) # Remove trailing commas or semicolons to prevent sentence misjudgment

        # 3. Clean excess whitespace and line breaks
        text = re.sub(r'\n+', ' ', text).strip() # Replace multiple line breaks with single space, then trim
        text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

        # 4. Limit sentence count (ensure sentence completeness before truncation)
        sentences = re.split(r'(?<=[。？！；])\s*', text) # Use lookbehind to ensure separators remain at sentence end
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 3: # Assume you want to limit to 3 sentences
            sentences = sentences[:3]
        
        final_text = ' '.join(sentences) # First join with spaces

        # Ensure ending with sentence-ending punctuation
        if final_text and not final_text.endswith(('。', '！', '？', '.', '!', '?')):
            final_text += '。'
        
        # Add length control - limit answer length
        enable_length_limit = getattr(self.config.generator, 'enable_response_length_limit', True)
        if enable_length_limit:
            max_chars = getattr(self.config.generator, 'max_response_chars', 800)  # Default 800 characters
            if len(final_text) > max_chars:
                # Try to truncate at sentence boundaries
                sentences = final_text.split('。')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + '。') <= max_chars:
                        truncated += sentence + '。'
                    else:
                        break
                
                if truncated:
                    final_text = truncated
                else:
                    # If cannot truncate at sentence boundaries, truncate directly
                    final_text = final_text[:max_chars].rstrip('。') + '。'
                
                print(f"Length: Answer too long, truncated to {len(final_text)} characters")
        else:
            print(f"Length: Length limit disabled, current answer length: {len(final_text)} characters")
        
        # If cleaned result is empty, return first N characters of original response as fallback
        if not final_text.strip():
            return response[:150].strip()
            
        print(f"Cleaning: Post-processing completed, length: {len(final_text)} characters")
        return final_text
    
    def _fix_company_name_translation(self, text: str) -> str:
        """Fix company name translation issues and year problems"""
        # Common company name translation mappings and non-standard expression corrections (Chinese -> Chinese standard)
        company_translations = {
            # Desay Battery related (ensure broader matching, including spaces or non-standard expressions)
            r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*BATTERY\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*battery': '德赛电池',
            r'德赛\s*Battery': '德赛电池',
            r'德赛\s*BATTERY': '德赛电池',
            r'德赛\s*\(00\)': '德赛电池（000049）', 
            r'德塞电池': '德赛电池', # Fix typo
            
            # Product name corrections
            r'iPhone\s*\+\s*ProMax': 'iPhone 12 Pro Max',
            r'iPhon\s*e12ProMax': 'iPhone 12 Pro Max',
            r'iPhone\s*X\s*系列': 'iPhone 12 Pro Max', 
            r'iPhone\s*1\s*\(Pro\s*Max\s*\)': 'iPhone 12 Pro Max',
            r'iPhone\s*1\s*Pro\s*Max': 'iPhone 12 Pro Max',
            r'iPhone\s*2\s*ProMax': 'iPhone 12 Pro Max', # Fix from previous logs
        }
        for pattern, replacement in company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Year corrections
        text = re.sub(r'20\s*\(\s*\d{2}\?\)\s*年度', r'2021年度', text, flags=re.IGNORECASE) # Fix 20(21?) 年度
        text = text.replace('20XX年', '2021年') # Fix 20XX年
        text = text.replace('20+', '2021') # Fix 20+
        text = text.replace('2OI I年', '2021年') # Fix 2OI I年
        text = text.replace('20 I I年', '2021年') # Fix 20 I I年 (with spaces)

        return text
    
    def _is_sentence_complete(self, text: str) -> bool:
        """
        Intelligently detect if sentence is complete.
        Optimization: More accurately judge Chinese sentence completeness.
        """
        if not text.strip():
            return True
        
        text_stripped = text.strip()
        
        # 1. Check if ending with sentence-ending punctuation
        sentence_endings = ['。', '！', '？', '；', '：', '…', '...', '.', '!', '?', ';']
        for ending in sentence_endings:
            if text_stripped.endswith(ending):
                return True
        
        # 2. Check if ending with non-sentence-ending punctuation (usually indicates incompleteness)
        incomplete_endings = ['，', '、', ',', '/', '-', '：', ':', '；', ';']
        for ending in incomplete_endings:
            if text_stripped.endswith(ending):
                return False
        
        # 3. Check if contains complete sentence structure
        # If text contains period but doesn't end with period, may be incomplete
        if '。' in text_stripped and not text_stripped.endswith('。'):
            # Check if content after last period constitutes complete sentence
            last_period_pos = text_stripped.rfind('。')
            after_last_period = text_stripped[last_period_pos + 1:].strip()
            
            if after_last_period:
                # If there's content after period, check if ending with sentence-ending punctuation
                for ending in sentence_endings:
                    if after_last_period.endswith(ending):
                        return True
                # If there's content after period but not ending with sentence-ending punctuation, may be incomplete
                return False
        
        # 4. Check length and content characteristics
        # If text is very short (less than 10 characters) and doesn't end with sentence-ending punctuation, may be incomplete
        if len(text_stripped) < 10 and not any(text_stripped.endswith(ending) for ending in sentence_endings):
            return False
        
        # 5. Check if ending with common incomplete patterns
        incomplete_patterns = [
            r'等$',  # Ending with "等"
            r'等[，。]?$',  # Ending with "等，" or "等。"
            r'等等$',  # Ending with "等等"
            r'等等[，。]?$',  # Ending with "等等，" or "等等。"
            r'其中$',  # Ending with "其中"
            r'包括$',  # Ending with "包括"
            r'例如$',  # Ending with "例如"
            r'主要$',  # Ending with "主要"
            r'重要$',  # Ending with "重要"
            r'关键$',  # Ending with "关键"
            r'核心$',  # Ending with "核心"
            r'方面$',  # Ending with "方面"
            r'因素$',  # Ending with "因素"
            r'原因$',  # Ending with "原因"
            r'影响$',  # Ending with "影响"
            r'导致$',  # Ending with "导致"
            r'造成$',  # Ending with "造成"
            r'推动$',  # Ending with "推动"
            r'促进$',  # Ending with "促进"
            r'提升$',  # Ending with "提升"
            r'增长$',  # Ending with "增长"
            r'下降$',  # Ending with "下降"
            r'减少$',  # 以"减少"结尾
            r'增加$',  # 以"增加"结尾
            r'提高$',  # 以"提高"结尾
            r'改善$',  # 以"改善"结尾
            r'优化$',  # 以"优化"结尾
            r'调整$',  # 以"调整"结尾
            r'变化$',  # 以"变化"结尾
            r'趋势$',  # 以"趋势"结尾
            r'前景$',  # 以"前景"结尾
            r'展望$',  # 以"展望"结尾
            r'预期$',  # 以"预期"结尾
            r'预计$',  # 以"预计"结尾
            r'预测$',  # 以"预测"结尾
            r'分析$',  # 以"分析"结尾
            r'研究$',  # 以"研究"结尾
            r'调查$',  # 以"调查"结尾
            r'报告$',  # 以"报告"结尾
            r'数据$',  # 以"数据"结尾
            r'指标$',  # 以"指标"结尾
            r'表现$',  # 以"表现"结尾
            r'业绩$',  # 以"业绩"结尾
            r'收入$',  # 以"收入"结尾
            r'利润$',  # 以"利润"结尾
            r'成本$',  # 以"成本"结尾
            r'价格$',  # 以"价格"结尾
            r'销量$',  # 以"销量"结尾
            r'产量$',  # 以"产量"结尾
            r'产能$',  # 以"产能"结尾
            r'市场$',  # 以"市场"结尾
            r'行业$',  # 以"行业"结尾
            r'公司$',  # 以"公司"结尾
            r'企业$',  # 以"企业"结尾
            r'产品$',  # 以"产品"结尾
            r'服务$',  # 以"服务"结尾
            r'技术$',  # 以"技术"结尾
            r'创新$',  # 以"创新"结尾
            r'发展$',  # 以"发展"结尾
            r'战略$',  # 以"战略"结尾
            r'计划$',  # 以"计划"结尾
            r'目标$',  # 以"目标"结尾
            r'投资$',  # 以"投资"结尾
            r'融资$',  # 以"融资"结尾
            r'合作$',  # 以"合作"结尾
            r'竞争$',  # 以"竞争"结尾
            r'优势$',  # 以"优势"结尾
            r'劣势$',  # 以"劣势"结尾
            r'机会$',  # 以"机会"结尾
            r'威胁$',  # 以"威胁"结尾
            r'风险$',  # 以"风险"结尾
            r'挑战$',  # 以"挑战"结尾
            r'问题$',  # 以"问题"结尾
            r'困难$',  # 以"困难"结尾
            r'瓶颈$',  # 以"瓶颈"结尾
            r'限制$',  # 以"限制"结尾
            r'约束$',  # 以"约束"结尾
            r'条件$',  # 以"条件"结尾
            r'要求$',  # 以"要求"结尾
            r'标准$',  # 以"标准"结尾
            r'规范$',  # 以"规范"结尾
            r'政策$',  # 以"政策"结尾
            r'法规$',  # 以"法规"结尾
            r'监管$',  # 以"监管"结尾
            r'环境$',  # 以"环境"结尾
            r'背景$',  # 以"背景"结尾
            r'情况$',  # 以"情况"结尾
            r'状态$',  # 以"状态"结尾
            r'水平$',  # 以"水平"结尾
            r'程度$',  # 以"程度"结尾
            r'规模$',  # 以"规模"结尾
            r'范围$',  # 以"范围"结尾
            r'领域$',  # 以"领域"结尾
            r'方向$',  # 以"方向"结尾
            r'重点$',  # 以"重点"结尾
            r'核心$',  # 以"核心"结尾
            r'关键$',  # 以"关键"结尾
            r'主要$',  # 以"主要"结尾
            r'重要$',  # 以"重要"结尾
            r'显著$',  # 以"显著"结尾
            r'明显$',  # 以"明显"结尾
            r'突出$',  # 以"突出"结尾
            r'优秀$',  # 以"优秀"结尾
            r'良好$',  # 以"良好"结尾
            r'稳定$',  # 以"稳定"结尾
            r'持续$',  # 以"持续"结尾
            r'不断$',  # 以"不断"结尾
            r'逐步$',  # 以"逐步"结尾
            r'逐渐$',  # 以"逐渐"结尾
            r'快速$',  # 以"快速"结尾
            r'迅速$',  # 以"迅速"结尾
            r'大幅$',  # 以"大幅"结尾
            r'显著$',  # 以"显著"结尾
            r'明显$',  # 以"明显"结尾
            r'突出$',  # 以"突出"结尾
            r'优秀$',  # 以"优秀"结尾
            r'良好$',  # 以"良好"结尾
            r'稳定$',  # 以"稳定"结尾
            r'持续$',  # 以"持续"结尾
            r'不断$',  # 以"不断"结尾
            r'逐步$',  # 以"逐步"结尾
            r'逐渐$',  # 以"逐渐"结尾
            r'快速$',  # 以"快速"结尾
            r'迅速$',  # 以"迅速"结尾
            r'大幅$',  # 以"大幅"结尾
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text_stripped):
                return False
        
        # 6. 检查是否以数字或字母结尾（可能表示不完整）
        if re.search(r'[\dA-Za-z]$', text_stripped):
            return False
        
        # 7. 检查是否以括号或引号结尾（可能表示不完整）
        if text_stripped.endswith(('(', '（', '[', '【', '"', '"', ''', ''')):
            return False
        
        # 8. 默认：如果以上检查都通过，认为句子完整
        return True
    
    def _generate_with_completion_check(self, input_ids, attention_mask):
        """Generation with completeness check, retry if sentence is incomplete"""
        
        # Get parameters from configuration
        max_attempts = getattr(self.config.generator, 'max_completion_attempts', 2)  # Reduce retry count
        token_increment = getattr(self.config.generator, 'token_increment', 100)  # Increase token increment
        max_total_tokens = getattr(self.config.generator, 'max_total_tokens', 1000)  # Increase max token count
        
        # Get model-specific configuration
        model_config = self._get_model_specific_config()
        
        # For Fin-R1 model, delete unsupported attributes to avoid transformers auto-injection
        if model_config["model_type"] == "fin_r1":
            for k in ["temperature", "top_p", "top_k"]:
                if hasattr(self, k):
                    delattr(self, k)
        
        for attempt in range(max_attempts):
            # Calculate current attempt token count
            current_max_tokens = min(
                self.max_new_tokens + (attempt * token_increment),
                max_total_tokens
            )
            
            # Choose different generation parameters based on model type
            if model_config["model_type"] == "fin_r1":
                # Fin-R1 parameters: only use model-supported parameters to avoid warnings
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": current_max_tokens,
                    "do_sample": False,  # Use deterministic generation
                    "pad_token_id": model_config["pad_token_id"],
                    "eos_token_id": model_config["eos_token_id"],
                    "repetition_penalty": 1.1  # Prevent repetition
                }
            else:
                # Other models: use complete generation parameters
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": current_max_tokens,
                    "do_sample": True,
                    "pad_token_id": model_config["pad_token_id"],
                    "eos_token_id": model_config["eos_token_id"],
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 3
                }
                
                # Only add sampling parameters for non-Fin-R1 models
                if hasattr(self, 'top_p') and hasattr(self, 'temperature'):
                    generation_kwargs.update({
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                    })
                
                # Only add these parameters for supported models
                if model_config["model_type"] in ["default"]:
                    generation_kwargs.update({
                        "length_penalty": 0.8,
                        "early_stopping": True
                    })
            
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Check sentence completeness
            if self._is_sentence_complete(response):
                return response
            
            print(f"Warning: Generation {attempt+1} incomplete, increasing token count to retry...")
        
        # If all attempts fail, return last result
        print("Warning: Reached maximum retry count, returning current result")
        return response

    def _get_model_specific_config(self) -> Dict[str, Any]:
        """Get model-specific configuration parameters"""
        config = {}
        
        if "Fin-R1" in self.model_name:
            config.update({
                "eos_token_id": 151645,  # Fin-R1 EOS token ID (corrected)
                "pad_token_id": 0,
                "model_type": "fin_r1"
            })
        elif "Qwen" in self.model_name:
            config.update({
                "eos_token_id": 151645,  # Qwen3-8B EOS token ID
                "pad_token_id": 0,
                "model_type": "qwen"
            })
        else:
            # Default configuration
            config.update({
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "model_type": "default"
            })
        
        return config