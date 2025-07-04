#!/usr/bin/env python3
"""
RAG系统增强器
集成英文Prompt流程到现有RAG系统
"""

# 临时关闭warnings，避免transformers参数警告
import warnings
warnings.filterwarnings("ignore")

# 更精确地过滤transformers生成参数警告
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入RAG系统的LocalLLMGenerator
try:
    from xlm.components.generator.local_llm_generator import LocalLLMGenerator
    USE_RAG_GENERATOR = True
    print("✅ 使用RAG系统的LocalLLMGenerator")
except ImportError as e:
    USE_RAG_GENERATOR = False
    print(f"⚠️ 导入RAG组件失败: {e}")
    print("请确保RAG系统已正确安装")

try:
    from xlm.components.prompts.english_prompt_integrator import english_prompt_integrator
    english_prompt_integrator_available = True
except ImportError as e:
    print(f"⚠️ 导入english_prompt_integrator失败: {e}")
    english_prompt_integrator_available = False

# 条件导入RAG组件
FAISSRetriever_available = False
CrossEncoderReranker_available = False
FAISSRetriever = None
CrossEncoderReranker = None

try:
    from xlm.components.retriever.faiss_retriever import FAISSRetriever
    FAISSRetriever_available = True
except ImportError as e:
    print(f"⚠️ 导入FAISSRetriever失败: {e}")

try:
    from xlm.components.reranker.cross_encoder_reranker import CrossEncoderReranker
    CrossEncoderReranker_available = True
except ImportError as e:
    print(f"⚠️ 导入CrossEncoderReranker失败: {e}")

class EnhancedRAGSystem:
    """增强版RAG系统"""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.llm_generator = None
        self.retriever = None
        self.reranker = None
        self.english_prompt_integrator = english_prompt_integrator if 'english_prompt_integrator' in globals() else None
        
    def initialize_components(self):
        """初始化组件"""
        print("🔄 初始化RAG组件...")
        
        try:
            # 初始化LLM生成器
            if USE_RAG_GENERATOR:
                self.llm_generator = LocalLLMGenerator(
                    model_name="SUFE-AIFLM-Lab/Fin-R1",
                    device=self.device,
                    use_quantization=True,
                    quantization_type="4bit"
                )
                print("✅ LLM生成器初始化成功")
            else:
                print("⚠️ 无法使用RAG系统的LocalLLMGenerator")
                self.llm_generator = None
            
            # 初始化检索器（如果可用）
            if FAISSRetriever_available and FAISSRetriever:
                try:
                    self.retriever = FAISSRetriever()
                    print("✅ 检索器初始化成功")
                except Exception as e:
                    print(f"⚠️ 检索器初始化失败: {e}")
                    self.retriever = None
            else:
                print("⚠️ 检索器组件不可用")
                self.retriever = None
            
            # 初始化重排序器（如果可用）
            if CrossEncoderReranker_available and CrossEncoderReranker:
                try:
                    self.reranker = CrossEncoderReranker()
                    print("✅ 重排序器初始化成功")
                except Exception as e:
                    print(f"⚠️ 重排序器初始化失败: {e}")
                    self.reranker = None
            else:
                print("⚠️ 重排序器组件不可用")
                self.reranker = None
            
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise
    
    def process_english_query(self, query: str, context: str, top_k: int = 5) -> Dict[str, Any]:
        """处理英文查询"""
        try:
            if not self.llm_generator:
                return {
                    "query": query,
                    "context": context,
                    "error": "LLM生成器未初始化",
                    "success": False
                }
            
            # 1. 创建英文Prompt
            if self.english_prompt_integrator:
                # 从context中提取summary（如果有的话）
                summary = None
                if "Summary:" in context and "Full Context:" in context:
                    # 如果context已经包含summary格式，提取出来
                    parts = context.split("Full Context:", 1)
                    if len(parts) == 2:
                        summary = parts[0].replace("Summary:", "").strip()
                        context = parts[1].strip()
                
                prompt = self.english_prompt_integrator.create_english_prompt(
                    context=context,
                    question=query,
                    summary=summary
                )
            else:
                # 备用方案：简单prompt
                prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:"
            
            # 2. 生成回答
            responses = self.llm_generator.generate([prompt])
            generated_answer = responses[0] if responses else ""
            
            # 3. 后处理
            cleaned_answer = self._clean_response(generated_answer)
            
            return {
                "query": query,
                "context": context,
                "raw_response": generated_answer,
                "cleaned_answer": cleaned_answer,
                "template_info": self.english_prompt_integrator.get_template_info() if self.english_prompt_integrator else {"name": "Simple Template"},
                "success": True
            }
            
        except Exception as e:
            print(f"❌ 处理英文查询失败: {e}")
            return {
                "query": query,
                "context": context,
                "error": str(e),
                "success": False
            }
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """将messages转换为文本格式"""
        if not messages:
            return ""
        
        text_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if content:
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
                else:
                    text_parts.append(content)
        
        return "\\n".join(text_parts)
    
    def _clean_response(self, response: str) -> str:
        """简单的响应清理"""
        if not response:
            return ""
        
        # 移除常见的格式标记
        response = response.replace("**", "").replace("*", "").replace("```", "")
        response = response.strip()
        
        return response
    
    def process_multilingual_query(self, query: str, context: str, language: str = "auto") -> Dict[str, Any]:
        """处理多语言查询"""
        # 检测语言
        if language == "auto":
            language = self.detect_language(query)
        
        if language == "english":
            return self.process_english_query(query, context)
        else:
            # 使用原有的多语言处理逻辑
            return self.process_other_language_query(query, context, language)
    
    def detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 简单的英文检测
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        total_chars = sum(1 for char in text if char.isalpha())
        
        if total_chars > 0 and english_chars / total_chars > 0.8:
            return "english"
        else:
            return "chinese"  # 默认为中文
    
    def process_other_language_query(self, query: str, context: str, language: str) -> Dict[str, Any]:
        """处理其他语言查询（使用原有逻辑）"""
        # 这里可以集成原有的多语言处理逻辑
        return {
            "query": query,
            "context": context,
            "language": language,
            "message": "使用原有多语言处理逻辑",
            "success": True
        }

def create_enhanced_rag_system(device: str = "auto") -> EnhancedRAGSystem:
    """创建增强版RAG系统"""
    system = EnhancedRAGSystem(device=device)
    system.initialize_components()
    return system
