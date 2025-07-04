#!/usr/bin/env python3
"""
增强版英文Prompt集成模块
集成comprehensive_evaluation_enhanced.py的逻辑到英文处理系统
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from xlm.components.prompt_templates.template_loader import template_loader
from xlm.components.prompts.enhanced_english_prompt_integrator import enhanced_english_prompt_integrator

class EnglishPromptIntegrator:
    """增强版英文Prompt集成器"""
    
    def __init__(self):
        self.template_name = "rag_english_template"
        self.enhanced_integrator = enhanced_english_prompt_integrator
        self.use_enhanced_logic = True  # 默认使用增强逻辑
    
    def create_english_prompt(self, context: str, question: str, summary: Optional[str] = None) -> str:
        """创建英文Prompt字符串"""
        if self.use_enhanced_logic:
            # 使用增强版逻辑
            enhanced_prompt, metadata = self.enhanced_integrator.create_enhanced_prompt(context, question, summary)
            return enhanced_prompt
        else:
            # 使用原始模板逻辑
            prompt = template_loader.format_template(
                self.template_name,
                context=context,
                question=question
            )
            
            if prompt is None:
                # 回退到简单英文prompt
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            return prompt
    
    def create_english_prompt_messages(self, context: str, question: str, summary: Optional[str] = None) -> List[Dict[str, Any]]:
        """创建英文Prompt消息列表（JSON聊天格式）"""
        if self.use_enhanced_logic:
            # 使用增强版逻辑
            enhanced_prompt, metadata = self.enhanced_integrator.create_enhanced_prompt(context, question, summary)
            
            # 解析prompt为system和user部分
            if "<system>" in enhanced_prompt and "<user>" in enhanced_prompt:
                # 提取system和user部分
                system_match = re.search(r'<system>(.*?)</system>', enhanced_prompt, re.DOTALL)
                user_match = re.search(r'<user>(.*?)</user>', enhanced_prompt, re.DOTALL)
                
                if system_match and user_match:
                    system_content = system_match.group(1).strip()
                    user_content = user_match.group(1).strip()
                    
                    # 构建JSON格式
                    chat_data = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    
                    # 添加元数据到第一条消息
                    chat_data[0]["metadata"] = metadata
                    
                    return chat_data
            
            # 如果不是标准格式，返回简单的用户消息
            return [
                {"role": "user", "content": enhanced_prompt, "metadata": metadata}
            ]
        else:
            # 使用原始逻辑
            formatted_prompt = self.create_english_prompt(context, question, summary)
            
            # 检测是否包含===SYSTEM===和===USER===格式
            if "===SYSTEM===" in formatted_prompt and "===USER===" in formatted_prompt:
                # 提取SYSTEM和USER部分
                system_start = formatted_prompt.find("===SYSTEM===")
                user_start = formatted_prompt.find("===USER===")
                
                if system_start != -1 and user_start != -1:
                    system_content = formatted_prompt[system_start + 12:user_start].strip()
                    user_content = formatted_prompt[user_start + 10:].strip()
                    
                    # 构建JSON格式
                    chat_data = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    
                    return chat_data
            
            # 如果不是标准格式，返回简单的用户消息
            return [
                {"role": "user", "content": formatted_prompt}
            ]
    
    def get_template_info(self) -> Dict[str, Any]:
        """获取模板信息"""
        base_info = {
            "name": self.template_name,
            "language": "english",
            "optimized": True,
            "description": "增强版英文Prompt模板，集成comprehensive_evaluation_enhanced.py逻辑",
            "file_path": "data/prompt_templates/rag_english_template.txt",
            "format": "enhanced_with_metadata"
        }
        
        if self.use_enhanced_logic:
            base_info.update({
                "enhanced_logic": True,
                "features": self.enhanced_integrator.get_template_info()["features"],
                "version": self.enhanced_integrator.get_template_info()["version"]
            })
        
        return base_info
    
    def set_enhanced_logic(self, enabled: bool):
        """设置是否使用增强逻辑"""
        self.use_enhanced_logic = enabled
    
    def get_enhanced_metadata(self, context: str, question: str, summary: Optional[str] = None) -> Dict[str, Any]:
        """获取增强逻辑的元数据"""
        if self.use_enhanced_logic:
            _, metadata = self.enhanced_integrator.create_enhanced_prompt(context, question, summary)
            return metadata
        else:
            return {"enhanced_logic": False}
    
    def extract_answer_from_response(self, raw_response: str) -> str:
        """从模型响应中提取最终答案"""
        if self.use_enhanced_logic:
            return self.enhanced_integrator.extract_answer_from_response(raw_response)
        else:
            # 简单回退：直接返回原始响应
            return raw_response.strip()

# 全局实例
english_prompt_integrator = EnglishPromptIntegrator()
