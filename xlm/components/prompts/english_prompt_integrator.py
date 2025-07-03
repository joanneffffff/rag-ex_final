#!/usr/bin/env python3
"""
英文Prompt集成模块
将验证过的英文Prompt模板集成到RAG系统中
"""

from typing import List, Dict, Any, Optional
from xlm.components.prompt_templates.template_loader import template_loader

class EnglishPromptIntegrator:
    """英文Prompt集成器"""
    
    def __init__(self):
        self.template_name = "rag_english_template"
    
    def create_english_prompt(self, context: str, question: str, summary: Optional[str] = None) -> str:
        """创建英文Prompt字符串"""
        # 使用模板加载器格式化模板，与中文模板保持一致
        prompt = template_loader.format_template(
            self.template_name,
            context=context,
            question=question
        )
        
        if prompt is None:
            # 回退到简单英文prompt
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        return prompt
    
    def create_english_prompt_messages(self, context: str, question: str, summary: Optional[str] = None) -> List[Dict[str, str]]:
        """创建英文Prompt消息列表（JSON聊天格式）"""
        # 首先获取格式化的模板字符串
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
        return {
            "name": self.template_name,
            "language": "english",
            "optimized": True,
            "description": "经过验证的英文Prompt模板，在TatQA数据集上表现优秀，包含Few-Shot COT示例",
            "file_path": "data/prompt_templates/rag_english_template.txt",
            "format": "===SYSTEM=== and ===USER==="
        }

# 全局实例
english_prompt_integrator = EnglishPromptIntegrator()
