#!/usr/bin/env python3
"""
共享资源管理器 - 避免重复加载模板和模型
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from threading import Lock

class SharedResourceManager:
    """共享资源管理器 - 单例模式"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._templates: Dict[str, str] = {}
            self._llm_generators: Dict[str, Any] = {}
            self._templates_loaded = False
            self._initialized = True
    
    def get_templates(self) -> Dict[str, str]:
        """获取模板字典，如果未加载则加载"""
        if not self._templates_loaded:
            self._load_templates()
        return self._templates
    
    def get_template(self, template_name: str) -> Optional[str]:
        """获取指定模板"""
        templates = self.get_templates()
        return templates.get(template_name)
    
    def _load_templates(self):
        """加载所有模板文件"""
        if self._templates_loaded:
            return
            
        template_dir = Path("data/prompt_templates")
        if not template_dir.exists():
            print(f"Warning: Template directory {template_dir} does not exist")
            self._templates_loaded = True
            return
        
        for template_file in template_dir.glob("*.txt"):
            template_name = template_file.stem
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self._templates[template_name] = f.read().strip()
                print(f"Loaded template: {template_name}")
            except Exception as e:
                print(f"Error loading template {template_name}: {e}")
        
        self._templates_loaded = True
    
    def get_llm_generator(self, model_name: str, **kwargs) -> Optional[Any]:
        """获取LLM生成器，如果未加载则加载"""
        if model_name not in self._llm_generators:
            # 延迟导入避免循环依赖
            try:
                from xlm.components.generator.local_llm_generator import LocalLLMGenerator
                self._llm_generators[model_name] = LocalLLMGenerator(
                    model_name=model_name,
                    **kwargs
                )
                print(f"Shared LLM Generator '{model_name}' loaded")
            except Exception as e:
                print(f"Error loading shared LLM generator {model_name}: {e}")
                return None
        
        return self._llm_generators[model_name]
    
    def clear_templates(self):
        """清除模板缓存"""
        self._templates.clear()
        self._templates_loaded = False
    
    def clear_llm_generators(self):
        """清除LLM生成器缓存"""
        for generator in self._llm_generators.values():
            if hasattr(generator, 'unload_model'):
                generator.unload_model()
        self._llm_generators.clear()

# 全局实例
shared_resource_manager = SharedResourceManager() 