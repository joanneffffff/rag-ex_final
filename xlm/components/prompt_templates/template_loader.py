"""
Prompt Template Loader

This module loads prompt templates from text files.
"""

import os
from pathlib import Path
from typing import Dict, Optional


class PromptTemplateLoader:
    """Prompt模板加载器"""
    
    def __init__(self, template_dir: str = "data/prompt_templates"):
        self.template_dir = Path(template_dir)
        self._templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        """加载所有模板文件"""
        if not self.template_dir.exists():
            print(f"Warning: Template directory {self.template_dir} does not exist")
            return
        
        for template_file in self.template_dir.glob("*.txt"):
            template_name = template_file.stem
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self._templates[template_name] = f.read().strip()
                print(f"Loaded template: {template_name}")
            except Exception as e:
                print(f"Error loading template {template_name}: {e}")
    
    def get_template(self, template_name: str) -> Optional[str]:
        """获取指定名称的模板"""
        return self._templates.get(template_name)
    
    def format_template(self, template_name: str, **kwargs) -> Optional[str]:
        """格式化模板"""
        template = self.get_template(template_name)
        if template is None:
            return None
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Error formatting template {template_name}: missing key {e}")
            return None
        except Exception as e:
            print(f"Error formatting template {template_name}: {e}")
            return None
    
    def list_templates(self) -> list:
        """列出所有可用的模板"""
        return list(self._templates.keys())


# 全局实例
template_loader = PromptTemplateLoader() 