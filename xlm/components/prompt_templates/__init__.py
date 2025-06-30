"""
Prompt Templates Package

This package contains prompt templates for different use cases in the RAG system.
"""

from .template_loader import PromptTemplateLoader, template_loader

__all__ = [
    'PromptTemplateLoader',
    'template_loader'
] 