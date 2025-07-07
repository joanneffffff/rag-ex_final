#!/usr/bin/env python3
"""
上下文分离器
将混合的上下文内容分离为 table_context 和 text_context
让 LLM 更清晰地理解不同类型信息的来源和结构
"""

import re
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class SeparatedContext:
    """分离后的上下文结构"""
    table_context: str
    text_context: str
    context_type: str  # "table", "text", "table-text", "unknown"
    metadata: Dict


class ContextSeparator:
    """上下文分离器"""
    
    def __init__(self):
        """初始化分离器"""
        # 表格标识模式
        self.table_patterns = [
            r'Table ID:.*?\n',
            r'Headers:.*?\n',
            r'Row \d+:.*?\n',
            r'Category:.*?\n',
            r'Table Topic:.*?\n',
            r'Details for item.*?\n',
            r'Other data item:.*?\n',
            r'Data item:.*?\n'
        ]
        
        # 段落标识模式
        self.paragraph_patterns = [
            r'Paragraph ID:.*?\n',
            r'Paragraph \d+:.*?\n',
            r'Note \d+:.*?\n'
        ]
        
        # 表格行模式
        self.table_row_pattern = r'Row \d+:.*?\n'
        
        # 表格头部模式
        self.table_header_pattern = r'Headers:.*?\n'
        
    def separate_context(self, context: str) -> SeparatedContext:
        """
        分离上下文为 table_context 和 text_context
        
        Args:
            context: 原始上下文字符串
            
        Returns:
            SeparatedContext: 分离后的上下文结构
        """
        if not context or not context.strip():
            return SeparatedContext(
                table_context="",
                text_context="",
                context_type="unknown",
                metadata={"error": "Empty context"}
            )
        
        # 1. 判断上下文类型
        context_type = self._determine_context_type(context)
        
        # 2. 根据类型进行分离
        if context_type == "table":
            return self._separate_table_only(context)
        elif context_type == "text":
            return self._separate_text_only(context)
        elif context_type == "table-text":
            return self._separate_mixed_context(context)
        else:
            return self._separate_unknown_context(context)
    
    def _determine_context_type(self, context: str) -> str:
        """判断上下文类型"""
        has_table_id = "Table ID:" in context
        has_paragraph_id = "Paragraph ID:" in context
        has_table_headers = "Headers:" in context
        has_table_rows = bool(re.search(r'Row \d+:', context))
        
        # 检查是否有有意义的文本内容（非表格结构）
        lines = context.split('\n')
        text_lines = 0
        
        for line in lines:
            line = line.strip()
            # 跳过表格结构行
            if (line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:', 'Table Topic:', 
                                'Details for item', 'Other data item', 'Data item')) or
                re.match(r'^[\d\s\-\.,$%()]+$', line)):  # 纯数字/符号行
                continue
            # 检查是否有有意义的文本内容
            if len(line) > 20 and not re.match(r'^[\d\s\-\.,$%()]+$', line):
                text_lines += 1
        
        # 判断类型
        if has_table_id or has_table_headers or has_table_rows:
            if text_lines > 2:
                return "table-text"
            else:
                return "table"
        elif has_paragraph_id or text_lines > 2:
            return "text"
        else:
            return "unknown"
    
    def _separate_table_only(self, context: str) -> SeparatedContext:
        """分离纯表格上下文"""
        # 清理和格式化表格内容
        table_context = self._clean_table_context(context)
        
        return SeparatedContext(
            table_context=table_context,
            text_context="",
            context_type="table",
            metadata={
                "separation_method": "table_only",
                "table_lines": len([line for line in context.split('\n') if 'Row' in line or 'Headers' in line]),
                "has_table_id": "Table ID:" in context
            }
        )
    
    def _separate_text_only(self, context: str) -> SeparatedContext:
        """分离纯文本上下文"""
        # 清理文本内容
        text_context = self._clean_text_context(context)
        
        return SeparatedContext(
            table_context="",
            text_context=text_context,
            context_type="text",
            metadata={
                "separation_method": "text_only",
                "text_length": len(text_context),
                "has_paragraph_id": "Paragraph ID:" in context
            }
        )
    
    def _separate_mixed_context(self, context: str) -> SeparatedContext:
        """分离混合上下文"""
        lines = context.split('\n')
        table_lines = []
        text_lines = []
        
        current_section = "unknown"
        in_table_section = False
        in_paragraph_section = False
        
        for line in lines:
            line = line.strip()
            
            # 检测段落开始
            if line.startswith('Paragraph ID:'):
                current_section = "text"
                in_paragraph_section = True
                in_table_section = False
                text_lines.append(line)
                continue
            
            # 检测表格开始
            if line.startswith('Table ID:'):
                current_section = "table"
                in_table_section = True
                in_paragraph_section = False
                table_lines.append(line)
                continue
            
            # 根据当前部分处理内容
            if in_table_section:
                # 在表格部分
                if (line.startswith(('Headers:', 'Row', 'Category:', 'Table Topic:', 
                                    'Details for item', 'Other data item', 'Data item')) or
                    re.match(r'^[\d\s\-\.,$%()]+$', line) and len(line) < 50):
                    table_lines.append(line)
                elif line.startswith('Paragraph ID:'):
                    # 遇到新的段落，切换到文本部分
                    current_section = "text"
                    in_table_section = False
                    in_paragraph_section = True
                    text_lines.append(line)
                elif line and len(line) > 20:
                    # 有意义的文本内容，可能是表格的说明
                    text_lines.append(line)
                elif line:
                    # 其他表格相关行
                    table_lines.append(line)
            
            elif in_paragraph_section:
                # 在段落部分
                if line.startswith('Table ID:'):
                    # 遇到新的表格，切换到表格部分
                    current_section = "table"
                    in_table_section = True
                    in_paragraph_section = False
                    table_lines.append(line)
                elif line.startswith(('Paragraph ID:', 'Paragraph', 'Note')):
                    text_lines.append(line)
                elif line and len(line) > 5:
                    # 有意义的文本内容
                    text_lines.append(line)
            
            else:
                # 未知部分，根据内容特征判断
                if (line.startswith(('Table ID:', 'Headers:', 'Row', 'Category:', 'Table Topic:', 
                                    'Details for item', 'Other data item', 'Data item')) or
                    re.match(r'^[\d\s\-\.,$%()]+$', line) and len(line) < 50):
                    current_section = "table"
                    in_table_section = True
                    table_lines.append(line)
                elif line.startswith(('Paragraph ID:', 'Paragraph', 'Note')) or len(line) > 20:
                    current_section = "text"
                    in_paragraph_section = True
                    text_lines.append(line)
                elif line:
                    # 默认添加到文本部分
                    text_lines.append(line)
        
        # 清理和格式化
        table_context = self._clean_table_context('\n'.join(table_lines))
        text_context = self._clean_text_context('\n'.join(text_lines))
        
        return SeparatedContext(
            table_context=table_context,
            text_context=text_context,
            context_type="table-text",
            metadata={
                "separation_method": "mixed_context_with_ids",
                "table_lines_count": len(table_lines),
                "text_lines_count": len(text_lines),
                "table_length": len(table_context),
                "text_length": len(text_context),
                "has_table_id": "Table ID:" in context,
                "has_paragraph_id": "Paragraph ID:" in context
            }
        )
    
    def _separate_unknown_context(self, context: str) -> SeparatedContext:
        """处理未知类型的上下文"""
        # 尝试智能分离
        lines = context.split('\n')
        table_lines = []
        text_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否像表格数据
            if (re.match(r'^[\w\s]+\s*\|\s*[\d\-\.,$%()]+', line) or  # 包含分隔符的行
                re.match(r'^[\d\s\-\.,$%()]+$', line) and len(line) < 50):  # 纯数字行
                table_lines.append(line)
            else:
                text_lines.append(line)
        
        table_context = self._clean_table_context('\n'.join(table_lines))
        text_context = self._clean_text_context('\n'.join(text_lines))
        
        # 判断最终类型
        if table_context and text_context:
            final_type = "table-text"
        elif table_context:
            final_type = "table"
        elif text_context:
            final_type = "text"
        else:
            final_type = "unknown"
        
        return SeparatedContext(
            table_context=table_context,
            text_context=text_context,
            context_type=final_type,
            metadata={
                "separation_method": "unknown_context",
                "original_length": len(context),
                "table_lines_count": len(table_lines),
                "text_lines_count": len(text_lines)
            }
        )
    
    def _clean_table_context(self, table_context: str) -> str:
        """清理表格上下文"""
        if not table_context:
            return ""
        
        lines = table_context.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # 保持表格结构，但清理多余的空格
                cleaned_line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_text_context(self, text_context: str) -> str:
        """清理文本上下文"""
        if not text_context:
            return ""
        
        lines = text_context.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:  # 过滤太短的行
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def format_for_prompt(self, separated_context: SeparatedContext, question: str) -> Dict[str, str]:
        """
        格式化分离后的上下文为 prompt 参数
        
        Args:
            separated_context: 分离后的上下文
            question: 问题
            
        Returns:
            Dict: 包含格式化参数的字典
        """
        return {
            "question": question,
            "table_context": separated_context.table_context,
            "text_context": separated_context.text_context,
            "context_type": separated_context.context_type
        }


# 创建全局实例
context_separator = ContextSeparator()


def separate_context_for_prompt(context: str, question: str) -> Dict[str, str]:
    """
    便捷函数：分离上下文并格式化为 prompt 参数
    
    Args:
        context: 原始上下文
        question: 问题
        
    Returns:
        Dict: 格式化后的参数
    """
    separated = context_separator.separate_context(context)
    return context_separator.format_for_prompt(separated, question)


def get_context_separation_info(context: str) -> Dict:
    """
    获取上下文分离信息（不进行实际分离）
    
    Args:
        context: 原始上下文
        
    Returns:
        Dict: 分离信息
    """
    separator = ContextSeparator()
    context_type = separator._determine_context_type(context)
    
    return {
        "context_type": context_type,
        "context_length": len(context),
        "has_table_id": "Table ID:" in context,
        "has_paragraph_id": "Paragraph ID:" in context,
        "has_table_headers": "Headers:" in context,
        "has_table_rows": bool(re.search(r'Row \d+:', context))
    } 