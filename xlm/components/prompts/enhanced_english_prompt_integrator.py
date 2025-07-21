#!/usr/bin/env python3
"""
增强版英文Prompt集成器
集成comprehensive_evaluation_enhanced.py的逻辑到英文处理系统
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from xlm.utils.context_separator import context_separator, separate_context_for_prompt
from xlm.utils.context_separator import context_separator, separate_context_for_prompt

class ContentType(Enum):
    """内容类型枚举"""
    TABLE = "table"
    TEXT = "text"
    TABLE_TEXT = "table-text"
    UNKNOWN = "unknown"

class QueryType(Enum):
    """查询类型枚举"""
    LIST = "list"
    CALC = "calc"
    TEXTUAL = "textual"
    UNKNOWN = "unknown"

def extract_final_answer_with_rescue(raw_output: str) -> str:
    """
    从模型的原始输出中智能提取最终答案。
    它首先尝试寻找<answer>标签，如果失败或为空，则启动救援逻辑从<think>标签中提取。
    """
    def _clean_extracted_text(text: str) -> str:
        """对提取出的文本进行通用清理"""
        text = text.strip()
        # 移除模型可能错误复制进来的 Prompt 指令 (假设这些文本不会出现在正确答案中)
        text = text.replace("[重要：只在这里提供最终答案。无解释，无单位，无多余文本。]", "").strip()
        
        # 移除常见的引导词句，并处理大小写不敏感
        text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 移除末尾可能的多余标点符号，如句号、逗号、分号 (但保留百分号)
        text = re.sub(r'[\.。;,]$', '', text).strip()

        # 标准化百分号 (例如 "percent" -> "%")
        text = re.sub(r'\s*percent\s*', '%', text, flags=re.IGNORECASE).strip()
        
        # 移除常见的货币符号和单位词 (如果你的 expected_answer 不包含这些)
        text = re.sub(r'(\$|million|billion|usd|eur|pounds|£)', '', text, flags=re.IGNORECASE).strip()
        
        # 移除数字中的逗号 (如果你的 expected_answer 不包含逗号)
        text = text.replace(',', '')
        
        # 移除负数括号 (例如 "(33)" -> "-33")
        if text.startswith('(') and text.endswith(')'):
            text = '-' + text[1:-1] # 转换为负数
            
        return text

    # 1. 尝试从 <answer> 标签中提取
    answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        if content:
            return _clean_extracted_text(content)

    # 2. 如果 <answer> 标签失败或为空，尝试从 <think> 标签中提取
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    if not think_match:
        # 如果连 <think> 标签都没有，尝试提取原始输出的最后一行作为答案
        lines = raw_output.strip().split('\n')
        return _clean_extracted_text(lines[-1]) if lines else ""

    think_content = think_match.group(1)
    
    # --- 2.1. 尝试寻找结论性短语 ---
    conclusion_phrases = [
        r'the\s*final\s*answer\s*is[:\s]*',
        r'the\s*answer\s*is[:\s]*', 
        r'therefore,\s*the\s*answer\s*is[:\s]*', 
        r'the\s*result\s*is[:\s]*', 
        r'equals\s*to[:\s]*', 
        r'is\s*equal\s*to[:\s]*', 
        r'the\s*value\s*is[:\s]*', 
        r'the\s*change\s*is[:\s]*', 
        r'the\s*amount\s*is[:\s]*',
        r'conclusion[:\s]*', 
        r'final\s*extracted\s*value/calculated\s*result[:\s]*',
        r'final\s*number[:\s]*',
        r'adjusted\s*net\s*income\s*is[:\s]*',
        r'percentage\s*change\s*is[:\s]*', 
        r'decreased\s*by[:\s]*', 
        r'increased\s*by[:\s]*',
        r'net\s*change\s*is[:\s]*', # 增加更多通用模式
        r'total\s*is[:\s]*',
        r'resulting\s*in[:\s]*', # 捕获 "resulting in X"
        r'is[:\s]*([-+]?[\d,\.]+%?)' # 捕获"is:"后面直接跟的数字或百分比
    ]
    
    for phrase_pattern in conclusion_phrases:
        # 捕获短语后到下一个标签、双换行符或字符串结束的内容 (非贪婪)
        conclusion_match = re.search(
            f'{phrase_pattern}(.*?)(?:$|<answer>|<think>|\\n\\n|\\Z)', 
            think_content, 
            re.IGNORECASE | re.DOTALL 
        )
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            # 确保提取的内容不包含思考过程中的步骤编号
            if re.fullmatch(r'\d+\.', conclusion.split('\n')[0].strip()):
                continue # 如果第一行是步骤编号，跳过
            
            return _clean_extracted_text(conclusion)
    
    # --- 2.2. 如果结论性短语不匹配，尝试寻找最后一个符合数值/百分比/常见格式的字符串 ---
    # 优先匹配行尾的数字或百分比，因为它们更可能是最终答案
    potential_answers_raw = re.findall(r'[-+]?\s*\(?[\d,\.]+\)?%?\s*$', think_content, re.MULTILINE)
    if not potential_answers_raw:
        # 如果行尾没有，在整个文本中从后往前找所有可能的数字/百分比
        potential_answers_raw = re.findall(r'[-+]?\s*\(?[\d,\.]+\)?%?', think_content)
    
    if potential_answers_raw:
        # 逆序遍历，找到最接近末尾且最可能是答案的有效项
        for item_raw in reversed(potential_answers_raw):
            item = item_raw.strip()
            if not item: continue
            
            # 排除明显的步骤编号或短语 (如"1.", "2.", "Step 1:")
            if re.fullmatch(r'(\d+\.|\bstep\s*\d+\b)[:\s]*', item, re.IGNORECASE):
                continue

            cleaned_item = _clean_extracted_text(item)
            
            # 简单的验证，确保不是空的或纯粹的标点
            if cleaned_item and len(cleaned_item) > 0 and not re.fullmatch(r'[^\w\s\d%.-]*', cleaned_item):
                return cleaned_item
                
    # --- 2.3. 最后回退：如果以上都失败，取 <think> 内容的最后一行 ---
    lines = [line for line in think_content.strip().split('\n') if line.strip()]
    if lines:
        return _clean_extracted_text(lines[-1])
    return "" # 如果 think 也是空的，返回空字符串

class EnhancedEnglishPromptIntegrator:
    """增强版英文Prompt集成器"""
    
    def __init__(self):
        """初始化集成器"""
        self.template_info = {
            "name": "enhanced_english_prompt_integrator",
            "version": "2.0",
            "features": ["content_type_detection", "hybrid_decision", "smart_template_selection", "answer_extraction"]
        }
        
        # 加载模板
        self._load_templates()
    
    def _load_templates(self):
        """加载各种模板"""
        self.templates = {
            "table": self._load_template_from_file("template_for_table_answer.txt"),
            "text": self._load_template_from_file("template_for_text_answer.txt"),
            "table_text": self._load_template_from_file("template_for_hybrid_answer.txt"),
            "default": self._load_template_from_file("rag_english_template.txt")
        }
    
    def _load_template_from_file(self, filename: str) -> str:
        """从文件加载模板"""
        try:
            file_path = f"data/prompt_templates/{filename}"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析模板，提取system和user部分
            if "<system>" in content and "<user>" in content:
                system_match = re.search(r'<system>(.*?)</system>', content, re.DOTALL)
                user_match = re.search(r'<user>(.*?)</user>', content, re.DOTALL)
                
                if system_match and user_match:
                    system_content = system_match.group(1).strip()
                    user_content = user_match.group(1).strip()
                    
                    # 重新组合为完整模板
                    return f"<system>\n{system_content}\n</system>\n<user>\n{user_content}\n</user>"
            
            return content
        except FileNotFoundError:
            return self._get_default_template()
        except Exception as e:
            print(f"⚠️ 加载模板文件 {filename} 失败: {e}")
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """默认模板"""
        return """<system>
You are a world-class financial analyst AI. Answer the question based on the provided context.

You MUST respond in the following two-part format:
1.  First, a <think> tag containing your step-by-step reasoning.
2.  Second, an <answer> tag containing the final, concise answer.

<think>
... your step-by-step reasoning process ...
</think>
<answer>
... your final, direct, and concise answer ...
</answer>

---
### Context
{context}

### Question
{question}

### Your Response
"""

    def determine_context_type(self, context: str) -> ContentType:
        """根据context内容判断结构类型，基于Table ID和Paragraph ID"""
        has_table_id = "Table ID:" in context
        has_paragraph_id = "Paragraph ID:" in context
        
        # 移除ID标识行，获取纯内容
        content_without_ids = re.sub(r'(Table ID|Paragraph ID):.*?\n', '', context, flags=re.DOTALL)
        # 移除表格结构标识
        content_without_table = re.sub(r'Headers:.*?\n', '', content_without_ids, flags=re.DOTALL)
        content_without_table = re.sub(r'Row \d+:.*?\n', '', content_without_table, flags=re.DOTALL)
        content_without_table = re.sub(r'Category:.*?\n', '', content_without_table, flags=re.DOTALL)
        
        # 检查是否有有意义的文本内容
        has_meaningful_text = any(len(line.strip()) > 20 for line in content_without_table.split('\n'))
        
        if has_table_id and has_paragraph_id:
            return ContentType.TABLE_TEXT
        elif has_table_id:
            return ContentType.TABLE
        elif has_paragraph_id:
            return ContentType.TEXT
        elif has_meaningful_text:
            return ContentType.TEXT
        else:
            return ContentType.UNKNOWN

    def analyze_query_features(self, query: str) -> Dict[str, bool]:
        """分析查询特征"""
        query_lower = query.lower()
        
        # 列表/枚举特征
        list_indicators = [
            'list', 'all', 'every', 'each', 'both', 'multiple', 'several',
            'what are', 'which of', 'name all', 'identify all', 'find all'
        ]
        is_list = any(indicator in query_lower for indicator in list_indicators)
        
        # 计算特征
        calc_indicators = [
            'calculate', 'compute', 'sum', 'total', 'average', 'mean', 'percentage',
            'how much', 'what is the total', 'what is the sum', 'what percentage'
        ]
        is_calc = any(indicator in query_lower for indicator in calc_indicators)
        
        # 解释性/文本特征
        textual_indicators = [
            'explain', 'describe', 'what does', 'what is', 'how does', 'why',
            'define', 'meaning', 'purpose', 'reason', 'method', 'approach'
        ]
        is_textual = any(indicator in query_lower for indicator in textual_indicators)
        
        return {
            'is_list': is_list,
            'is_calc': is_calc,
            'is_textual': is_textual
        }

    def hybrid_decision(self, context: str, query: str) -> ContentType:
        """混合决策算法，基于Table ID和Paragraph ID进行精确路由"""
        context_type = self.determine_context_type(context)
        query_features = self.analyze_query_features(query)
        
        # 优先级最高：如果问题明确是列表/枚举
        if query_features['is_list']:
            if context_type == ContentType.TABLE:
                return ContentType.TABLE
            elif context_type == ContentType.TEXT:
                return ContentType.TEXT
            elif context_type == ContentType.TABLE_TEXT:
                return ContentType.TABLE_TEXT
            else:  # UNKNOWN
                return ContentType.TEXT  # 默认回退到文本处理

        # 第二优先级：计算性问题，强烈依赖数值数据
        if query_features['is_calc']:
            if context_type == ContentType.TABLE:
                return ContentType.TABLE
            elif context_type == ContentType.TABLE_TEXT:
                return ContentType.TABLE_TEXT
            else:  # TEXT or UNKNOWN
                return ContentType.TEXT

        # 第三优先级：解释性/事实性问题
        if query_features['is_textual']:
            if context_type == ContentType.TEXT:
                return ContentType.TEXT
            elif context_type == ContentType.TABLE_TEXT:
                return ContentType.TEXT  # 解释性问题优先从文本获取
            elif context_type == ContentType.TABLE:
                return ContentType.TABLE
            else:  # UNKNOWN
                return ContentType.TEXT

        # 默认情况：基于内容类型
        if context_type == ContentType.UNKNOWN:
            return ContentType.TEXT  # 默认回退到文本处理
        else:
            return context_type

    def create_enhanced_prompt(self, context: str, question: str, summary: Optional[str] = None) -> Tuple[str, Dict]:
        """创建增强版prompt，同时使用summary和context"""
        # 1. 内容类型判断
        context_type = self.determine_context_type(context)
        
        # 2. 混合决策
        decision_type = self.hybrid_decision(context, question)
        
        # 3. 查询特征分析
        query_features = self.analyze_query_features(question)
        
        # 4. 选择模板
        if decision_type == ContentType.TABLE:
            template = self.templates["table"]
        elif decision_type == ContentType.TEXT:
            template = self.templates["text"]
        elif decision_type == ContentType.TABLE_TEXT:
            template = self.templates["table_text"]
        else:
            template = self.templates["default"]
        
        # 5. 准备完整上下文（包含summary和context）
        full_context = context
        if summary and summary.strip():
            # 如果有summary，将其添加到context前面
            full_context = f"Summary: {summary}\n\nFull Context: {context}"
        
        # 6. 格式化prompt
        formatted_prompt = template.format(
            context=full_context,
            question=question
        )
        
        # 7. 返回结果和元数据
        metadata = {
            "context_type": context_type.value,
            "decision_type": decision_type.value,
            "query_features": query_features,
            "template_used": list(self.templates.keys())[list(self.templates.values()).index(template)],
            "enhanced_logic": True,
            "has_summary": bool(summary and summary.strip()),
            "summary_length": len(summary) if summary else 0,
            "context_length": len(context)
        }
        
        return formatted_prompt, metadata

    def extract_answer_from_response(self, raw_response: str) -> str:
        """从模型响应中提取最终答案"""
        return extract_final_answer_with_rescue(raw_response)

    def get_template_info(self) -> Dict:
        """获取模板信息"""
        return self.template_info

# 创建全局实例
enhanced_english_prompt_integrator = EnhancedEnglishPromptIntegrator() 