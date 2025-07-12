from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput
from xlm.components.prompt_templates.template_loader import template_loader
import os
import collections
import re
from langdetect import detect, LangDetectException
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

# 导入增强版英文prompt集成器
try:
    from xlm.components.prompts.enhanced_english_prompt_integrator import EnhancedEnglishPromptIntegrator, extract_final_answer_with_rescue
    ENHANCED_ENGLISH_AVAILABLE = True
except ImportError:
    ENHANCED_ENGLISH_AVAILABLE = False
    print("⚠️ 增强版英文prompt集成器不可用，将使用基础模板")

# ===================================================================
# 英文Prompt模板处理函数 - 与comprehensive_evaluation_enhanced_new_1.py保持一致
# ===================================================================

def _load_template_content_from_file_english(template_file_name: str) -> Optional[str]:
    """Loads the full string content of an English Prompt template from a specified file."""
    template_path = Path("data/prompt_templates") / template_file_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ English Template file not found: {template_path}. Please ensure the file exists.")
        return None

def _parse_template_string_to_messages(template_full_string: str, query: str, context: str = "") -> List[Dict[str, str]]:
    """
    Parses the template string and creates messages list for English evaluation.
    This function is adapted from comprehensive_evaluation_enhanced_new_1.py
    """
    messages = []
    
    # 1. Extract SYSTEM message (everything from ===SYSTEM=== to ===USER===)
    system_match = re.search(r'===SYSTEM===(.*?)===USER===', template_full_string, re.DOTALL)
    if system_match:
        system_content = system_match.group(1).strip()
        # Clean up unwanted parts from SYSTEM content
        system_content = re.sub(r'---CRITICAL RULES for the <answer> tag[\s\S]*', '', system_content).strip()
        system_content = re.sub(r'---[\s\S]*', '', system_content).strip()
        messages.append({"role": "system", "content": system_content})
    
    # 2. Create USER message with the actual query and context
    # The template has examples, but we'll create a simple format for the actual query
    user_content = f"Q: {query}\nTable Context: {context}\nText Context: {context}\n<answer>"
    messages.append({"role": "user", "content": user_content})

    return messages

def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Converts messages list to ChatML format string expected by Fin-R1 (Qwen2.5 based).
    """
    if not messages:
        return ""

    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            formatted_prompt += f"<|im_start|>system\n{content.strip()}<|im_end|>\n"
        elif role == "user":
            formatted_prompt += f"<|im_start|>user\n{content.strip()}<|im_end|>\n"
        elif role == "assistant":
            # Assistant role is usually part of few-shot examples
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"

    # Append assistant start tag to indicate model should start generating new assistant response
    formatted_prompt += "<|im_start|>assistant\n"

    return formatted_prompt

def get_final_prompt_messages_english(context: str, query: str) -> List[Dict[str, str]]:
    """
    Constructs the messages list for English evaluation, using the specified template file.
    This function is adapted from comprehensive_evaluation_enhanced_new_1.py
    """
    template_file_name = "unified_english_template_no_think.txt"
    template_full_string = _load_template_content_from_file_english(template_file_name)
    
    if template_full_string is None:
        # 回退到简单prompt
        return [{"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer:"}]

    return _parse_template_string_to_messages(template_full_string, query, context)

# ===================================================================
# 答案提取函数 - 与comprehensive_evaluation_enhanced_new_1.py保持一致
# ===================================================================

def _shared_text_standardizer_english(text: str) -> str:
    """
    Helper function to standardize English text for both answer extraction and F1 score calculation.
    Strictly follows the rules from the English Prompt Template.
    """
    text = text.strip()
    
    # Lowercase all text
    text = text.lower()

    # 递归替换所有 \text{...} 为 ...（保留内容）
    while True:
        new_text = re.sub(r'\\text\{([^}]*)\}', r'\1', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    # 其余 LaTeX 格式直接去掉
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove currency symbols and common unit words based on prompt rule
    text = re.sub(r'\b(million|billion|thousand|trillion|usd|eur|gbp|m|b)\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[\$£€]', '', text).strip()

    # Remove commas from numbers
    text = text.replace(',', '')

    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Normalize percentages
    text = text.replace(' percent', '%').replace('pct', '%')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    
    # Remove common introductory phrases
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation
    if text.endswith('%'):
        text = re.sub(r'[\.,;]$', '', text).strip()
    else:
        text = re.sub(r'[\.,;%]$', '', text).strip() 
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()

    return text

def extract_final_answer_from_tag(raw_output: str) -> str:
    """
    Extracts the final answer from the model's raw output by looking for the <answer> tag.
    Returns NOT_FOUND_REPLY_ENGLISH if no valid answer found or tag is empty.
    """
    NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."
    
    # First, try to find <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        # Ensure extracted content is not empty or an empty tag itself (e.g., <answer></answer>)
        if content and content.lower() not in ['<final></final>', '<answer></answer>', '<final-answer></final-answer>']:
            
            # Try to extract the most concise answer from the content
            # Look for patterns that might contain the actual answer
            
            # 1. Look for boxed answers: \boxed{...}
            # 使用更复杂的正则表达式来处理嵌套大括号
            boxed_match = re.search(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', content)
            if boxed_match:
                return _shared_text_standardizer_english(boxed_match.group(1))
            
            # 2. Look for percentage patterns: 12.82%
            percentage_match = re.search(r'(\d+\.?\d*)\s*%', content)
            if percentage_match:
                return _shared_text_standardizer_english(percentage_match.group(0))
            
            # 3. Look for numerical answers at the end of sentences
            # This is for cases like "Thus, the answer is 12.82%"
            final_number_match = re.search(r'(?:thus|therefore|answer is|result is)\s+(?:approximately\s+)?(\d+\.?\d*)', content, re.IGNORECASE)
            if final_number_match:
                return _shared_text_standardizer_english(final_number_match.group(1))
            
            # 4. Look for the largest numerical value (likely the answer)
            # This helps when there are multiple numbers in the text
            numbers = re.findall(r'\b(\d+(?:,\d+)*)\b', content)
            if numbers:
                # Convert to integers for comparison, removing commas
                number_values = [int(num.replace(',', '')) for num in numbers]
                largest_number = max(number_values)
                return _shared_text_standardizer_english(str(largest_number))
            
            # 5. If no specific pattern found, return the original content
            return _shared_text_standardizer_english(content)
    
    # If no <answer> tags found, look for boxed answers in the entire text
    # 使用更复杂的正则表达式来处理嵌套大括号
    boxed_match = re.search(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', raw_output)
    if boxed_match:
        return _shared_text_standardizer_english(boxed_match.group(1))
    
    # If no valid <answer> structure is found or content is invalid,
    # return the specific "not found" phrase.
    return NOT_FOUND_REPLY_ENGLISH

# ===================================================================
# RAG系统类
# ===================================================================

class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retriever_top_k: int,
        prompt_template: Optional[str] = None, # No longer used, but kept for compatibility
        use_cot: bool = False,  # 是否使用Chain-of-Thought
        use_simple: bool = False,  # 是否使用超简洁模式
        use_enhanced_english: bool = True,  # 是否使用增强版英文prompt
    ):
        self.retriever = retriever
        self.generator = generator
        # self.prompt_template is now obsolete
        self.retriever_top_k = retriever_top_k
        self.use_cot = use_cot
        self.use_simple = use_simple
        self.use_enhanced_english = use_enhanced_english and ENHANCED_ENGLISH_AVAILABLE
        
        # 初始化增强版英文prompt集成器
        if self.use_enhanced_english:
            self.enhanced_english_integrator = EnhancedEnglishPromptIntegrator()
        else:
            self.enhanced_english_integrator = None

    def run(self, user_input: str, language: Optional[str] = None) -> RagOutput:
        # 1. Detect language of the user's question
        if language is None:
            try:
                lang = detect(user_input)
            except LangDetectException:
                lang = 'en' # Default to English if detection fails
            is_chinese_q = lang.startswith('zh')
            language = 'zh' if is_chinese_q else 'en'
        else:
            is_chinese_q = (language == 'zh')
        
        # 2. Retrieve relevant documents
        print(f"开始统一RAG检索...")
        print(f"查询: {user_input}")
        print(f"语言: {language}")
        
        # 检查检索器类型和配置
        use_faiss = getattr(self.retriever, 'use_faiss', False)
        has_reranker = hasattr(self.retriever, 'reranker') and getattr(self.retriever, 'reranker', None) is not None
        
        # 获取详细的配置信息
        config_obj = getattr(self.retriever, 'config', None)
        if config_obj and hasattr(config_obj, 'retriever'):
            retrieval_top_k = config_obj.retriever.retrieval_top_k
            rerank_top_k = config_obj.retriever.rerank_top_k
        else:
            retrieval_top_k = 100  # 默认值
            rerank_top_k = 10      # 默认值
        
        print(f"使用FAISS: {use_faiss}")
        print(f"启用重排序器: {has_reranker}")
        print(f"FAISS检索数量: {retrieval_top_k}")
        print(f"重排序器数量: {rerank_top_k}")
        
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=user_input, top_k=self.retriever_top_k, return_scores=True
        )
        
        # 安全地获取文档数量
        doc_count = len(retrieved_documents) if isinstance(retrieved_documents, list) else 1
        print(f"FAISS检索完成，找到 {doc_count} 个文档")
        if has_reranker:
            print(f"重排序器处理完成，返回 {doc_count} 个文档")
        print(f"使用生成器生成答案...")

        # 3. For Chinese queries, we should use multi-stage retrieval system
        # For now, we'll use a simple fallback for Chinese queries
        if is_chinese_q:
            no_context_message = "未找到合适的语料，请检查数据源。建议使用多阶段检索系统处理中文查询。"
        else:
            no_context_message = "No suitable context found for your question. Please check the data sources."

        if not retrieved_documents or (isinstance(retrieved_documents, list) and len(retrieved_documents) == 0):
            return RagOutput(
                retrieved_documents=[],
                retriever_scores=[],
                prompt="",
                generated_responses=[no_context_message],
                metadata={}
            )

        # 构建上下文字符串
        if isinstance(retrieved_documents, list):
            context_parts = []
            for doc in retrieved_documents:
                if hasattr(doc, 'content'):
                    content = doc.content
                    # 处理不同类型的content
                    if isinstance(content, dict):
                        # 如果是字典，优先提取context字段，然后是content字段
                        content_dict = content  # type: Dict[str, Any]
                        if 'context' in content_dict:
                            context_parts.append(str(content_dict.get('context', '')))
                        elif 'content' in content_dict:
                            context_parts.append(str(content_dict.get('content', '')))
                        else:
                            # 如果没有找到context或content字段，将整个字典转为字符串
                            context_parts.append(str(content))
                    elif isinstance(content, str):
                        context_parts.append(content)
                    else:
                        # 其他类型转为字符串
                        context_parts.append(str(content))
            
            context_str = "\n\n".join(context_parts)
        else:
            context_str = str(retrieved_documents)
        
        # 4. Create the final prompt using unified logic for English queries
        try:
            if is_chinese_q:
                # 中文查询使用多阶段检索系统，这里只是回退
                prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{user_input}\n\n回答："
                template_type = "ZH-MULTI-STAGE"
            else:
                # 英文查询使用与comprehensive_evaluation_enhanced_new_1.py相同的逻辑
                # 移除混合决策，只使用unified_english_template_no_think.txt模板
                messages = get_final_prompt_messages_english(context_str, user_input)
                prompt = _convert_messages_to_chatml(messages)
                template_type = "EN-UNIFIED-TEMPLATE"
                print(f"使用统一英文模板: unified_english_template_no_think.txt")
        except Exception as e:
            # 回退到简单prompt
            if is_chinese_q:
                prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{user_input}\n\n回答："
                template_type = "ZH-FALLBACK"
            else:
                prompt = f"Context: {context_str}\nQuestion: {user_input}\nAnswer:"
                template_type = "EN-FALLBACK"
        
        # 5. Generate the response
        try:
            generated_responses = self.generator.generate(texts=[prompt])
        except Exception as e:
            raise e
        
        # 6. 对英文查询进行答案提取处理 - 使用与comprehensive_evaluation_enhanced_new_1.py相同的逻辑
        if not is_chinese_q:
            try:
                # 提取最终答案
                raw_response = generated_responses[0] if generated_responses else ""
                extracted_answer = extract_final_answer_from_tag(raw_response)
                
                # 如果提取成功，替换原始响应
                if extracted_answer and extracted_answer.strip():
                    generated_responses = [extracted_answer]
                    print(f"答案提取成功: {extracted_answer[:100]}...")
                else:
                    print("答案提取失败，使用原始响应")
            except Exception as e:
                print(f"答案提取过程出错: {e}，使用原始响应")
        
        # 7. Gather metadata
        retriever_model_name = ""
        # 安全地检查retriever是否有encoder属性
        try:
            if hasattr(self.retriever, 'encoder_ch') and hasattr(self.retriever, 'encoder_en'):
                if is_chinese_q:
                    encoder_ch = getattr(self.retriever, 'encoder_ch', None)
                    if encoder_ch and hasattr(encoder_ch, 'model_name'):
                        retriever_model_name = getattr(encoder_ch, 'model_name', 'unknown')
                else:
                    encoder_en = getattr(self.retriever, 'encoder_en', None)
                    if encoder_en and hasattr(encoder_en, 'model_name'):
                        retriever_model_name = getattr(encoder_en, 'model_name', 'unknown')
        except Exception:
            retriever_model_name = "unknown"

        # 确保retrieved_documents是列表类型
        if not isinstance(retrieved_documents, list):
            retrieved_documents = [retrieved_documents]
        
        # 确保retriever_scores是列表类型且包含float值
        if not isinstance(retriever_scores, list):
            retriever_scores = [retriever_scores] if retriever_scores is not None else []
        
        # 确保retriever_scores只包含float值
        float_scores = []
        for score in retriever_scores:
            if isinstance(score, (int, float)):
                float_scores.append(float(score))
            else:
                float_scores.append(0.0)

        # 构建增强的metadata
        metadata_dict = dict(
            retriever_model_name=retriever_model_name,
            top_k=self.retriever_top_k,
            generator_model_name=self.generator.model_name,
            prompt_template=f"Template-{template_type}",
            question_language="zh" if is_chinese_q else "en",
            use_cot=self.use_cot,
            use_simple=self.use_simple,
            use_enhanced_english=self.use_enhanced_english
        )

        result = RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=float_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=metadata_dict,
        )
        return result
