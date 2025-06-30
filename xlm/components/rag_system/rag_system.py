from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput
from xlm.components.prompt_templates.template_loader import template_loader
import os
import collections
import re
from langdetect import detect, LangDetectException
from typing import List, Union, Optional, Dict, Any

# Define the robust "Golden Prompts" directly in the code (only for English)
PROMPT_TEMPLATE_EN = """You are a highly analytical and precise financial expert. Your task is to answer the user's question **strictly based on the provided <context> information**.

**CRITICAL: Your output must be a pure, direct answer. Do NOT include any self-reflection, thinking process, prompt analysis, irrelevant comments, format markers (like boxed, numbered lists, bold text), or any form of meta-commentary. Do NOT quote or restate the prompt content. Your answer must end directly and concisely without any follow-up explanations.**

Requirements:
1.  **Strictly adhere to the provided <context>. Do not use any external knowledge or make assumptions.**
2.  If the <context> does not contain sufficient information to answer the question, state: "The answer cannot be found in the provided context."
3.  For questions involving financial predictions or future outlook, prioritize information explicitly stated as forecasts or outlooks within the <context>. If the <context> specifies a report date, base your answer on that date's perspective.
4.  Provide a concise and direct answer in complete sentences.
5.  Do not repeat the question or add conversational fillers.

Example 1:
Context: Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. iPhone sales increased 2.8% to $39.7 billion.
Question: How did Apple perform in Q3 2023?
Answer: Apple's Q3 2023 performance was mixed. Total revenue declined 1.4% to $81.8 billion, but iPhone sales grew 2.8% to $39.7 billion.

Example 2:
Context: Tesla's vehicle deliveries in Q2 2023 reached 466,140 units, up 83% from the previous year. Production capacity utilization improved to 95%.
Question: What were Tesla's delivery numbers in Q2 2023?
Answer: Tesla delivered 466,140 vehicles in Q2 2023, representing an 83% increase from the previous year.

Context:
{context}

Question: {question}

Answer:"""

# 超简洁版本（推荐用于生产环境）
PROMPT_TEMPLATE_ZH_SIMPLE = """基于上下文信息，用一句话回答用户问题。不要添加任何格式标记、编号或额外说明。

**极度重要：你的输出必须是纯粹、直接的回答，不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗）、或任何形式的元评论。请勿引用或复述Prompt内容。你的回答必须直接、简洁地结束，不带任何引导语或后续说明。**

上下文：{context}
问题：{question}
回答："""

# 简洁版本（平衡质量和长度）
PROMPT_TEMPLATE_ZH_CLEAN = """基于以下上下文信息，直接回答用户问题。要求：
1. 只使用提供的信息
2. 回答要简洁，不超过100字
3. 不要添加任何格式标记、编号或额外说明
4. 用自然的中文表达

**极度重要：你的输出必须是纯粹、直接的回答，不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗）、或任何形式的元评论。请勿引用或复述Prompt内容。你的回答必须直接、简洁地结束，不带任何引导语或后续说明。**

上下文：{context}
问题：{question}
回答："""

PROMPT_TEMPLATE_ZH = """基于以下上下文信息，直接回答用户问题。只使用提供的信息，不要添加任何外部知识或格式化内容。

**极度重要：你的输出必须是纯粹、直接的回答，不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗）、或任何形式的元评论。请勿引用或复述Prompt内容。你的回答必须直接、简洁地结束，不带任何引导语或后续说明。**

示例1：
上下文：中国平安2023年第一季度实现营业收入2,345.67亿元，同比增长8.5%；净利润为156.78亿元，同比增长12.3%。
问题：中国平安的业绩如何？
回答：中国平安2023年第一季度业绩表现良好，营业收入同比增长8.5%至2,345.67亿元，净利润同比增长12.3%至156.78亿元。

示例2：
上下文：腾讯控股2023年上半年游戏业务收入同比下降5.2%，广告业务收入同比增长3.1%。
问题：腾讯的游戏业务表现如何？
回答：腾讯2023年上半年游戏业务收入同比下降5.2%，表现不佳。

上下文：
{context}

问题：{question}

回答："""

# Chain-of-Thought版本（优化版，隐藏思考过程）
PROMPT_TEMPLATE_ZH_COT = """你是一位专业的金融分析师。请基于以下上下文信息，通过内部思考来回答用户问题。

**极度重要：你的输出必须是纯粹、直接的回答，不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 boxed、数字列表、加粗）、或任何形式的元评论。请勿引用或复述Prompt内容。你的回答必须直接、简洁地结束，不带任何引导语或后续说明。**

重要要求：
1. 请进行内部思考，但不要输出任何思考步骤或过程
2. 直接给出最终答案，不要包含"思考"、"步骤"等词汇
3. 只使用提供的上下文信息，不要添加外部知识
4. 回答要简洁直接，用自然的中文表达

上下文：
{context}

问题：{question}

回答："""


class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retriever_top_k: int,
        prompt_template: Optional[str] = None, # No longer used, but kept for compatibility
        use_cot: bool = False,  # 是否使用Chain-of-Thought
        use_simple: bool = False,  # 是否使用超简洁模式
    ):
        self.retriever = retriever
        self.generator = generator
        # self.prompt_template is now obsolete
        self.retriever_top_k = retriever_top_k
        self.use_cot = use_cot
        self.use_simple = use_simple

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
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=user_input, top_k=self.retriever_top_k, return_scores=True
        )

        # 3. For Chinese queries, we should use multi-stage retrieval system
        # For now, we'll use a simple fallback for Chinese queries
        if is_chinese_q:
            no_context_message = "未找到合适的语料，请检查数据源。建议使用多阶段检索系统处理中文查询。"
        else:
            template_name = "rag_english_template"
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
        
        # 4. Create the final prompt using template
        try:
            if is_chinese_q:
                # 中文查询使用多阶段检索系统，这里只是回退
                prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{user_input}\n\n回答："
            else:
                # 英文查询使用模板
                prompt = template_loader.format_template(
                    template_name,
                    context=context_str, 
                    question=user_input
                )
                if prompt is None:
                    raise Exception("Template formatting failed")
        except Exception as e:
            # 回退到简单prompt
            if is_chinese_q:
                prompt = f"基于以下上下文回答问题：\n\n{context_str}\n\n问题：{user_input}\n\n回答："
            else:
                prompt = f"Context: {context_str}\nQuestion: {user_input}\nAnswer:"
        
        # 5. Generate the response
        try:
            generated_responses = self.generator.generate(texts=[prompt])
        except Exception as e:
            raise e
        
        # 6. Gather metadata
        retriever_model_name = ""
        # 检查retriever是否有encoder属性
        if hasattr(self.retriever, 'encoder_ch') and hasattr(self.retriever, 'encoder_en'):
            if is_chinese_q:
                retriever_model_name = getattr(self.retriever.encoder_ch, 'model_name', 'unknown')
            else:
                retriever_model_name = getattr(self.retriever.encoder_en, 'model_name', 'unknown')

        # 确定prompt模板类型
        if is_chinese_q:
            template_type = "ZH-MULTI-STAGE"
        else:
            template_type = "EN"

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

        result = RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=float_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=dict(
                retriever_model_name=retriever_model_name,
                top_k=self.retriever_top_k,
                generator_model_name=self.generator.model_name,
                prompt_template=f"Template-{template_type}",
                question_language="zh" if is_chinese_q else "en",
                use_cot=self.use_cot,
                use_simple=self.use_simple
            ),
        )
        return result
