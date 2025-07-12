#!/usr/bin/env python3
"""
生成模块性能评估脚本 - 对比 Fin-R1 和 Qwen3-8B 在中文数据集上的表现。
支持批量随机样本测试，并输出详细日志。
利用双 GPU 进行模型并行加载和评估以加速。
Prompt Template 内容从外部文件加载。
增加了 F1-score 和 Exact Match 的正确计算（使用jieba分词）。
统计了输入/输出 Token 数和纯生成时间。
优化了后处理逻辑以匹配更严格的Prompt Template，并针对Qwen3-8B进行定制。
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
import time
import re
import gc
import json
import argparse
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Dict, Any
from collections import Counter
import string
import jieba # 确保jieba库已导入并启用
import logging # 导入 logging 模块
import subprocess
import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed # 导入并行处理模块

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- 日志配置 ---
# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
# 使用当前时间戳作为文件名，确保每次运行的日志文件唯一
log_file_path = log_dir / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.log"

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 根 logger 设置为 DEBUG，以便文件记录所有信息

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 清除可能已存在的处理器，避免重复记录
# 这一步是为了防止在Jupyter Notebook或反复执行脚本时，日志处理器被重复添加
if logger.hasHandlers():
    logger.handlers.clear()

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # 控制台只输出 INFO 及以上
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 文件处理器
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG) # 文件中记录所有 DEBUG 及以上级别的日志
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 导入配置文件 (请确保 config/parameters.py 存在并定义了 config.generator.cache_dir)
try:
    from config.parameters import config
    logger.info(f"✅ 使用配置文件中的缓存路径: {config.generator.cache_dir}")
except ImportError:
    logger.warning("⚠️ 无法导入配置文件，使用默认缓存路径 '/users/sgjfei3/data/huggingface'")
    class Config: # 定义一个假的config类，防止报错
        class Generator:
            cache_dir = "/users/sgjfei3/data/huggingface"
        generator = Generator()
    config = Config()

# ====================================================================================
# 后处理模块定义 (专门针对中文)
# ====================================================================================

def _fix_company_name_translation(text: str) -> str:
    """
    修正公司名称翻译问题和年份问题 (仅限中文)。
    使用与RAG系统相同的复杂逻辑。
    """
    # 常见的公司名称翻译映射和不规范表达修正（中文 -> 中文标准）
    company_translations = {
        # 德赛电池相关 (确保匹配更宽泛，包括空格或不规范表达)
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*BATTERY\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
        r'德赛\s*BATTERY': '德赛电池',
        r'德赛\s*\(00\)': '德赛电池（000049）',
        r'德塞电池': '德赛电池', # 修正错别字

        # 产品名修正
        r'iPhone\s*\+\s*ProMax': 'iPhone 12 Pro Max',
        r'iPhon\s*e12ProMax': 'iPhone 12 Pro Max',
        r'iPhone\s*X\s*系列': 'iPhone 12 Pro Max',
        r'iPhone\s*1\s*\(Pro\s*Max\s*\)': 'iPhone 12 Pro Max',
        r'iPhone\s*1\s*Pro\s*Max': 'iPhone 12 Pro Max',
        r'iPhone\s*2\s*ProMax': 'iPhone 12 Pro Max',
    }
    for pattern, replacement in company_translations.items():
        text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)

    # 年份修正
    text = re.sub(r'20\s*\(\s*\d{2}\?\)\s*年度', r'2021年度', text, flags=re.IGNORECASE)
    text = text.replace('20XX年', '2021年')
    text = text.replace('20+', '2021')
    text = text.replace('2OI I年', '2021年')
    text = text.replace('20 I I年', '2021年')

    return text


def _clean_response_common(text: str) -> str:
    """
    通用的后处理模块：清除模型输出中的通用污染内容和格式噪音。
    适用于 Fin-R1 和 Qwen3-8B 移除思考过程后的文本。
    增加了数字和单位的统一处理。
    """
    original_text_for_debug = text # 用于调试，保留原始文本
    text = _fix_company_name_translation(text) # 调用公司名称和年份修正

    # 噪音模式，更具体的、更可能出现在Prompt开头或中间的噪音模式优先处理
    patterns_to_remove = [
        # 模型自我评论和通用引导语
        r'我需要检查这个回答是否符合要求.*?====', # 匹配从"我需要检查"到"===="
        r'\*\*注意\*\*:.*?改进后的版本[:：]', # 匹配"**注意**:"到"改进后的版本:"
        r'上面的答案虽然符合要求.*?以下是改进后的版本:', # 同上
        r'###\s*改进版答案', # 移除 ### 改进版答案 标题
        r'###\s*回答', # 移除 ### 回答 标题
        r'回答完成后立即停止生成', # 移除prompt的最后指令
        r'回答完成并停止', # 移除prompt的最后指令
        r'确保回答', # 移除prompt的最后指令
        r'用户可能', # 移除prompt的最后指令
        r'总结一下', # 移除prompt的最后指令
        r'请用简洁', # 移除prompt的最后指令
        r'进一步简化', # 移除prompt的最后指令
        r'再简化的版本', # 移除prompt的最后指令
        r'最终答案定稿如下', # 移除prompt的最后指令
        r'这个总结全面', # 移除核心点总结标题
        r'核心点总结[:：]?', # 移除核心点总结标题
        r'以上分析是否正确？还有哪些方面可以改进？', 
        r'您的分析基本合理，但在某些地方可以进一步完善和细化。以下是几点改进建议：',
        r'如有需要进一步细化某一方面的内容，请告知。',
        r'注意：以上论断完全依赖于已公开披露的信息资源 ; 对未来的具体前景尚需结合更多实时数据加以验证和完善', 
        r'（注意此段文字虽详细阐述了几方面因素及其相互作用机制，但由于题干要求高度浓缩为一句话内完成表述，故在此基础上进行了适当简化压缩）', 
        r'请注意，以上内容是对.*?展望，并非绝对结论。', 
        r'实际走势还需结合实际情况不断评估调整。希望这个回答对你有所帮助！', 
        r'要预测.*?做出判断[:：]?', 
        r'以下是几个关键因素和步骤[:：]?',
        r'综上所述[:：]?', 
        r'最终结论[:：]?',
        r'答案示例[:：]?',
        r'最终确认[:：]?',
        r'答案忠实地反映了原始文档的内容而无多余推断',
        r'回答[:：]\s*$', # 移除独立的"回答："或"回答："在行尾
        r'回答是：\s*', # 移除"回答是："
        r'以下是原因：\s*', # 移除"以下是原因："

        # ChatML和Markdown噪音
        r'<\|im_start\|>.*?<\|im_end\|>', # 移除所有ChatML的im_start/im_end标签
        r'\\boxed\{.*?\}', # 移除\boxed{}格式
        r'\\text\{.*?\}', # 移除LaTeX text格式
        r'\\s*', # 移除一些 LaTeX 相关的空白
        r'[\u2460-\u2469]\s*', # 移除带圈数字，如 ①

        # 清除Prompt中存在的结构性标记，如果它们意外出现在答案中
        r'===SYSTEM===[\s\S]*?===USER===', # 移除System部分
        r'---[\s\S]*?---', # 移除USER部分的---分隔符及其中间的所有内容（如果意外复制）
        r'【公司财务报告摘要】[\s\S]*?【完整公司财务报告片段】', # 移除摘要和片段标签
        r'【用户问题】[\s\S]*?【回答】', # 移除问题和回答标签

        r'Question:\n.*?\nTable Context:',
        r'Table Context:\n.*?\nText Context:',
        r'Text Context:\n.*?\nQuestion:',
        r'Context:\n.*?\nQuestion:',
        r'Assistant\'s Response:',
        r'\(参阅第三部分\)',
        r'\(详情见第②段\)',

        # Fin-R1常见冗余（新增或优化）
        r'根据提供的数据，', 
        r'此数据直接来源于.*?。', 
        r'这与摘要中的数据一致，无需进一步分析。', 
        r'根据提供的数据，这只股票的市盈率（TTM）在过去20个交易日中呈现先升后降的趋势。',
        r'结合近期股价波动和成交量情况，预计未来一个月内，受激励计划实施初期的正面情绪推动，股价可能继续小幅上扬，但需警惕费用增加和业绩压力带来的回调风险。综合评估，恒生电子的股票下个月收益存在不确定性，建议密切关注后续财报及市场反应。', 
        r'但是由于缺乏具体的技术指标分析和量化模型支持，无法精确预测下个月的具体涨跌情况及其概率。因此，根据现有信息，无法提供此项信息。', 
        r'综合来看，短期内股价可能面临一定调整，但长期仍看好其表现。未来一个月内，考虑到市场情绪和技术面因素，股价下跌的概率一般，建议投资者密切关注后续财报及行业动态。', 
        r'考虑到公司基本面改善潜力及市场情绪回暖的可能性，', 
        r'这与摘要信息一致。具体步骤如下：首先统计所有日期对应的换手率数值，然后求其算术平均值。经验证，各日期换手率之和除以天数等于1.2447，符合摘要描述。', 
        r'市场数据显示股价波动较大，但长期看好其业绩复苏潜力。综合来看，华电国际下个月的预期收益存在不确定性，需关注政策及行业动态。上涨概率取决于经济恢复速度及股市情绪等因素，建议持续跟踪。', 
        r'结合上下文，这可能意味着超过50%的可能性，但缺乏精确的数据支持。故根据现有信息，无法提供此项信息。', 
        r'基于现有信息，无法提供此项信息。', 
        r'\s*该值直接来源于报告中的.*?。', # 针对 "此值直接来源于报告中的"
        r'\s*因此，答案直接来源于这两份文件的共同确认。', # 针对 "因此，答案直接来源于这两份文件的共同确认。"
        r'具体来看，从2023年4月24日的43.2846逐步上升至4月27日的40.819，随后开始下滑，并于5月24日降至34.7638。整体表现为波动上升后回落。', # 顺丰控股特定冗余

        # 其他格式噪音（注意顺序，以免被前面的通用规则覆盖）
        r'^\s*[\d]+\.\s*', # 列表开头的数字序号或点，仅在行首
        r'^\s*[-*•·]\s*', # 列表开头的破折号或点，仅在行首
        r'^\s*\((\w|[一二三四五六七八九十])+\)\s*', # 列表开头的括号序号，仅在行首
        r'---', # 剩余的横线
        r'===', # 剩余的等号
        r'___', # 剩余的下划线
    ]

    for pattern in patterns_to_remove:
        original_length = len(text)
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE).strip()
        if len(text) != original_length:
            logger.debug(f"模式 '{pattern}' 移除了 {original_length - len(text)} 个字符")

    # 特殊处理需要替换的模式
    text = re.sub(r'(\d+)\.\s*(\d+)', r'\1.\2', text)  # 移除小数点后的空格
    text = re.sub(r'(\d+),\s*(\d{3})', r'\1\2', text)  # 移除千分位逗号后的空格
    text = re.sub(r'(?<=\d)\s*(?=%)', '', text)  # 移除数字和百分号之间的空格
    text = re.sub(r'(?<=\d)\s*(?=万|亿)', '', text)  # 移除数字和中文单位万/亿之间的空格
    text = re.sub(r'(?<=\d)\s*(?=元)', '', text)  # 移除数字和"元"之间的空格
    
    # 特殊处理加粗和斜体格式，保留内容但移除标记
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)  # 移除加粗标记但保留内容
    text = re.sub(r'\*(.*?)\*', r'\1', text, flags=re.DOTALL)      # 移除斜体标记但保留内容

    # 替换连续的换行符为单个空格，并移除多余空格
    text = re.sub(r'\n+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    # 移除末尾的逗号、分号，在最后处理以防误删
    text = re.sub(r'[，；,;]$', '', text).strip()

    # === 数字和单位的统一处理（新增或优化） ===
    text = text.replace('亿元', '亿') 
    text = text.replace('元。', '。') # 移除多余的“元”字如果紧跟句号
    text = text.replace('元', '') # 移除所有“元”字，如果期望参考答案中没有单位

    # 最后阶段的句子截断逻辑（保持，作为兜底）
    sentence_endings = r'(?<=[。？！；.])\s*'
    sentences = re.split(sentence_endings, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # *** 你已经决定移除句子数量的硬性截断，所以此处逻辑保持注释状态 ***
    # if len(sentences) > 5: # Prompt要求3-5句，可以考虑保留此逻辑
    #     logger.warning(f"⚠️ 文本句子数超过5句 ({len(sentences)}句)，进行强制截断：保留前5句。")
    #     sentences = sentences[:5]
    
    final_text = ' '.join(sentences)

    # 确保以句号结尾
    if final_text and not final_text.endswith(('.', '!', '?', '。', '！', '？')):
        final_text += '。'
    
    # 移除字符长度限制，允许完整回答
    logger.debug(f"📏 当前回答长度: {len(final_text)} 字符。")
    
    # 如果清理后为空，返回一个默认提示或原始响应的前N个字符作为兜底，防止空答案
    if not final_text.strip():
        logger.warning(f"⚠️ 清理后答案为空，返回原始响应前150字符作为兜底。")
        logger.debug(f"原始文本长度: {len(text)}, 清理后长度: {len(final_text)}")
        return text[:150].strip() + "..." if len(text) > 150 else text.strip()

    logger.debug(f"原始文本 (前500):\n{original_text_for_debug[:500]}...\n---清理后文本 (前500):\n{final_text[:500]}...")
    return final_text


def _clean_qwen3_8b_response_specific(raw_text: str) -> str:
    """
    专门为Qwen3-8B模型设计的后处理：更积极地移除其常见的<think>标签和思考过程。
    """
    logger.debug(f"🔍 Qwen3-8B 原始文本长度: {len(raw_text)}")
    
    # 检查是否包含<think>标签
    if '<think>' in raw_text:
        logger.debug(f"🔍 检测到<think>标签，开始清理...")
        
        # 首先尝试找到<think>标签的位置
        think_start = raw_text.find('<think>')
        think_end = raw_text.find('</think>')
        
        if think_start != -1:
            if think_end != -1 and think_end > think_start:
                # 有完整的<think>...</think>标签
                cleaned_text = raw_text[:think_start] + raw_text[think_end + 8:]  # 8 = len('</think>')
                logger.debug(f"🔍 移除完整<think>标签，长度从 {len(raw_text)} 变为 {len(cleaned_text)}")
            else:
                # 只有<think>标签，没有</think>
                cleaned_text = raw_text[:think_start]
                logger.debug(f"🔍 移除不完整<think>标签，长度从 {len(raw_text)} 变为 {len(cleaned_text)}")
        else:
            # 没有<think>标签，直接使用原始文本
            cleaned_text = raw_text
            logger.debug(f"🔍 未找到<think>标签，使用原始文本")
    else:
        logger.debug(f"🔍 未检测到<think>标签，直接使用原始文本")
        cleaned_text = raw_text
    
    logger.debug(f"🔍 移除<think>标签后长度: {len(cleaned_text)}")
    
    # 如果清理后为空，尝试更宽松的清理策略
    if not cleaned_text.strip():
        logger.warning(f"⚠️ 移除<think>标签后文本为空，尝试更宽松的清理策略")
        # 尝试移除常见的思考过程开头，但保留更多内容
        thought_patterns_loose = [
            r'好的，我现在需要回答用户关于.*?的问题。',
            r'好的，我现在需要处理用户关于.*?的问题。',
            r'首先，我要仔细阅读用户提供的资料，.*?确保答案准确且符合要求。',
            r'首先，查看提供的资料，用户给出了两个部分：摘要和完整报告片段。',
            r'首先，看用户的问题，他们想知道的是.*?。',
            r'接下来查看提供的资料。财务报告摘要里提到，',
            r'接着看完整片段，里面有一个字段是',
        ]
        for pattern in thought_patterns_loose:
            cleaned_text = re.sub(pattern, '', raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
            if cleaned_text.strip():
                break
    
    # 再次尝试移除常见的思考过程开头和引导语，针对<think>标签被截断或没有正确闭合后的残余
    # 这些模式应该放在移除 <think> 标签之后，针对残余的思考引导语
    thought_patterns_specific_to_qwen = [
        r'用户的问题非常直接，询问的是具体数值，所以答案应该直接引用这两个来源中的数值。',
        r'需要确认是否要引用具体日期，但用户的问题已经指定了日期，而两个数据源都对应这个日期。',
        r'因此，正确回答应该是.*?，不需要额外解释，因为用户只问数值。',
        r'确保不引入其他信息，比如分析或比较，只需直接回答即可。',
        r'确认没有其他信息冲突，但这里两个来源一致，因此正确无误。',
        r'确认没有其他隐藏的信息或者需要进一步计算的地方，因为用户的问题非常明确',
        r'接下来要组织语言，把波动的关键节点列出来。',
        r'现在要确认这两个数据是否一致。',
        r'需要把这些时间点和数值对应起来，确保准确无误。',
        r'需要注意是否有其他相关指标，比如.*?但用户问的是当天的.*?所以应该使用非.*?的数据。',
        r'无需进一步分析，因为两者均明确指出该日期的.*?。',
        r'用户的问题是询问这个具体数值，所以答案应该直接引用这两个来源中的数值。',
        r'在完整报告中，有明确的数值：',
        r'这两个数字对应的是用户询问的内容。',
        r'需要注意的是，用户要求直接回答，不需要额外解释，或只给出准确数值。', # 针对这种结尾
        r'接下来查看完整财务报告片段，里面', # Qwen3-8B的think-ahead
        r'首先看财务报告摘要，里面提到', # Qwen3-8B的think-ahead
        r'完整报告中提到，', # Qwen3-8B的think-ahead
        r'摘要里明确提到', # Qwen3-8B的think-ahead
    ]
    # 对 cleaned_text 进行第二轮清理，捕获更细致的思考引导语
    for pattern in thought_patterns_specific_to_qwen:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 对最终清理过的文本进行通用清理
    # 这里使用 _clean_response_common，而不是直接复制其所有规则，保持代码模块化
    final_cleaned_text = _clean_response_common(cleaned_text)
    
    logger.debug(f"🔍 Qwen3-8B 最终清理后长度: {len(final_cleaned_text)}")
    logger.debug(f"Qwen3-8B 原始文本 (前500):\n{raw_text[:500]}...\n---Qwen3-8B 清理后 (前500):\n{final_cleaned_text[:500]}...")
    return final_cleaned_text


# ====================================================================================
# Prompt 构造辅助函数 (从外部文件加载)
# ====================================================================================

def _load_template_content_from_file(template_file_name: str) -> str:
    """从指定文件中加载Prompt模板的完整字符串内容"""
    template_path = Path("data/prompt_templates") / template_file_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"❌ 模板文件未找到: {template_path}，请确保文件存在。")
        sys.exit(1)

def get_messages_for_test(summary: str, context: str, query: str, 
                          template_file_name: str = "multi_stage_chinese_template_with_fewshot.txt") -> List[Dict[str, str]]:
    """
    构建用于测试的 messages 列表，从指定模板文件加载内容，并将 item_instruction 融入 Prompt。
    Args:
        summary (str): LLM Qwen2-7B 生成的摘要。
        context (str): 完整上下文（已包含摘要）。
        query (str): 用户问题。
        template_file_name (str): 要加载的模板文件名。
    Returns:
        List[Dict[str, str]]: 构建好的 messages 列表。
    """
    template_full_string = _load_template_content_from_file(template_file_name)

    messages = []
    # 使用正则表达式分割所有部分，并保留分隔符内容
    parts = re.split(r'(===SYSTEM===|===USER===|===ASSISTANT===)', template_full_string, flags=re.DOTALL)

    # 移除第一个空字符串（如果存在）和多余的空白
    parts = [p.strip() for p in parts if p.strip()]

    current_role = None
    current_content = []

    for part in parts:
        if part in ["===SYSTEM===", "===USER===", "===ASSISTANT==="]:
            if current_role is not None:
                messages.append({"role": current_role.lower().replace("===", ""), "content": "\n".join(current_content).strip()})
            current_role = part
            current_content = []
        else:
            current_content.append(part)

    # 添加最后一个部分的 message
    if current_role is not None:
        messages.append({"role": current_role.lower().replace("===", ""), "content": "\n".join(current_content).strip()})

    # 替换占位符
    for message in messages:
        if message["role"] == "user":
            modified_content = message["content"]
            modified_content = modified_content.replace('{query}', query)
            modified_content = modified_content.replace('{summary}', summary)
            modified_content = modified_content.replace('{context}', context)
            message["content"] = modified_content

    logger.debug(f"构建的 messages: {messages}")
    return messages


def _convert_messages_to_chatml(messages: List[Dict[str, str]]) -> str:
    """
    将 messages 列表转换为 Fin-R1 (Qwen2.5 based) 期望的ChatML格式字符串。
    Qwen系列标准应该是 `im_end`
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
            # 这里的 assistant 角色通常是 Few-shot 示例的一部分
            formatted_prompt += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"

    # 在最后追加一个 <|im_start|>assistant\n，表示希望模型开始生成新的 assistant 回复
    formatted_prompt += "<|im_start|>assistant\n"

    logger.debug(f"转换后的 ChatML Prompt (前500字符):\n{formatted_prompt[:500]}...")
    return formatted_prompt


# ====================================================================================
# 模型加载和生成器包装类
# ====================================================================================

class ModelLoader:
    """负责加载和卸载模型，并提供生成接口"""
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device
        self.is_loaded = False

        cache_dir = config.generator.cache_dir

        if "Fin-R1" in model_name:
            local_fin_r1_path = f"{cache_dir}/models--SUFE-AIFLM-Lab--Fin-R1/snapshots/026768c4a015b591b54b240743edeac1de0970fa"
            if os.path.exists(local_fin_r1_path):
                self.model_path = local_fin_r1_path
                logger.info(f"✅ [{self.model_name}] 使用本地缓存模型: {self.model_path}")
            else:
                self.model_path = "SUFE-AIFLM-Lab/Fin-R1"
                logger.warning(f"⚠️ [{self.model_name}] 本地缓存未找到，将从Hub下载: {self.model_path}")
        elif "Qwen3-8B" in model_name:
            local_qwen_path = f"{cache_dir}/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
            if os.path.exists(local_qwen_path):
                self.model_path = local_qwen_path
                logger.info(f"✅ [{self.model_name}] 使用本地缓存模型: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-8B"
                logger.warning(f"⚠️ [{self.model_name}] 本地缓存未找到，将从Hub下载: {self.model_path}")
        else:
            self.model_path = model_name
            logger.warning(f"⚠️ [{self.model_name}] 模型路径 '{model_name}' 未知，尝试从Hugging Face Hub加载。建议提前下载到本地。")

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    def load_model(self):
        if self.is_loaded:
            logger.info(f"✅ [{self.model_name}] 已加载到 {self.device}，无需重复加载。")
            return

        logger.info(f"🔄 [{self.model_name}] 正在加载模型到 {self.device} 从 {self.model_path}")
        is_local_path = Path(self.model_path).exists() and Path(self.model_path).is_dir()

        cache_dir = config.generator.cache_dir
        tokenizer_args = {"trust_remote_code": True, "local_files_only": is_local_path, "cache_dir": cache_dir}
        model_args = {
            "torch_dtype": torch.float16,
            "device_map": self.device,  # 明确指定设备
            "trust_remote_code": True,
            "quantization_config": self.quantization_config,
            "local_files_only": is_local_path,
            "cache_dir": cache_dir
        }

        try:
            logger.info(f"🔧 [{self.model_name}] 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_args)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"✅ [{self.model_name}] Tokenizer加载完成. Chat Template: {self.tokenizer.chat_template}")

            logger.info(f"🔧 [{self.model_name}] 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_args)
            self.model.eval()
            logger.info(f"✅ [{self.model_name}] 模型加载完成. 设备: {self.model.device.type}:{self.model.device.index}, 量化: 4bit")
            self.is_loaded = True
        except Exception as e:
            logger.exception(f"❌ [{self.model_name}] 模型加载失败: {e}")
            self.unload_model()
            raise

    def unload_model(self):
        if not self.is_loaded:
            return

        logger.info(f"🗑️ [{self.model_name}] 卸载模型并清理显存...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            self.is_loaded = False
            logger.info(f"✅ [{self.model_name}] 显存已清理。")
        except Exception as e:
            logger.error(f"❌ 卸载 [{self.model_name}] 时发生错误: {e}")

    def generate(self, prompt_string: str, max_new_tokens: int = 512, do_sample: bool = False, repetition_penalty: float = 1.1) -> Dict[str, Any]: # 提高max_new_tokens默认值
        """
        生成文本，期望输入已经是 ChatML 格式的字符串。
        返回包含生成文本、输入和输出token数的字典。
        """
        if not self.is_loaded:
            raise RuntimeError(f"模型 {self.model_name} 未加载。请先调用 load_model()。")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError(f"模型 {self.model_name} 的tokenizer或model为None。请重新加载模型。")

        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        start_gen_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
        end_gen_time = time.time()

        generated_tokens_ids = outputs[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)

        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = generated_tokens_ids.shape[0]

        logger.debug(f"[{self.model_name}] 输入tokens: {input_token_count}, 输出tokens: {output_token_count}, 生成时间: {end_gen_time - start_gen_time:.2f}s")
        return {
            "generated_text": generated_text,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "generation_time_pure": end_gen_time - start_gen_time
        }

# ====================================================================================
# 评估指标计算 (基于词重叠，针对中文使用jieba)
# ====================================================================================

def normalize_answer_chinese(s: str) -> str:
    """
    针对中文进行答案归一化：移除标点、转换全角字符为半角、去除多余空格、分词并小写。
    使用jieba进行中文分词，获得更准确的F1和EM评估。
    """
    if not s:
        return ""

    s = s.strip().lower()

    s = s.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?').replace('；', ';')
    s = s.replace('（', '(').replace('）', ')')

    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~“”‘’【】『』《》—…·～「」～￥%#@！&（）《》]'
    s = re.sub(punctuation_pattern, '', s)

    # 关键修改：使用jieba进行分词
    tokens = list(jieba.cut(s)) 

    normalized_tokens = [token for token in tokens if token.strip()]
    return " ".join(normalized_tokens)


def get_tokens_chinese(s: str) -> List[str]:
    """获取中文分词后的tokens列表。"""
    return normalize_answer_chinese(s).split()

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """计算F1分数 (基于词重叠)。"""
    gold_tokens = get_tokens_chinese(ground_truth)
    pred_tokens = get_tokens_chinese(prediction)

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配率。"""
    return float(normalize_answer_chinese(prediction) == normalize_answer_chinese(ground_truth))

# ====================================================================================
# 主测试逻辑
# ====================================================================================

def run_chinese_comparison_test(args):
    logger.info("🚀 中文模型对比测试开始...")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        logger.warning(f"❌ 警告：检测到 {num_gpus} 块可用 GPU (少于 2 块)。将退化为单卡顺序评估模式。")
        model_configs = [
            ("Fin-R1", "cuda:0"),
            ("Qwen3-8B", "cuda:0") # 都会加载到 cuda:0，但会按顺序加载和卸载
        ]
        single_gpu_sequential_mode = True
    else:
        logger.info(f"✅ 检测到 {num_gpus} 块 GPU。尝试分配 Fin-R1 到 cuda:0，Qwen3-8B 到 cuda:1。")
        model_configs = [
            ("Fin-R1", "cuda:0"),
            ("Qwen3-8B", "cuda:1")
        ]
        single_gpu_sequential_mode = False

    model_loaders = {}
    for name, dev in model_configs:
        model_loaders[name] = ModelLoader(name, dev)

    data_path = args.data_path
    sample_size = args.sample_size
    template_file_name = "multi_stage_chinese_template_with_fewshot.txt" # <-- 确保这里指向你新的Few-shot模板

    logger.info(f"📊 加载数据集: {data_path}")
    try:
        from utils.data_loader import load_json_or_jsonl, sample_data
        dataset_full = load_json_or_jsonl(data_path)

        # --- 数据筛选逻辑 ---
        filtered_dataset = []
        # 定义你关心的预测类答案模式
        # 注意：这里我们放宽了对标点的匹配，以增加鲁棒性，但仍确保核心格式
        target_prediction_pattern = re.compile(r"这个股票的下月最终收益结果是:['‘’](涨|跌)['’],?(上涨|下跌)概率:(极大|较大|中上|一般)[。.]?")

        for item in dataset_full:
            instruction_content = item.get("instruction", "").strip()
            answer_content = item.get("answer", "").strip()
            
            # 条件1: instruction 为空 (通用问答，如数值抽取、摘要等)
            # 条件2: instruction 不为空，并且 answer 匹配特定预测模式 (预测类问答)
            # 这种筛选方式旨在保留所有有意义的评估样本
            if not instruction_content or target_prediction_pattern.fullmatch(answer_content):
                filtered_dataset.append(item)
            else:
                logger.debug(f"跳过样本 (instruction存在但answer不匹配预测模式): Query='{item.get('query', '')[:50]}...', Answer='{answer_content[:50]}...'")
        
        if sample_size > 0:
            dataset_to_evaluate = sample_data(filtered_dataset, sample_size, 42)
            logger.info(f"✅ 筛选并随机采样 {len(dataset_to_evaluate)} 个样本进行评估。")
        else:
            dataset_to_evaluate = filtered_dataset
            logger.info(f"✅ 筛选后，加载了全部 {len(dataset_to_evaluate)} 个样本进行评估。")

        if not dataset_to_evaluate:
            logger.error("❌ 没有符合条件的样本进行评估，请检查数据集和筛选条件。")
            return

    except Exception as e:
        logger.exception(f"❌ 数据集加载或筛选过程中发生错误: {e}")
        return

    all_results_data = []

    if single_gpu_sequential_mode:
        logger.info("\n--- 进入单 GPU 顺序评估模式 ---")
        for model_name, loader in model_loaders.items():
            try:
                logger.info(f"\n🔄 正在加载模型: {model_name} 到 {loader.device}")
                loader.load_model()
                logger.info(f"✅ 模型 {model_name} 加载完成，开始评估...")
                
                model_specific_results = evaluate_model_on_dataset(
                    model_name, loader, dataset_to_evaluate, template_file_name,
                    args.max_new_tokens, args.do_sample, args.repetition_penalty, args.save_frequency
                )
                all_results_data.extend(model_specific_results)
                logger.info(f"\n--- {model_name} 评估完成 ---")
            except Exception as e:
                logger.exception(f"❌ 模型 {model_name} 评估过程中发生错误: {e}")
            finally:
                loader.unload_model()
                logger.info(f"✅ 模型 {model_name} 卸载完成")
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        logger.info(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
    else: # 双 GPU 并行模式
        logger.info("\n🔄 开始顺序加载模型，然后并行评估...")
        
        # 顺序加载模型以避免设备冲突
        loaded_models = {}
        for model_name, loader in model_loaders.items():
            try:
                logger.info(f"🔄 正在加载 {model_name} 到 {loader.device}...")
                loader.load_model()
                loaded_models[model_name] = loader
                logger.info(f"✅ {model_name} 加载完成，设备: {loader.device}")
            except Exception as e:
                logger.exception(f"❌ 模型 {model_name} 加载失败: {e}")
                loader.unload_model()

        if not loaded_models:
            logger.error("❌ 没有模型成功加载，退出评估。")
            return

        logger.info(f"\n✅ 成功加载 {len(loaded_models)} 个模型，开始并行评估...")
        logger.info(f"📊 当前GPU内存状态:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

        # 并行评估 - 使用线程池实现伪并行（原始实现）
        logger.info(f"\n🚀 启动 {len(loaded_models)} 个模型的并行评估...")
        logger.info(f"📊 并行模型列表: {list(loaded_models.keys())}")
        
        # 准备所有任务
        tasks = []
        for model_name, loader in loaded_models.items():
            logger.info(f"✅ 准备 {model_name} 评估任务...")
            tasks.append((model_name, loader, dataset_to_evaluate, template_file_name, args.max_new_tokens, args.do_sample, args.repetition_penalty, args.save_frequency))
        
        # 使用线程池同时执行所有任务
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # 同时提交所有任务
            futures = {executor.submit(evaluate_model_on_dataset, *task): task[0] for task in tasks}
            
            # 等待所有任务完成
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    model_specific_results = future.result()
                    all_results_data.extend(model_specific_results)
                    logger.info(f"\n🎉 {model_name} 评估完成！")
                except Exception as e:
                    logger.exception(f"❌ 模型 {model_name} 评估过程中发生错误: {e}")
                finally:
                    if model_name in loaded_models:
                        loaded_models[model_name].unload_model()
        
        logger.info(f"\n📊 并行模式评估后，最终GPU内存状态:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"   GPU {i}: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

    # --- 评估完成，保存所有结果 ---
    output_filename = f"comparison_results_chinese_{os.path.basename(data_path).replace('.jsonl', '')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_data, f, ensure_ascii=False, indent=4)
    logger.info(f"\n🎉 评估完成！详细结果已保存到: {output_filename}")

    # 汇总并打印最终对比结果
    logger.info("\n--- 最终模型对比摘要 ---")
    model_summaries = {}
    for result in all_results_data:
        model_name = result["model"]
        if model_name not in model_summaries:
            model_summaries[model_name] = {
                "total_f1": 0.0,
                "total_em": 0.0,
                "total_gen_time": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_smart_context_length": 0,
                "total_summary_length": 0,
                "total_original_context_length": 0,
                "count": 0
            }

        model_summaries[model_name]["total_f1"] += result["f1_score"]
        model_summaries[model_name]["total_em"] += result["exact_match"]
        model_summaries[model_name]["total_gen_time"] += result["generation_time_pure"]
        model_summaries[model_name]["total_input_tokens"] += result["input_token_count"]
        model_summaries[model_name]["total_output_tokens"] += result["output_token_count"]
        model_summaries[model_name]["total_smart_context_length"] += result.get("smart_context_length", 0)
        model_summaries[model_name]["total_summary_length"] += result.get("summary_length", 0)
        model_summaries[model_name]["total_original_context_length"] += result.get("original_context_length", 0)
        model_summaries[model_name]["count"] += 1

    for model_name, data in model_summaries.items():
        if data["count"] > 0:
            avg_f1 = data["total_f1"] / data["count"]
            avg_em = data["total_em"] / data["count"]
            avg_gen_time = data["total_gen_time"] / data["count"]
            avg_input_tokens = data["total_input_tokens"] / data["count"]
            avg_output_tokens = data["total_output_tokens"] / data["count"]
            avg_smart_context_length = data["total_smart_context_length"] / data["count"]
            avg_summary_length = data["total_summary_length"] / data["count"]
            avg_original_context_length = data["total_original_context_length"] / data["count"]
        else:
            avg_f1, avg_em, avg_gen_time, avg_input_tokens, avg_output_tokens = 0.0, 0.0, 0.0, 0.0, 0.0
            avg_smart_context_length, avg_summary_length, avg_original_context_length = 0.0, 0.0, 0.0

        logger.info(f"\n模型: {model_name}")
        logger.info(f"  评估样本数: {data['count']}")
        logger.info(f"  平均 F1-score: {avg_f1:.4f}")
        logger.info(f"  平均 Exact Match: {avg_em:.4f}")
        logger.info(f"  平均生成耗时 (纯推理): {avg_gen_time:.2f} 秒/样本")
        logger.info(f"  平均输入 Token 数: {avg_input_tokens:.1f}")
        logger.info(f"  平均输出 Token 数: {avg_output_tokens:.1f}")
        logger.info(f"  平均智能Context长度: {avg_smart_context_length:.1f} 字符")
        logger.info(f"  平均Summary长度: {avg_summary_length:.1f} 字符")
        logger.info(f"  平均原始Context长度: {avg_original_context_length:.1f} 字符")
    logger.info("----------------------------")


def evaluate_model_on_dataset(model_name: str, loader: ModelLoader, dataset: List[Dict[str, Any]], template_file_name: str, max_new_tokens: int, do_sample: bool, repetition_penalty: float, save_frequency: int = 10) -> List[Dict[str, Any]]:
    """
    在特定数据集上评估单个模型。此函数将在独立的线程中运行。
    """
    model_results = []

    logger.info(f"\n[线程] 开始评估 {model_name} 在 {loader.device} 上...")

    pbar = tqdm(dataset, desc=f"评估 {model_name} ({loader.device})")
    for i, item in enumerate(pbar):
        # 基础查询
        query = item.get("generated_question", "") or item.get("query", "") or item.get("question", "")
        # 最终查询 = 基础查询 + 指令
        item_instruction = item.get("instruction", "").strip()
        final_query = item_instruction if item_instruction else query
        
        summary = item.get("summary", "")
        context = item.get("context", "")
        expected_answer = item.get("answer", "")
        original_context_length = len(item.get("context", "")) # 获取原始context长度

        context_for_prompt = build_smart_context(summary, context, final_query)
        messages = get_messages_for_test(summary, context_for_prompt, final_query, template_file_name) 
        prompt_string_for_model = _convert_messages_to_chatml(messages)

        try:
            gen_output = loader.generate(
                prompt_string=prompt_string_for_model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty
            )
            generated_text = gen_output["generated_text"]
            
            # 根据模型名称选择不同的后处理函数
            if model_name == "Qwen3-8B":
                final_answer = _clean_qwen3_8b_response_specific(generated_text)
            else: # 默认为 Fin-R1 或其他模型
                final_answer = _clean_response_common(generated_text) 

            f1 = calculate_f1_score(final_answer, expected_answer)
            em = calculate_exact_match(final_answer, expected_answer)

            model_results.append({
                "model": model_name,
                "sample_id": i,
                "query": query,
                "item_instruction": item_instruction,
                "final_query": final_query,
                "expected_answer": expected_answer,
                "raw_generated_text": generated_text,
                "final_answer": final_answer,
                "f1_score": f1,
                "exact_match": em,
                "generation_time_pure": gen_output["generation_time_pure"],
                "input_token_count": gen_output["input_token_count"],
                "output_token_count": gen_output["output_token_count"],
                "smart_context_length": len(context_for_prompt),
                "summary_length": len(summary),
                "original_context_length": original_context_length
            })
        except Exception as e:
            logger.exception(f"❌ [线程] {model_name} 样本 {i} 评估失败: {e}")
            model_results.append({
                "model": model_name,
                "sample_id": i,
                "query": query,
                "item_instruction": item_instruction,
                "final_query": final_query,
                "expected_answer": expected_answer,
                "raw_generated_text": "[ERROR]",
                "final_answer": "[ERROR]",
                "f1_score": 0.0,
                "exact_match": 0.0,
                "generation_time_pure": 0.0,
                "input_token_count": 0,
                "output_token_count": 0,
                "smart_context_length": len(context_for_prompt),
                "summary_length": len(summary),
                "original_context_length": original_context_length,
                "error": str(e)
            })
        # 每处理指定条数自动保存一次
        if (i + 1) % save_frequency == 0 or (i + 1) == len(dataset):
            partial_file = f"partial_results_{model_name}.json"
            try:
                with open(partial_file, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=4)
                logger.info(f"✅ 已自动保存前 {i+1} 条结果到 {partial_file}")
            except Exception as e:
                logger.warning(f"⚠️ 自动保存分批结果失败: {e}")
    return model_results


def build_smart_context(summary: str, context: str, query: str) -> str:
    """
    智能构建context，目前主要是对原始context进行可能的格式处理和长度限制，以便更好地填充到Prompt。
    这个函数负责将原始的 `context` 字符串进行处理（例如，如果是字典字符串则格式化，并进行长度截断）。
    这个处理后的 `context` 将被填充到 Prompt Template 中的 `{context}` 占位符。
    """
    processed_context = context
    try:
        # 尝试将 context 解析为字典，如果是则格式化为可读的JSON
        # 注意：这里使用 json.loads() 代替 eval() 更安全，但需要先替换单引号为双引号
        context_data = json.loads(context.replace("'", '"')) 
        if isinstance(context_data, dict):
            processed_context = json.dumps(context_data, ensure_ascii=False, indent=2)
            logger.debug("✅ Context识别为字典字符串并已格式化为JSON。")
    except (json.JSONDecodeError, TypeError):
        logger.debug("⚠️ Context非JSON字符串格式，直接使用原始context。")
        pass

    # 这里的长度限制是针对被处理后的context
    max_processed_context_length = 3500 # 字符长度，作为粗略限制
    if len(processed_context) > max_processed_context_length:
        logger.warning(f"⚠️ 处理后的Context长度过长 ({len(processed_context)}字符)，进行截断。")
        processed_context = processed_context[:max_processed_context_length] + "..."

    return processed_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文模型对比测试脚本")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据集文件路径 (jsonl格式，例如 data/alphafin/alphafin_eval_samples_updated.jsonl)")
    parser.add_argument("--sample_size", type=int, default=100, help="随机采样的样本数量 (0表示评估全部，默认为100)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="模型生成最大新Token数") # 提高默认值
    parser.add_argument("--do_sample", action='store_true', help="是否使用采样生成 (如果设置了此flag，则为True，默认False)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数")
    parser.add_argument("--save_frequency", type=int, default=10, help="自动保存频率 (每处理多少条数据保存一次，默认为10)")

    args = parser.parse_args()
    run_chinese_comparison_test(args)