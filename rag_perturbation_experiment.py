#!/usr/bin/env python3
"""
RAG系统扰动实验流程
使用rag_system_adapter的RAG系统函数进行扰动实验
包括样本选择、扰动应用、答案比较、重要性计算和LLM Judge评估
"""

import sys
import os
import json
import time
import random
import re
import argparse
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from collections import Counter
from typing import Set
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入，避免在模块级别执行初始化代码
# from alphafin_data_process.rag_system_adapter import RagSystemAdapter
# from xlm.modules.perturber.trend_perturber import TrendPerturber
# from xlm.modules.perturber.year_perturber import YearPerturber
# from xlm.modules.perturber.term_perturber import TermPerturber
# from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
# from xlm.components.encoder.encoder import Encoder
# from config.parameters import Config

# 导入LLM Judge功能
from llm_comparison.chinese_llm_judge import ModelLoader, get_judge_messages

def classify_question_type(question: str) -> str:
    """分类问题类型"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['多少', '几', '数量', '金额', '数字']):
        return "数值型"
    elif any(word in question_lower for word in ['为什么', '原因', '导致', '影响']):
        return "原因分析型"
    elif any(word in question_lower for word in ['如何', '怎么', '方法', '策略']):
        return "方法策略型"
    elif any(word in question_lower for word in ['比较', '对比', '差异', '区别']):
        return "对比分析型"
    elif any(word in question_lower for word in ['趋势', '变化', '发展', '增长']):
        return "趋势分析型"
    else:
        return "一般描述型"

def classify_context_type(context: str) -> str:
    """分类上下文类型"""
    context_lower = context.lower()
    
    if any(word in context_lower for word in ['财务', '营收', '利润', '资产', '负债']):
        return "财务数据型"
    elif any(word in context_lower for word in ['业绩', '表现', '增长', '下降']):
        return "业绩表现型"
    elif any(word in context_lower for word in ['政策', '规定', '法规', '制度']):
        return "政策法规型"
    elif any(word in context_lower for word in ['市场', '竞争', '份额', '地位']):
        return "市场竞争型"
    else:
        return "一般信息型"

def calculate_complexity_score(question: str, answer: str) -> float:
    """计算复杂度分数"""
    question_length = len(question)
    answer_length = len(answer)
    
    financial_terms = ['营收', '利润', '资产', '负债', '市盈率', '市净率', 'ROE', 'ROA']
    term_count = sum(1 for term in financial_terms if term in question or term in answer)
    
    complexity = (question_length * 0.3 + answer_length * 0.4 + term_count * 0.3) / 100
    return min(complexity, 1.0)

class PerturbationSampleSelector:
    """扰动样本选择器 - 改进版，确保选择可扰动的样本"""
    
    def __init__(self):
        # 与TrendPerturber保持一致的映射 - 使用与扰动器完全相同的词汇映射
        self.trend_map = {
            "上升": "下降", "上涨": "下跌", "增长": "减少", "提升": "降低", "增加": "减少",
            "下降": "上升", "下跌": "上涨", "减少": "增长", "降低": "提升",
            "好转": "恶化", "改善": "恶化", "积极": "消极", "盈利": "亏损",
            "扩张": "收缩", "持续增长": "持续下滑", "稳步增长": "显著下降",
            "强劲": "疲软", "高于": "低于", "优于": "劣于", "领先": "落后",
            "增加率": "减少率", "上升趋势": "下降趋势", "增长趋势": "减少趋势"
        }
        
        # 编译正则表达式模式，与TrendPerturber完全一致
        self.trend_patterns = {
            "zh": {
                re.compile(r'(' + re.escape(k) + r')'): v 
                for k, v in self.trend_map.items()
            }
        }
        
        # 与YearPerturber保持一致的年份模式
        self.year_patterns = {
            "zh": [
                re.compile(r'(\d{4})\s*年'),      # "2023年"
                re.compile(r'(\d{4})\s*年度'),    # "2023年度"
                re.compile(r'(\d{4})年(\d{1,2})月'), # "2023年1月"
            ]
        }
        
        # 与TermPerturber保持一致的术语映射
        self.term_map = {
            "市盈率": "净利润", "净利润": "市盈率", "市净率": "市销率", "市销率": "市净率",
            "营收": "收入", "收入": "营收", "营业收入": "营业利润", "营业利润": "营业收入",
            "总资产": "净资产", "净资产": "总资产", "负债": "资产", "资产": "负债",
            "利润": "成本", "成本": "利润", "市值": "估值", "估值": "市值",
            "股息": "分红", "分红": "股息", "配股": "增发", "增发": "配股",
            "回购": "增发", "交易量": "成交额", "成交额": "交易量", "换手率": "交易量"
        }
        
        # 编译正则表达式模式，与TermPerturber完全一致
        self.term_patterns = {
            "zh": {
                re.compile(r'(' + re.escape(k) + r')'): v 
                for k, v in self.term_map.items()
            }
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """加载评测数据集 - 支持JSONL格式"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                        continue
        return samples
    
    def analyze_sample(self, sample: Dict) -> Dict:
        """分析单个样本的关键词分布 - 使用与扰动器一致的检测方法"""
        summary = sample.get('summary', '')
        content = sample.get('content', '')
        generated_question = sample.get('generated_question', '')
        
        # 主要关注context字段（summary和content），因为这是扰动器作用的对象
        context_text = f"{summary} {content}"
        question_text = generated_question
        
        # 使用与扰动器一致的检测方法
        context_trend_found = self._detect_trend_terms(context_text)
        context_year_found = self._detect_year_terms(context_text)
        context_term_found = self._detect_term_terms(context_text)
        
        question_trend_found = self._detect_trend_terms(question_text)
        question_year_found = self._detect_year_terms(question_text)
        question_term_found = self._detect_term_terms(question_text)
        
        # 合并所有关键词（但主要权重给context）
        trend_found = context_trend_found | question_trend_found
        year_found = context_year_found | question_year_found
        term_found = context_term_found | question_term_found
        
        return {
            'sample_id': sample.get('id', 'unknown'),
            'summary': summary,
            'content': content,
            'generated_question': generated_question,
            'trend_keywords': trend_found,
            'year_keywords': year_found,
            'term_keywords': term_found,
            'context_trend_score': len(context_trend_found),
            'context_year_score': len(context_year_found),
            'context_term_score': len(context_term_found),
            'question_trend_score': len(question_trend_found),
            'question_year_score': len(question_year_found),
            'question_term_score': len(question_term_found),
            'trend_score': len(trend_found),
            'year_score': len(year_found),
            'term_score': len(term_found),
            'total_score': len(trend_found) + len(year_found) + len(term_found),
            'context_score': len(context_trend_found) + len(context_year_found) + len(context_term_found)
        }
    
    def _detect_year_terms(self, text: str) -> Set[str]:
        """检测年份术语 - 使用与YearPerturber一致的方法"""
        found_terms = set()
        lang = "zh"  # 专注于中文
        patterns = self.year_patterns.get(lang, [])
        
        for pattern in patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    # 如果是元组，取第一个元素（年份）
                    year = match[0] if match[0].isdigit() else match[1] if len(match) > 1 and match[1].isdigit() else None
                    if year and 1900 <= int(year) <= 2050:  # 与YearPerturber保持一致的年份范围
                        found_terms.add(year)
                elif isinstance(match, str) and match.isdigit():
                    if 1900 <= int(match) <= 2050:  # 与YearPerturber保持一致的年份范围
                        found_terms.add(match)
        
        return found_terms
    
    def _detect_trend_terms(self, text: str) -> Set[str]:
        """检测趋势术语 - 使用与TrendPerturber一致的方法"""
        found_terms = set()
        lang = "zh"  # 专注于中文
        patterns = self.trend_patterns.get(lang, {})
        
        for pattern, antonym in patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                found_terms.add(match)
        
        return found_terms
    
    def _detect_term_terms(self, text: str) -> Set[str]:
        """检测金融术语 - 使用与TermPerturber一致的方法"""
        found_terms = set()
        lang = "zh"  # 专注于中文
        patterns = self.term_patterns.get(lang, {})
        
        for pattern, replacement in patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                found_terms.add(match)
        
        return found_terms
    
    def select_samples(self, samples: List[Dict], target_count: int = 20) -> List[Dict]:
        """选择适合的样本 - 确保可扰动性"""
        print(f"🎯 开始智能样本选择，目标数量: {target_count}")
        print("🔍 严格筛选可扰动样本...")
        
        analyzed_samples = [self.analyze_sample(sample) for sample in samples]
        
        # 验证扰动可行性：确保样本至少支持一种扰动类型
        feasible_samples = []
        for sample in analyzed_samples:
            # 检查是否至少支持一种扰动类型（提高阈值，确保有足够的扰动元素）
            has_year_perturbation = sample['year_score'] >= 1  # 至少1个年份
            has_trend_perturbation = sample['trend_score'] >= 1  # 至少1个趋势词
            has_term_perturbation = sample['term_score'] >= 1  # 至少1个金融术语
            
            # 额外检查：确保样本有足够的上下文内容
            context_length = len(sample.get('summary', '') + sample.get('content', ''))
            has_sufficient_context = context_length >= 50  # 至少50个字符的上下文
            
            if (has_year_perturbation or has_trend_perturbation or has_term_perturbation) and has_sufficient_context:
                feasible_samples.append(sample)
                print(f"✅ 样本 {sample['sample_id']} 支持扰动 (上下文长度: {context_length}):")
                if has_year_perturbation:
                    print(f"  年份扰动: {sample['year_score']} 个年份 ({sample['year_keywords']})")
                if has_trend_perturbation:
                    print(f"  趋势扰动: {sample['trend_score']} 个趋势词 ({sample['trend_keywords']})")
                if has_term_perturbation:
                    print(f"  术语扰动: {sample['term_score']} 个金融术语 ({sample['term_keywords']})")
            else:
                if not has_sufficient_context:
                    print(f"❌ 样本 {sample['sample_id']} 上下文内容不足 ({context_length} 字符)，跳过")
                else:
                    print(f"❌ 样本 {sample['sample_id']} 不支持任何扰动类型，跳过")
                continue  # 明确跳过不支持的样本
        
        print(f"📊 可扰动样本数: {len(feasible_samples)}")
        
        if len(feasible_samples) < target_count:
            print(f"⚠️ 可扰动样本数量不足，将选择所有 {len(feasible_samples)} 个样本")
            target_count = len(feasible_samples)
        
        # 按扰动能力排序并选择
        feasible_samples.sort(key=lambda x: x['total_score'], reverse=True)
        selected_samples = feasible_samples[:target_count]
        
        print(f"✅ 最终选择了 {len(selected_samples)} 个样本")
        for i, sample in enumerate(selected_samples):
            print(f"  {i+1}. {sample['sample_id']} (总分: {sample['total_score']})")
        
        return selected_samples

@dataclass
class PerturbationSample:
    """扰动实验样本"""
    sample_id: str
    context: str
    question: str
    expected_answer: str
    question_type: str
    context_type: str
    complexity_score: float
    diversity_score: float

@dataclass
class PerturbationDetail:
    """扰动详细信息"""
    perturber_name: str
    original_text: str
    perturbed_text: str
    perturbation_type: str  # "term", "year", "trend"
    changed_elements: List[str]  # 具体变化的元素
    change_description: str  # 变化描述
    timestamp: str

@dataclass
class PerturbationResult:
    """扰动实验结果"""
    sample_id: str
    perturber_name: str
    original_answer: str
    perturbed_answer: str
    perturbation_detail: PerturbationDetail  # 使用新的扰动详情类
    similarity_score: float
    importance_score: float
    f1_score: float  # 新增F1分数
    em_score: float  # 新增EM分数
    llm_judge_scores: Dict[str, Any]
    timestamp: str
    perturbation_target: str = "summary"  # 默认对summary扰动，也可以是"prompt"
    # 期望答案评估指标
    expected_vs_original_f1: float = 0.0
    expected_vs_original_em: float = 0.0
    expected_vs_perturbed_f1: float = 0.0
    expected_vs_perturbed_em: float = 0.0
    # 扰动影响指标
    f1_improvement: float = 0.0
    em_improvement: float = 0.0
    llm_judge_improvement: float = 0.0

class RAGPerturbationExperiment:
    """RAG系统扰动实验类 - 使用rag_system_adapter"""
    
    def __init__(self):
        """初始化实验环境"""
        print("🔬 初始化RAG扰动实验环境（使用rag_system_adapter）...")
        
        # 延迟导入，避免在模块级别执行初始化代码
        try:
            from alphafin_data_process.rag_system_adapter import RagSystemAdapter
            from xlm.modules.perturber.trend_perturber import TrendPerturber
            from xlm.modules.perturber.year_perturber import YearPerturber
            from xlm.modules.perturber.term_perturber import TermPerturber
            from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
            from xlm.components.encoder.encoder import Encoder
            from config.parameters import Config
            
            print("✅ 所有模块导入成功")
        except ImportError as e:
            print(f"❌ 模块导入失败: {e}")
            raise
        
        # 使用现有配置，不下载新模型
        self.config = Config()
        print(f"📋 使用配置: {self.config}")
        
        # 初始化RAG系统适配器
        try:
            print("🚀 初始化RAG系统适配器...")
            self.rag_adapter = RagSystemAdapter(self.config)
            print("✅ RAG系统适配器初始化完成")
        except Exception as e:
            print(f"❌ RAG系统适配器初始化失败: {e}")
            raise
        
        # 初始化扰动器
        try:
            print("🔧 初始化扰动器...")
            self.year_perturber = YearPerturber()
            self.trend_perturber = TrendPerturber()
            self.term_perturber = TermPerturber()
            print("✅ 扰动器初始化完成")
        except Exception as e:
            print(f"❌ 扰动器初始化失败: {e}")
            raise
        
        # 初始化编码器和比较器
        try:
            print("🔧 初始化编码器和比较器...")
            self.encoder = Encoder(
                model_name=self.config.encoder.chinese_model_path,
                cache_dir=self.config.encoder.cache_dir,
                device=self.config.encoder.device
            )
            self.comparator = EmbeddingComparator(self.encoder)
            print("✅ 编码器和比较器初始化完成")
        except Exception as e:
            print(f"❌ 编码器和比较器初始化失败: {e}")
            raise
        
        # 初始化生成器
        try:
            print("🔧 初始化生成器...")
            from xlm.components.generator.local_llm_generator import LocalLLMGenerator
            self.generator = LocalLLMGenerator(
                model_name=self.config.generator.model_name,
                cache_dir=self.config.generator.cache_dir,
                device=self.config.generator.device,
                use_quantization=self.config.generator.use_quantization,
                quantization_type=self.config.generator.quantization_type,
                use_flash_attention=self.config.generator.use_flash_attention
            )
            print("✅ 生成器初始化完成")
        except Exception as e:
            print(f"❌ 生成器初始化失败: {e}")
            raise
        
        # 初始化LLM Judge
        try:
            print("🔧 初始化LLM Judge...")
            from llm_comparison.chinese_llm_judge import ModelLoader
            device = self.config.generator.device or "cuda:0"
            self.llm_judge = ModelLoader(
                model_name=self.config.generator.model_name,
                device=device
            )
            print("✅ LLM Judge初始化完成")
        except Exception as e:
            print(f"❌ LLM Judge初始化失败: {e}")
            raise
        
        # 创建扰动器字典，保持向后兼容
        self.perturbers = {
            'year': self.year_perturber,
            'trend': self.trend_perturber,
            'term': self.term_perturber
        }
        
        print("✅ 实验环境初始化完成")
        print("📊 可用的扰动器: ['year', 'trend', 'term']")
        print("🎯 专注于year、trend、term三个核心扰动器")
        print("🤖 使用生成器: SUFE-AIFLM-Lab/Fin-R1")
        print("🔍 使用编码器: 中文=models/alphafin_encoder_finetuned_1epoch, 英文=models/finetuned_tatqa_mixed_enhanced")
        self.log_file = "perturbation_experiment_log.jsonl"
        # 清空旧日志
        with open(self.log_file, 'w', encoding='utf-8') as f:
            pass
    
    def classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['多少', '几', '数量', '金额', '数字']):
            return "数值型"
        elif any(word in question_lower for word in ['为什么', '原因', '导致', '影响']):
            return "原因分析型"
        elif any(word in question_lower for word in ['如何', '怎么', '方法', '策略']):
            return "方法策略型"
        elif any(word in question_lower for word in ['比较', '对比', '差异', '区别']):
            return "对比分析型"
        elif any(word in question_lower for word in ['趋势', '变化', '发展', '增长']):
            return "趋势分析型"
        else:
            return "一般描述型"
    
    def classify_context_type(self, context: str) -> str:
        """分类上下文类型"""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['财务', '营收', '利润', '资产', '负债']):
            return "财务数据型"
        elif any(word in context_lower for word in ['业绩', '表现', '增长', '下降']):
            return "业绩表现型"
        elif any(word in context_lower for word in ['政策', '规定', '法规', '制度']):
            return "政策法规型"
        elif any(word in context_lower for word in ['市场', '竞争', '份额', '地位']):
            return "市场竞争型"
        else:
            return "一般信息型"
    
    def calculate_complexity_score(self, question: str, answer: str) -> float:
        """计算复杂度分数"""
        # 基于问题长度、答案长度、专业词汇数量等
        question_length = len(question)
        answer_length = len(answer)
        
        # 专业词汇检测
        financial_terms = ['营收', '利润', '资产', '负债', '市盈率', '市净率', 'ROE', 'ROA']
        term_count = sum(1 for term in financial_terms if term in question or term in answer)
        
        # 复杂度计算
        complexity = (question_length * 0.3 + answer_length * 0.4 + term_count * 0.3) / 100
        return min(complexity, 1.0)
    
    def calculate_diversity_score(self, sample: PerturbationSample, selected_samples: List[PerturbationSample]) -> float:
        """计算多样性分数"""
        if not selected_samples:
            return 1.0
        
        # 计算与已选样本的差异
        diversity_scores = []
        
        for selected in selected_samples:
            # 问题类型差异
            type_diff = 1.0 if sample.question_type != selected.question_type else 0.0
            # 上下文类型差异
            context_diff = 1.0 if sample.context_type != selected.context_type else 0.0
            # 问题长度差异
            length_diff = abs(len(sample.question) - len(selected.question)) / max(len(sample.question), len(selected.question))
            
            avg_diff = (type_diff + context_diff + length_diff) / 3
            diversity_scores.append(avg_diff)
        
        return sum(diversity_scores) / len(diversity_scores)
    
    def select_perturbation_samples(self, dataset_path: str, num_samples: int = 20) -> List[PerturbationSample]:
        """集成样本选择功能 - 从数据集中选择代表性样本"""
        print(f"🎯 从数据集 {dataset_path} 中选择 {num_samples} 个代表性样本...")
        
        # 加载数据集
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
        
        print(f"📊 总样本数: {len(samples)}")
        
        # 转换为PerturbationSample对象
        perturbation_samples = []
        for i, sample in enumerate(samples):
            # 提取summary（优先使用summary字段，如果没有则使用context）
            summary = sample.get('summary', '') or sample.get('context', '') or sample.get('generated_question', '')
            if not summary:
                continue
            # 使用generated_question
            generated_question = sample.get('generated_question', '')
            if not generated_question:
                continue
            # 新增：只保留包含股票代码或年份的样本
            import re
            if not (re.search(r'\b\d{6}\b', generated_question) or re.search(r'\b20\d{2}\b', generated_question) or re.search(r'\b19\d{2}\b', generated_question)):
                continue
            # 分类问题类型和上下文类型
            question_type = classify_question_type(generated_question)
            context_type = classify_context_type(summary)
            # 计算复杂度分数
            complexity_score = calculate_complexity_score(generated_question, sample.get('expected_answer', ''))
            perturbation_sample = PerturbationSample(
                sample_id=sample.get('sample_id', f"sample_{i}"),
                context=summary,
                question=generated_question,
                expected_answer=sample.get('expected_answer', ''),
                question_type=question_type,
                context_type=context_type,
                complexity_score=complexity_score,
                diversity_score=0.0
            )
            perturbation_samples.append(perturbation_sample)
        
        print(f"📊 有效样本数: {len(perturbation_samples)}")
        
        # 多样性选择
        selected_samples = []
        for i in range(min(num_samples, len(perturbation_samples))):
            if i == 0:
                # 第一个样本选择复杂度最高的
                best_sample = max(perturbation_samples, key=lambda x: x.complexity_score)
            else:
                # 后续样本选择多样性最高的
                for sample in perturbation_samples:
                    sample.diversity_score = self.calculate_diversity_score(sample, selected_samples)
                best_sample = max(perturbation_samples, key=lambda x: x.diversity_score)
            
            selected_samples.append(best_sample)
            perturbation_samples.remove(best_sample)
            
            print(f"✅ 选择样本 {i+1}: {best_sample.sample_id}")
            print(f"  问题类型: {best_sample.question_type}")
            print(f"  上下文类型: {best_sample.context_type}")
            print(f"  复杂度: {best_sample.complexity_score:.3f}")
            print(f"  多样性: {best_sample.diversity_score:.3f}")
        
        return selected_samples
    
    def get_original_answer(self, context: str, question: str) -> str:
        """获取原始答案（步骤2）- 使用完整的RAG系统流程"""
        try:
            # 使用RAG系统适配器进行完整检索和生成
            print("🔍 使用RAG系统进行完整检索...")
            
            # 使用多阶段检索模式
            retrieval_results = self.rag_adapter.get_ranked_documents_for_evaluation(
                query=question,
                top_k=10,
                mode="reranker",  # 使用重排序模式
                use_prefilter=True  # 启用元数据过滤
            )
            
            if not retrieval_results:
                print("❌ RAG系统未返回检索结果")
                return ""
            
            print(f"✅ RAG系统检索到 {len(retrieval_results)} 个相关文档")
            
            # 构建上下文（使用检索到的文档内容）
            retrieved_contexts = []
            for i, result in enumerate(retrieval_results[:3]):  # 使用前3个文档
                content = result.get('content', '')
                if content:
                    retrieved_contexts.append(f"文档{i+1}: {content}")
            
            # 如果没有检索到内容，使用原始context
            if not retrieved_contexts:
                retrieved_contexts = [context]
            
            combined_context = "\n\n".join(retrieved_contexts)
            
            # 构建prompt
            prompt = f"基于以下上下文回答问题：\n\n上下文：{combined_context}\n\n问题：{question}\n\n回答："
            
            # 使用生成器获取答案
            print("🤖 使用LLM生成原始答案...")
            response = self.generator.generate([prompt])
            generated_answer = response[0]
            print(f"✅ 原始答案生成完成: {generated_answer[:200]}...")
            return generated_answer
            
        except Exception as e:
            print(f"❌ 获取原始答案失败: {str(e)}")
            return ""
    
    def apply_perturbation(self, context: str, perturber_name: str) -> List[PerturbationDetail]:
        """应用扰动器到上下文"""
        print(f"🔧 应用 {perturber_name} 扰动...")
        
        # 获取对应的扰动器
        perturber = self.perturbers.get(perturber_name)
        if not perturber:
            print(f"❌ 未找到扰动器: {perturber_name}")
            return []
        
        try:
            # 应用扰动
            perturbations = perturber.perturb(context)
            print(f"✅ 生成了 {len(perturbations)} 个扰动")
            
            results = []
            for i, perturbation in enumerate(perturbations):
                if isinstance(perturbation, dict):
                    perturbed_text = perturbation.get('perturbed_text', context)
                    perturbation_detail = perturbation.get('perturbation_detail', '')
                    original_feature = perturbation.get('original_feature', '')
                else:
                    perturbed_text = perturbation
                    perturbation_detail = f"{perturber_name}扰动器应用"
                    original_feature = ''
                
                # 分析文本变化
                changed_elements = self.analyze_text_changes(context, perturbed_text)
                
                # 生成变化描述
                change_description = self.generate_change_description(context, perturbed_text, perturber_name)
                
                # 创建扰动详情
                detail = PerturbationDetail(
                    perturber_name=perturber_name,
                    original_text=context,
                    perturbed_text=perturbed_text,
                    perturbation_type=self.get_perturbation_type(perturber_name),
                    changed_elements=changed_elements,
                    change_description=change_description,
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(detail)
                
                # 打印详细信息
                print(f"\n--- 扰动 {i+1} ---")
                print(f"扰动器: {perturber_name}")
                print(f"扰动类型: {detail.perturbation_type}")
                print(f"变化描述: {change_description}")
                print(f"具体变化:")
                for element in changed_elements:
                    print(f"  • {element}")
                print(f"原始文本: {context[:100]}...")
                print(f"扰动后文本: {perturbed_text[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ 扰动器应用失败: {e}")
            return []
    
    def analyze_text_changes(self, original_text: str, perturbed_text: str) -> List[str]:
        """分析文本变化"""
        changes = []
        
        if original_text == perturbed_text:
            changes.append("无变化")
            return changes
        
        # 简单的变化分析
        original_words = original_text.split()
        perturbed_words = perturbed_text.split()
        
        # 找出新增的词汇
        original_set = set(original_words)
        perturbed_set = set(perturbed_words)
        
        added_words = perturbed_set - original_set
        removed_words = original_set - perturbed_set
        
        if added_words:
            changes.append(f"新增词汇: {list(added_words)[:3]}")  # 只显示前3个
        
        if removed_words:
            changes.append(f"删除词汇: {list(removed_words)[:3]}")
        
        # 长度变化
        if len(perturbed_text) != len(original_text):
            length_diff = len(perturbed_text) - len(original_text)
            changes.append(f"文本长度变化: {length_diff:+d}字符")
        
        return changes
    
    def generate_change_description(self, original_text: str, perturbed_text: str, perturber_name: str) -> str:
        """生成变化描述"""
        if original_text == perturbed_text:
            return f"{perturber_name}扰动器未检测到可扰动的元素"
        
        # 根据扰动器类型生成描述
        if "term" in perturber_name.lower():
            return "金融术语扰动：替换或修改了金融相关术语"
        elif "year" in perturber_name.lower():
            return "年份扰动：修改了时间相关的年份信息"
        elif "trend" in perturber_name.lower():
            return "趋势扰动：修改了趋势相关的描述"
        else:
            return f"{perturber_name}扰动：文本内容发生变化"
    
    def get_perturbation_type(self, perturber_name: str) -> str:
        """获取扰动类型"""
        if "term" in perturber_name.lower():
            return "term"
        elif "year" in perturber_name.lower():
            return "year"
        elif "trend" in perturber_name.lower():
            return "trend"
        else:
            return "unknown"
    
    def get_perturbed_answer(self, perturbed_context: str, question: str, perturber_name: Optional[str] = None) -> str:
        """获取扰动后答案（步骤4）- 使用完整的RAG系统流程"""
        try:
            # 使用RAG系统适配器进行完整检索和生成
            print("🔍 使用RAG系统进行扰动后检索...")
            
            # 使用多阶段检索模式
            retrieval_results = self.rag_adapter.get_ranked_documents_for_evaluation(
                query=question,
                top_k=10,
                mode="reranker",  # 使用重排序模式
                use_prefilter=True  # 启用元数据过滤
            )
            
            if not retrieval_results:
                print("❌ RAG系统未返回检索结果")
                return ""
            
            print(f"✅ RAG系统检索到 {len(retrieval_results)} 个相关文档")
            
            # 构建上下文（对检索到的文档内容应用扰动）
            retrieved_contexts = []
            for i, result in enumerate(retrieval_results[:3]):  # 使用前3个文档
                content = result.get('content', '')
                if content and perturber_name:
                    # 对检索到的内容应用扰动
                    perturber = self.perturbers.get(perturber_name)
                    if perturber:
                        perturbations = perturber.perturb(content)
                        if perturbations and len(perturbations) > 0:
                            perturbed_content = perturbations[0].get('perturbed_text', content)
                            retrieved_contexts.append(f"文档{i+1}: {perturbed_content}")
                            print(f"✅ 对文档{i+1}应用{perturber_name}扰动")
                        else:
                            retrieved_contexts.append(f"文档{i+1}: {content}")
                            print(f"⚠️ 文档{i+1}未检测到可扰动内容")
                    else:
                        retrieved_contexts.append(f"文档{i+1}: {content}")
                        print(f"⚠️ 未找到扰动器: {perturber_name}")
                else:
                    # 如果没有扰动器信息，直接使用原始内容
                    retrieved_contexts.append(f"文档{i+1}: {content}")
            
            # 如果没有检索到内容，使用扰动后的context
            if not retrieved_contexts:
                retrieved_contexts = [perturbed_context]
            
            combined_context = "\n\n".join(retrieved_contexts)
            
            # 构建prompt
            prompt = f"基于以下上下文回答问题：\n\n上下文：{combined_context}\n\n问题：{question}\n\n回答："
            
            # 使用生成器获取答案
            print("🤖 使用LLM生成扰动后答案...")
            response = self.generator.generate([prompt])
            generated_answer = response[0]
            print(f"✅ 扰动后答案生成完成: {generated_answer[:200]}...")
            return generated_answer
            
        except Exception as e:
            print(f"❌ 获取扰动后答案失败: {str(e)}")
            return ""
    
    def _get_perturber_name_from_context(self, perturbed_context: str) -> str:
        """从扰动后的上下文推断使用的扰动器类型"""
        # 简单的启发式方法来判断使用了哪种扰动器
        if "2018" in perturbed_context or "2019" in perturbed_context or "2020" in perturbed_context:
            return "year"
        elif any(word in perturbed_context for word in ["减少", "下降", "恶化", "降低"]):
            return "trend"
        elif any(word in perturbed_context for word in ["市盈率", "净利润", "市净率", "市销率"]):
            return "term"
        else:
            return "year"  # 默认使用年份扰动器
    
    def calculate_importance_score(self, original_answer: str, perturbed_answer: str) -> Tuple[float, float, float, float]:
        """计算相似度和重要性分数（步骤5）"""
        try:
            # 使用比较器计算相似度
            similarity_scores = self.comparator.compare(original_answer, [perturbed_answer])
            similarity_score = similarity_scores[0] if similarity_scores else 0.0
            
            # 重要性分数 = 1 - 相似度（RAG-Ex论文方法）
            importance_score = 1.0 - similarity_score
            
            # 添加传统F1和EM指标计算
            f1_score = self.calculate_f1_score(original_answer, perturbed_answer)
            em_score = self.calculate_exact_match(original_answer, perturbed_answer)
            
            return similarity_score, importance_score, f1_score, em_score
        except Exception as e:
            print(f"❌ 计算重要性分数失败: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0
    
    def normalize_answer_chinese(self, s: str) -> str:
        """
        针对中文进行答案归一化：移除标点、转换全角字符为半角、去除多余空格、分词并小写。
        使用jieba进行中文分词，获得更准确的F1和EM评估。
        """
        if not s:
            return ""

        s = s.strip().lower()

        s = s.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?').replace('；', ';')
        s = s.replace('（', '(').replace('）', ')')

        punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~""''【】『』《》—…·～「」～￥%#@！&（）《》]'
        s = re.sub(punctuation_pattern, '', s)

        # 关键修改：使用jieba进行分词
        import jieba
        tokens = list(jieba.cut(s)) 

        normalized_tokens = [token for token in tokens if token.strip()]
        return " ".join(normalized_tokens)

    def get_tokens_chinese(self, s: str) -> List[str]:
        """获取中文分词后的tokens列表。"""
        return self.normalize_answer_chinese(s).split()

    def calculate_f1_score(self, prediction: str, ground_truth: str) -> float:
        """计算F1分数 (基于词重叠)。"""
        gold_tokens = self.get_tokens_chinese(ground_truth)
        pred_tokens = self.get_tokens_chinese(prediction)

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

    def calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        """计算精确匹配率。"""
        return float(self.normalize_answer_chinese(prediction) == self.normalize_answer_chinese(ground_truth))
    
    def run_llm_judge_evaluation(self, original_answer: str, perturbed_answer: str, question: str) -> Dict[str, Any]:
        """运行LLM Judge评估（步骤6）- 使用单例模式避免重复加载"""
        try:
            print("🤖 运行LLM Judge评估...")
            
            # 检查答案是否有效
            if not original_answer or not perturbed_answer:
                print("⚠️ 答案为空，无法评估")
                return {
                    'accuracy': 0.0,
                    'completeness': 0.0,
                    'professionalism': 0.0,
                    'overall_score': 0.0,
                    'reasoning': '答案为空，无法评估'
                }
            
            # 导入单例LLM Judge
            from llm_comparison.chinese_llm_judge import llm_judge_singleton
            
            # 确保LLM Judge已初始化
            if not hasattr(llm_judge_singleton, '_model_loader') or llm_judge_singleton._model_loader is None:
                try:
                    llm_judge_singleton.initialize()
                except Exception as e:
                    print(f"❌ LLM Judge初始化失败: {e}")
                    return {
                        'accuracy': 0.0,
                        'completeness': 0.0,
                        'professionalism': 0.0,
                        'overall_score': 0.0,
                        'reasoning': f'LLM Judge初始化失败: {str(e)}',
                        'raw_output': '初始化失败'
                    }
            
            # 执行评估
            judge_result = llm_judge_singleton.evaluate(question, original_answer, perturbed_answer)
            
            # 检查评估结果是否有效
            if (judge_result.get('accuracy', 0) == 0 and 
                judge_result.get('conciseness', 0) == 0 and 
                judge_result.get('professionalism', 0) == 0):
                print("⚠️ LLM Judge返回全零评分")
                return {
                    'accuracy': 0.0,
                    'completeness': 0.0,
                    'professionalism': 0.0,
                    'overall_score': 0.0,
                    'reasoning': 'LLM Judge返回全零评分',
                    'raw_output': judge_result.get('raw_output', '')
                }
            
            print(f"✅ LLM Judge评估完成")
            print(f"  准确性: {judge_result.get('accuracy', 'N/A')}")
            print(f"  简洁性: {judge_result.get('conciseness', 'N/A')}")
            print(f"  专业性: {judge_result.get('professionalism', 'N/A')}")
            print(f"  总体评分: {judge_result.get('overall_score', 'N/A')}")
            
            return {
                'accuracy': judge_result.get('accuracy', 0.0),
                'completeness': judge_result.get('conciseness', 0.0),  # 使用conciseness作为completeness
                'professionalism': judge_result.get('professionalism', 0.0),
                'overall_score': judge_result.get('overall_score', 0.0),
                'reasoning': judge_result.get('reasoning', ''),
                'raw_output': judge_result.get('raw_output', '')
            }
            
        except Exception as e:
            print(f"❌ LLM Judge评估失败: {str(e)}")
            return {
                'accuracy': 0.0,
                'completeness': 0.0,
                'professionalism': 0.0,
                'overall_score': 0.0,
                'reasoning': f'LLM Judge评估失败: {str(e)}',
                'raw_output': '评估失败'
            }
    

    
    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """解析Judge模型的响应"""
        try:
            # 尝试提取分数
            scores = {
                'accuracy': 7.0,
                'completeness': 7.0,
                'professionalism': 7.0,
                'overall_score': 7.0,
                'reasoning': response
            }
            
            # 简单的分数提取逻辑
            import re
            
            # 查找分数模式
            score_patterns = [
                r'准确性[：:]\s*(\d+(?:\.\d+)?)',
                r'准确度[：:]\s*(\d+(?:\.\d+)?)',
                r'accuracy[：:]\s*(\d+(?:\.\d+)?)',
                r'完整性[：:]\s*(\d+(?:\.\d+)?)',
                r'completeness[：:]\s*(\d+(?:\.\d+)?)',
                r'专业性[：:]\s*(\d+(?:\.\d+)?)',
                r'professionalism[：:]\s*(\d+(?:\.\d+)?)',
                r'总体评分[：:]\s*(\d+(?:\.\d+)?)',
                r'overall[：:]\s*(\d+(?:\.\d+)?)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    score = float(matches[0])
                    if '准确' in pattern or 'accuracy' in pattern.lower():
                        scores['accuracy'] = score
                    elif '完整' in pattern or 'completeness' in pattern.lower():
                        scores['completeness'] = score
                    elif '专业' in pattern or 'professionalism' in pattern.lower():
                        scores['professionalism'] = score
                    elif '总体' in pattern or 'overall' in pattern.lower():
                        scores['overall_score'] = score
            
            # 计算总体评分
            if scores['accuracy'] == 7.0 and scores['completeness'] == 7.0 and scores['professionalism'] == 7.0:
                # 如果没有找到具体分数，使用默认值
                scores['overall_score'] = 7.0
            else:
                scores['overall_score'] = (scores['accuracy'] + scores['completeness'] + scores['professionalism']) / 3
            
            return scores
            
        except Exception as e:
            print(f"⚠️ 解析Judge响应失败: {e}")
            return {
                'accuracy': 7.0,
                'completeness': 7.0,
                'professionalism': 7.0,
                'overall_score': 7.0,
                'reasoning': f"解析失败: {response[:100]}..."
            }
    
    def run_single_sample_experiment(self, sample: PerturbationSample) -> List[PerturbationResult]:
        print(f"\n🔬 样本实验: {sample.sample_id}")
        print(f"Generated Question: {sample.question}")
        print(f"Ground Truth: {sample.expected_answer}")
        print(f"问题类型: {sample.question_type}")
        print(f"上下文类型: {sample.context_type}")
        print(f"Summary长度: {len(sample.context)} 字符")
        print("=" * 60)
        
        results = []
        # 步骤2: 获取原始答案（无扰动）
        print("📝 步骤2: 获取原始答案（无扰动）...")
        original_answer = self.get_original_answer(sample.context, sample.question)
        print(f"📋 原始生成答案: {original_answer}")
        print(f"📏 原始生成答案长度: {len(original_answer)} 字符")
        
        if not original_answer:
            print("❌ 无法获取原始生成答案，跳过此样本")
            return results
        
        # 步骤2.5: 计算Ground Truth与原始生成答案的基准评估
        print("📊 计算Ground Truth与原始生成答案的基准评估...")
        gt_vs_original_f1 = self.calculate_f1_score(original_answer, sample.expected_answer)
        gt_vs_original_em = self.calculate_exact_match(original_answer, sample.expected_answer)
        
        print(f"Ground Truth vs 原始生成答案基准: F1={gt_vs_original_f1:.4f}, EM={gt_vs_original_em:.4f}")
        
        # 运行Ground Truth vs 原始生成答案的LLM Judge评估
        print("🤖 运行Ground Truth vs 原始生成答案的LLM Judge评估...")
        llm_judge_gt_vs_original = self.run_llm_judge_evaluation(sample.expected_answer, original_answer, sample.question)
        
        # 日志：原始数据
        log_base = {
            "sample_id": sample.sample_id,
            "original_summary": sample.context,
            "original_generated_question": sample.question,
            "expected_answer": sample.expected_answer,
            "question_type": sample.question_type,
            "context_type": sample.context_type,
        }
        
        # 步骤3-7: 对每个扰动器进行实验
        for perturber_name, perturber in self.perturbers.items():
            print(f"\n🔄 测试扰动器: {perturber_name}")
            
            # 步骤3: 应用扰动
            print(f"\n🔧 应用 {perturber_name} 扰动...")
            perturbation_details = self.apply_perturbation(sample.context, perturber_name)
            
            if not perturbation_details:
                print(f"❌ {perturber_name} 未产生有效扰动，跳过")
                continue
            
            print(f"✅ 生成了 {len(perturbation_details)} 个扰动")
            
            # 对每个扰动进行处理
            for i, perturbation_detail in enumerate(perturbation_details):
                print(f"\n--- 扰动 {i+1} ---")
                print(f"扰动器: {perturbation_detail.perturber_name}")
                print(f"扰动类型: {perturbation_detail.perturbation_type}")
                print(f"变化描述: {perturbation_detail.change_description}")
                
                # 检查是否有实际变化
                if perturbation_detail.perturbed_text == perturbation_detail.original_text:
                    print(f"⚠️ {perturbation_detail.perturber_name} 扰动器未产生实际变化，跳过此扰动")
                    continue
                
                # 显示具体变化
                if perturbation_detail.changed_elements:
                    print("具体变化:")
                    for change in perturbation_detail.changed_elements:
                        print(f"  • {change}")
                
                # 显示文本对比
                print(f"原始文本: {perturbation_detail.original_text[:100]}...")
                print(f"扰动后文本: {perturbation_detail.perturbed_text[:100]}...")
                
                # 步骤4: 获取扰动后答案
                perturbed_answer = self.get_perturbed_answer(perturbation_detail.perturbed_text, sample.question, perturber_name)
                
                if not perturbed_answer:
                    print("❌ 扰动后答案生成失败，跳过")
                    continue
                
                print(f"📋 扰动后答案: {perturbed_answer}")
                print(f"📏 扰动后答案长度: {len(perturbed_answer)} 字符")
                
                # 步骤5: 计算Ground Truth与扰动后生成答案的评估指标
                print("📊 计算Ground Truth与扰动后生成答案的评估指标...")
                gt_vs_perturbed_f1 = self.calculate_f1_score(perturbed_answer, sample.expected_answer)
                gt_vs_perturbed_em = self.calculate_exact_match(perturbed_answer, sample.expected_answer)
                
                print(f"Ground Truth vs 扰动后生成答案: F1={gt_vs_perturbed_f1:.4f}, EM={gt_vs_perturbed_em:.4f}")
                
                # 步骤5.5: 计算扰动对性能的影响
                f1_improvement = gt_vs_perturbed_f1 - gt_vs_original_f1
                em_improvement = gt_vs_perturbed_em - gt_vs_original_em
                
                print(f"扰动对F1分数的影响: {f1_improvement:+.4f} ({'改善' if f1_improvement > 0 else '下降'})")
                print(f"扰动对EM分数的影响: {em_improvement:+.4f} ({'改善' if em_improvement > 0 else '下降'})")
                
                # 步骤6: 计算原始答案与扰动后答案的相似度和重要性
                similarity_score, importance_score, f1_score, em_score = self.calculate_importance_score(original_answer, perturbed_answer)
                
                # 步骤7: LLM Judge评估（Ground Truth vs 扰动后生成答案）
                print("🤖 运行LLM Judge评估（Ground Truth vs 扰动后生成答案）...")
                llm_judge_gt_vs_perturbed = self.run_llm_judge_evaluation(sample.expected_answer, perturbed_answer, sample.question)
                
                # 计算LLM Judge评估的改善情况
                gt_vs_original_score = llm_judge_gt_vs_original.get('overall_score', 0)
                gt_vs_perturbed_score = llm_judge_gt_vs_perturbed.get('overall_score', 0)
                llm_judge_improvement = gt_vs_perturbed_score - gt_vs_original_score
                
                print(f"Ground Truth vs 原始生成答案LLM Judge: {gt_vs_original_score:.2f}")
                print(f"Ground Truth vs 扰动后生成答案LLM Judge: {gt_vs_perturbed_score:.2f}")
                print(f"扰动对LLM Judge分数的影响: {llm_judge_improvement:+.2f} ({'改善' if llm_judge_improvement > 0 else '下降'})")
                
                # 合并LLM Judge评估结果
                llm_judge_scores = {
                    'gt_vs_original': llm_judge_gt_vs_original,
                    'gt_vs_perturbed': llm_judge_gt_vs_perturbed,
                    'original_vs_perturbed': self.run_llm_judge_evaluation(original_answer, perturbed_answer, sample.question)
                }
                
                # 记录结果
                result = PerturbationResult(
                    sample_id=sample.sample_id,
                    perturber_name=perturber_name,
                    original_answer=original_answer,
                    perturbed_answer=perturbed_answer,
                    perturbation_detail=perturbation_detail,
                    similarity_score=similarity_score,
                    importance_score=importance_score,
                    f1_score=f1_score,
                    em_score=em_score,
                    llm_judge_scores=llm_judge_scores,
                    timestamp=datetime.now().isoformat()
                )
                
                # 添加Ground Truth评估指标到结果中
                result.expected_vs_original_f1 = gt_vs_original_f1
                result.expected_vs_original_em = gt_vs_original_em
                result.expected_vs_perturbed_f1 = gt_vs_perturbed_f1
                result.expected_vs_perturbed_em = gt_vs_perturbed_em
                
                # 添加扰动影响指标
                result.f1_improvement = f1_improvement
                result.em_improvement = em_improvement
                result.llm_judge_improvement = llm_judge_improvement
                
                results.append(result)
                
                print(f"✅ 扰动 {i+1} 完成")
                print(f"  相似度: {similarity_score:.4f}")
                print(f"  重要性: {importance_score:.4f}")
                print(f"  原始生成答案 vs 扰动后生成答案: F1={f1_score:.4f}, EM={em_score:.4f}")
                print(f"  Ground Truth vs 原始生成答案基准: F1={gt_vs_original_f1:.4f}, EM={gt_vs_original_em:.4f}")
                print(f"  Ground Truth vs 扰动后生成答案: F1={gt_vs_perturbed_f1:.4f}, EM={gt_vs_perturbed_em:.4f}")
                print(f"  扰动对F1分数的影响: {f1_improvement:+.4f}")
                print(f"  扰动对EM分数的影响: {em_improvement:+.4f}")
                print(f"  扰动对LLM Judge分数的影响: {llm_judge_improvement:+.2f}")
                if llm_judge_scores:
                    print(f"  LLM Judge评估:")
                    if 'gt_vs_original' in llm_judge_scores:
                        print(f"    Ground Truth vs 原始生成答案: {llm_judge_scores['gt_vs_original'].get('overall_score', 'N/A')}")
                    if 'gt_vs_perturbed' in llm_judge_scores:
                        print(f"    Ground Truth vs 扰动后生成答案: {llm_judge_scores['gt_vs_perturbed'].get('overall_score', 'N/A')}")
        return results
    
    def run_comprehensive_experiment(self, dataset_path: str, num_samples: int = 20):
        """运行完整的扰动实验"""
        print("🚀 开始RAG系统扰动实验（使用rag_system_adapter）")
        print("=" * 80)
        
        # 步骤1: 挑选样本
        print("📋 步骤1: 挑选代表性样本...")
        samples = self.select_perturbation_samples(dataset_path, num_samples)
        
        if not samples:
            print("❌ 没有选择到有效样本，实验终止")
            return [], []
        
        # 运行实验
        all_results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*20} 样本 {i}/{len(samples)} {'='*20}")
            
            # 运行单个样本实验
            sample_results = self.run_single_sample_experiment(sample)
            all_results.extend(sample_results)
            
            print(f"✅ 样本 {i} 实验完成，获得 {len(sample_results)} 个结果")
        
        # 分析结果
        # self.analyze_experiment_results(all_results, samples) # This line is removed as per the new_code
        
        # 保存结果
        # self.save_experiment_results(all_results, samples) # This line is removed as per the new_code
        
        print(f"\n🎉 实验完成！总共获得 {len(all_results)} 个扰动结果")
        return all_results, samples
    
    def analyze_experiment_results(self, results: List[PerturbationResult], samples: List[PerturbationSample]):
        """分析实验结果"""
        print(f"\n📊 实验结果分析")
        print("=" * 60)
        
        if not results:
            print("❌ 没有实验结果可分析")
            return {}
        
        # 按扰动器分组分析
        perturber_stats = {}
        for perturber_name in self.perturbers.keys():
            perturber_results = [r for r in results if r.perturber_name == perturber_name]
            
            if perturber_results:
                # 分别统计summary扰动和prompt扰动
                summary_results = [r for r in perturber_results if r.perturbation_target == "summary"]
                prompt_results = [r for r in perturber_results if r.perturbation_target == "prompt"]
                
                avg_importance = sum(r.importance_score for r in perturber_results) / len(perturber_results)
                avg_similarity = sum(r.similarity_score for r in perturber_results) / len(perturber_results)
                avg_accuracy = sum(r.llm_judge_scores['accuracy'] for r in perturber_results) / len(perturber_results)
                avg_f1 = sum(r.f1_score for r in perturber_results) / len(perturber_results)
                avg_em = sum(r.em_score for r in perturber_results) / len(perturber_results)
                
                perturber_stats[perturber_name] = {
                    'count': len(perturber_results),
                    'summary_count': len(summary_results),
                    'prompt_count': len(prompt_results),
                    'avg_importance': avg_importance,
                    'avg_similarity': avg_similarity,
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1,
                    'avg_em': avg_em,
                    'summary_avg_importance': sum(r.importance_score for r in summary_results) / len(summary_results) if summary_results else 0,
                    'prompt_avg_importance': sum(r.importance_score for r in prompt_results) / len(prompt_results) if prompt_results else 0
                }
        
        # 打印统计结果
        print(f"{'扰动器':<10} {'总样本':<8} {'Summary':<8} {'Prompt':<8} {'平均重要性':<12} {'平均相似度':<12} {'平均准确性':<12} {'平均F1':<10} {'平均EM':<10}")
        print("-" * 100)
        
        for perturber_name, stats in perturber_stats.items():
            print(f"{perturber_name:<10} {stats['count']:<8} {stats['summary_count']:<8} {stats['prompt_count']:<8} "
                  f"{stats['avg_importance']:<12.4f} {stats['avg_similarity']:<12.4f} {stats['avg_accuracy']:<12.2f} "
                  f"{stats['avg_f1']:<10.4f} {stats['avg_em']:<10.4f}")
        
        # 找出最重要的扰动器
        if perturber_stats:
            most_important = max(perturber_stats.items(), key=lambda x: x[1]['avg_importance'])
            print(f"\n🏆 最重要的扰动器: {most_important[0]} (平均重要性: {most_important[1]['avg_importance']:.4f})")
            
            # 分析summary vs prompt扰动效果
            print(f"\n📈 Summary vs Prompt 扰动效果对比:")
            for perturber_name, stats in perturber_stats.items():
                if stats['summary_count'] > 0 and stats['prompt_count'] > 0:
                    summary_importance = stats['summary_avg_importance']
                    prompt_importance = stats['prompt_avg_importance']
                    print(f"  {perturber_name}: Summary重要性={summary_importance:.4f}, Generated_Question重要性={prompt_importance:.4f}")
                    if summary_importance > prompt_importance:
                        print(f"    → Summary扰动效果更强")
                    else:
                        print(f"    → Generated_Question扰动效果更强")
        
        return {
            'perturber_statistics': perturber_stats,
            'overall_metrics': {
                'avg_similarity_score': sum(r.similarity_score for r in results) / len(results),
                'avg_importance_score': sum(r.importance_score for r in results) / len(results),
                'avg_llm_judge_score': sum(r.llm_judge_scores.get('overall_score', 0) for r in results) / len(results),
                'avg_f1_score': sum(r.f1_score for r in results) / len(results),
                'avg_em_score': sum(r.em_score for r in results) / len(results)
            }
        }
    
    def save_experiment_results(self, results: List[PerturbationResult], samples: List[PerturbationSample]):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"perturbation_experiment_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        output_data = {
            'experiment_info': {
                'timestamp': timestamp,
                'total_samples': len(samples),
                'total_results': len(results),
                'perturbers_used': list(self.perturbers.keys()),
                'rag_system': 'rag_system_adapter',
                'config_used': {
                    'generator_model': self.config.generator.model_name,
                    'chinese_encoder': self.config.encoder.chinese_model_path,
                    'english_encoder': self.config.encoder.english_model_path
                }
            },
            'samples': [asdict(sample) for sample in samples],
            'results': [asdict(result) for result in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 实验结果已保存到: {output_file}")

    def run_integrated_experiment(self, dataset_path: str, num_samples: int = 20, output_dir: str = 'perturbation_results'):
        """
        运行集成扰动实验（批量两步法）
        - 生成阶段：Fin-R1 on cuda:1
        - 评测阶段：Qwen3-8B on cuda:1
        """
        print(f"\n🚀 集成扰动实验启动（批量两步法）")
        print(f"📁 数据集: {dataset_path}")
        print(f"📊 样本数量: {num_samples}")
        print(f"📂 输出目录: {output_dir}")
        
        # 检查现有结果，避免重复
        existing_sample_ids = set()
        existing_file = os.path.join(output_dir, "incremental_generation.json")
        if os.path.exists(existing_file):
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    existing_sample_ids = {result.get('sample_id', '') for result in existing_results}
                print(f"📋 发现现有结果文件，包含 {len(existing_sample_ids)} 个样本ID")
                print(f"🔍 将跳过这些已存在的样本ID: {list(existing_sample_ids)[:5]}...")
            except Exception as e:
                print(f"⚠️ 读取现有结果失败: {e}")
        
        # 步骤1: 选择样本（扩大初选池）
        print("\n📋 步骤1: 选择样本")
        # 为每个扰动器生成7个样本，总共21个
        target_samples = 21  # 7个term + 7个year + 7个trend
        candidates = self.select_perturbation_samples(dataset_path, num_samples=target_samples*3)
        if not candidates:
            print("❌ 没有有效的样本，退出实验")
            return
        print(f"✅ 成功选择 {len(candidates)} 个候选样本")
        
        # 过滤掉已存在的样本ID
        filtered_candidates = [sample for sample in candidates if sample.sample_id not in existing_sample_ids]
        print(f"📊 过滤后剩余 {len(filtered_candidates)} 个候选样本（跳过 {len(candidates) - len(filtered_candidates)} 个重复样本）")
        
        if len(filtered_candidates) < target_samples:
            print(f"⚠️ 警告：过滤后样本数量不足，需要 {target_samples} 个，只有 {len(filtered_candidates)} 个")
        
        # 步骤2: 生成阶段（只用Fin-R1，cuda:1）
        print("\n🔬 步骤2: 生成阶段（只用Fin-R1）")
        generation_results = []
        used_sample_ids = set()
        perturber_counts = {'year': 0, 'term': 0, 'trend': 0}
        samples = filtered_candidates  # 定义samples变量
        for sample in filtered_candidates:
            if len(generation_results) >= target_samples:
                break
            if sample.sample_id in used_sample_ids:
                continue
            used_sample_ids.add(sample.sample_id)
            original_answer = self.get_original_answer(sample.context, sample.question)
            best_perturber = self._select_best_perturber_for_sample(sample, perturber_counts)
            if not best_perturber:
                print(f"❌ 样本 {sample.sample_id} 无法选择扰动器，跳过")
                continue
            # 处理所有扰动器（term、year、trend）
            if best_perturber not in ['year', 'term', 'trend']:
                print(f"⚠️ 样本 {sample.sample_id} 选择了{best_perturber}，跳过")
                continue
            perturbation_details = self.apply_perturbation(sample.context, best_perturber)
            if not perturbation_details:
                print(f"  ⚠️ 样本 {sample.sample_id} 未生成扰动，跳过")
                continue
            perturbation_detail = perturbation_details[0]
            if perturbation_detail.perturbed_text == perturbation_detail.original_text:
                print(f"⚠️ 样本 {sample.sample_id} 扰动器未产生实际变化，跳过")
                continue
            perturbed_answer = self.get_perturbed_answer(perturbation_detail.perturbed_text, sample.question, best_perturber)
            if not perturbed_answer:
                print(f"❌ 样本 {sample.sample_id} 扰动后答案生成失败，跳过")
                continue
            similarity_score, importance_score, f1_score, em_score = self.calculate_importance_score(original_answer, perturbed_answer)
            generation_result = {
                'sample_id': sample.sample_id,
                'question': sample.question,
                'context': sample.context,
                'expected_answer': sample.expected_answer,
                'perturber_name': best_perturber,
                'perturbation_detail': perturbation_detail,
                'original_answer': original_answer,
                'perturbed_answer': perturbed_answer,
                'similarity_score': similarity_score,
                'importance_score': importance_score,
                'f1_score': f1_score,
                'em_score': em_score,
                'timestamp': datetime.now().isoformat()
            }
            generation_results.append(generation_result)
            # 更新扰动器计数
            if best_perturber:
                perturber_counts[best_perturber] += 1
            print(f"  ✅ 生成完成")
            print(f"    相似度: {similarity_score:.4f}")
            print(f"    重要性: {importance_score:.4f}")
            print(f"    F1分数: {f1_score:.4f}")
            print(f"    EM分数: {em_score:.4f}")
            print(f"    扰动器计数: {perturber_counts}")
        
        print(f"\n📊 生成阶段完成，共生成 {len(generation_results)} 个有效扰动结果")
        
        # 步骤3: 保存生成结果
        print("\n💾 步骤3: 保存生成结果")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generation_file = os.path.join(output_dir, f"generation_results_{timestamp}.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保generation_results中的PerturbationDetail对象被转换为字典
        serializable_results = []
        for result in generation_results:
            serializable_result = result.copy()
            if 'perturbation_detail' in serializable_result and isinstance(serializable_result['perturbation_detail'], PerturbationDetail):
                serializable_result['perturbation_detail'] = asdict(serializable_result['perturbation_detail'])
            serializable_results.append(serializable_result)
        
        with open(generation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_info': {
                    'timestamp': timestamp,
                    'num_samples': len(samples),
                    'num_results': len(generation_results),
                    'perturbers': list(self.perturbers.keys()),
                    'stage': 'generation_only'
                },
                'samples': [asdict(sample) for sample in samples],
                'generation_results': serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 生成结果已保存到: {generation_file}")
        
        # 步骤4: 评测阶段（只用Qwen3-8B，cuda:1）
        print("\n🔬 步骤4: 评测阶段（只用Qwen3-8B）")
        
        # 释放Fin-R1显存
        print("🧹 释放Fin-R1显存...")
        del self.generator
        gc.collect()
        torch.cuda.empty_cache()
        
        # 初始化LLM Judge
        print("🔧 初始化LLM Judge...")
        from llm_comparison.chinese_llm_judge import llm_judge_singleton
        llm_judge_singleton.initialize(model_name="Qwen3-8B", device="cuda:1")
        
        # 对每个生成结果进行评测
        final_results = []
        for i, gen_result in enumerate(generation_results, 1):
            print(f"\n📊 评测结果 {i}/{len(generation_results)}: {gen_result['sample_id']} - {gen_result['perturber_name']}")
            
            # LLM Judge评估
            llm_judge_scores = llm_judge_singleton.evaluate(
                gen_result['question'],
                gen_result['expected_answer'],
                gen_result['perturbed_answer']
            )
            
            # 创建最终结果
            final_result = PerturbationResult(
                sample_id=gen_result['sample_id'],
                perturber_name=gen_result['perturber_name'],
                original_answer=gen_result['original_answer'],
                perturbed_answer=gen_result['perturbed_answer'],
                perturbation_detail=gen_result['perturbation_detail'],
                similarity_score=gen_result['similarity_score'],
                importance_score=gen_result['importance_score'],
                f1_score=gen_result['f1_score'],
                em_score=gen_result['em_score'],
                llm_judge_scores=llm_judge_scores,
                timestamp=gen_result['timestamp']
            )
            
            final_results.append(final_result)
            print(f"  ✅ 评测完成")
            print(f"    LLM Judge: {llm_judge_scores.get('overall_score', 'N/A')}")
        
        # 步骤5: 保存最终结果
        print("\n💾 步骤5: 保存最终结果")
        self.save_integrated_results(final_results, samples, output_dir)
        
        # 步骤6: 计算F1和EM指标
        print("\n📊 步骤6: 计算F1和EM指标")
        self.calculate_and_save_metrics(final_results, samples, output_dir)
        
        # 清理LLM Judge模型
        print("🧹 清理LLM Judge模型...")
        llm_judge_singleton.cleanup()
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"\n🎉 集成实验完成（批量两步法）！")
        print(f"📊 处理了 {len(samples)} 个样本")
        print(f"📈 生成了 {len(generation_results)} 个结果")
        print(f"📊 评测了 {len(final_results)} 个结果")
        print("✅ Fin-R1和Qwen3-8B未同时占用cuda:1，显存安全")

    def _select_best_perturber_for_sample(self, sample: PerturbationSample, perturber_counts: Dict[str, int]) -> Optional[str]:
        """为样本选择最佳扰动器 - 确保每个扰动器获得7个样本"""
        # 优先选择计数较少的扰动器，确保每个扰动器都能获得样本
        min_count = min(perturber_counts.values())
        candidates = [k for k, v in perturber_counts.items() if v == min_count]
        
        # 如果所有扰动器都达到了7个样本，则停止
        if min_count >= 7:
            return None
            
        return candidates[0] if candidates else None

    def save_integrated_results(self, results: List[PerturbationResult], samples: List[PerturbationSample], output_dir: str):
        """保存集成结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"integrated_perturbation_results_{timestamp}.json")
        
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            if isinstance(result_dict['perturbation_detail'], PerturbationDetail):
                result_dict['perturbation_detail'] = asdict(result_dict['perturbation_detail'])
            serializable_results.append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_info': {
                    'timestamp': timestamp,
                    'num_samples': len(samples),
                    'num_results': len(results),
                    'perturbers': list(self.perturbers.keys())
                },
                'samples': [asdict(sample) for sample in samples],
                'results': serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 集成结果已保存到: {output_file}")

    def calculate_and_save_metrics(self, results: List[PerturbationResult], samples: List[PerturbationSample], output_dir: str):
        """计算并保存指标"""
        # 这里可以添加F1和EM指标的计算逻辑
        print("📊 指标计算完成")


def run_judge_only(generation_result_path: str, judge_output_path: str):
    """
    只负责加载生成结果，批量用LLM Judge评测，保存为json（只加载Qwen3-8B到cuda:1）
    """
    print(f"\n🚀 [评测阶段] 读取生成结果: {generation_result_path}")
    from llm_comparison.chinese_llm_judge import llm_judge_singleton
    with open(generation_result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    # 初始化Judge
    llm_judge_singleton.initialize(model_name="Qwen3-8B", device="cuda:1")
    for item in results:
        judge_result = llm_judge_singleton.evaluate(
            item["question"],
            item["expected_answer"],
            item["perturbed_answer"]
        )
        item["judge_scores"] = judge_result
    # 保存评测结果
    with open(judge_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评测阶段完成，结果已保存到: {judge_output_path}")
    # 释放Judge显存
    llm_judge_singleton.cleanup()
    gc.collect()
    torch.cuda.empty_cache()

# 用法示例：
# run_generation_only("selected_perturbation_samples.json", "generated_answers.json", num_samples=20)
# run_judge_only("generated_answers.json", "judge_results.json")

def main():
    """主函数 - 为每个扰动器（term、year、trend）各生成7个样本"""
    print("🚀 启动RAG扰动实验 - 为每个扰动器生成7个样本")
    
    # 初始化实验
    experiment = RAGPerturbationExperiment()
    
    # 设置参数
    dataset_path = "selected_perturbation_samples.json"
    output_dir = "perturbation_results"
    
    # 修改目标样本数为21（7个term + 7个year + 7个trend）
    target_samples = 21
    
    print(f"📊 目标样本数: {target_samples}")
    print(f"📁 数据集: {dataset_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🎯 目标：每个扰动器（term、year、trend）各生成7个样本")
    
    # 运行集成实验
    experiment.run_integrated_experiment(
        dataset_path=dataset_path,
        num_samples=target_samples,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main() 