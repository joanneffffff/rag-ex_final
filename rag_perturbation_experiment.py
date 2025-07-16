#!/usr/bin/env python3
"""
RAG扰动实验主程序
包括样本选择、扰动应用、答案比较和重要性计算
"""

import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import gc

# 导入扰动器
from xlm.modules.perturber.term_perturber import TermPerturber
from xlm.modules.perturber.year_perturber import YearPerturber
from xlm.modules.perturber.trend_perturber import TrendPerturber

# 导入RAG系统适配器
from alphafin_data_process.rag_system_adapter import RagSystemAdapter

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerturbationDetail:
    """扰动详细信息"""
    original_text: str
    perturbed_text: str
    perturbation_type: str
    confidence: float = 1.0

@dataclass
class PerturbationResult:
    """扰动实验结果数据类"""
    sample_id: str
    question: str
    context: str
    expected_answer: str
    perturber_name: str
    perturbation_detail: PerturbationDetail
    original_answer: str
    perturbed_answer: str
    similarity_score: float = 0.0
    importance_score: float = 0.0
    f1_score: float = 0.0
    em_score: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        # 处理PerturbationDetail对象
        if isinstance(result['perturbation_detail'], PerturbationDetail):
            result['perturbation_detail'] = {
                'perturber_name': result['perturbation_detail'].perturbation_type,
                'original_text': result['perturbation_detail'].original_text,
                'perturbed_text': result['perturbation_detail'].perturbed_text,
                'perturbation_type': result['perturbation_detail'].perturbation_type,
                'changed_elements': [],
                'change_description': f"{result['perturbation_detail'].perturbation_type}扰动：修改了相关的内容信息",
                'timestamp': datetime.now().isoformat()
            }
        
        # 添加时间戳
        if not result['timestamp']:
            result['timestamp'] = datetime.now().isoformat()
        
        return result

class PerturbationExperiment:
    """扰动实验主类"""
    
    def __init__(self, config_path: str = "config/parameters.py"):
        """初始化扰动实验"""
        print("🔧 初始化扰动实验...")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化RAG系统适配器
        self.rag_system = RagSystemAdapter(self.config)
        
        # 初始化扰动器
        self.perturbers = {
            'term': TermPerturber(),
            'year': YearPerturber(),
            'trend': TrendPerturber()
        }
        
        # 初始化LLM生成器
        self.generator = self._init_generator()
        
        print("✅ 扰动实验初始化完成")
    
    def _load_config(self, config_path: str):
        """加载配置文件"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载配置文件: {config_path}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    
    def select_perturbation_samples(self, dataset_path: str, num_samples: int = 21) -> List[Dict[str, Any]]:
        """选择用于扰动实验的样本 - 确保样本可以被扰动"""
        print(f"📊 从 {dataset_path} 选择 {num_samples} 个可扰动样本...")
        
        # 读取数据集
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        
        print(f"📚 数据集包含 {len(samples)} 个样本")
        
        # 筛选可扰动的样本
        perturbable_samples = []
        for sample in samples:
            if self._is_sample_perturbable(sample):
                perturbable_samples.append(sample)
        
        print(f"🔍 找到 {len(perturbable_samples)} 个可扰动样本")
        
        # 如果可扰动样本不够，尝试更多样本
        if len(perturbable_samples) < num_samples:
            print(f"⚠️ 可扰动样本不足 ({len(perturbable_samples)} < {num_samples})，尝试扩大搜索范围...")
            # 重新检查所有样本，使用更宽松的标准
            for sample in samples:
                if sample not in perturbable_samples and self._is_sample_perturbable_relaxed(sample):
                    perturbable_samples.append(sample)
                    if len(perturbable_samples) >= num_samples:
                        break
        
        # 选择样本
        if len(perturbable_samples) <= num_samples:
            selected_samples = perturbable_samples
        else:
            selected_samples = random.sample(perturbable_samples, num_samples)
        
        print(f"✅ 已选择 {len(selected_samples)} 个可扰动样本")
        return selected_samples
    
    def _is_sample_perturbable(self, sample: Dict[str, Any]) -> bool:
        """检查样本是否可以被扰动"""
        try:
            # 提取样本信息
            question = sample.get('generated_question', sample.get('question', ''))
            context = sample.get('context', '')
            
            if not question or not context:
                return False
            
            # 检查每种扰动器是否能够扰动
            perturbable_count = 0
            for perturber_name in ['term', 'year', 'trend']:
                if perturber_name in self.perturbers:
                    perturber = self.perturbers[perturber_name]
                    perturbations = perturber.perturb(context)
                    if perturbations and len(perturbations) > 0:
                        perturbable_count += 1
            
            # 至少有一种扰动器能够扰动
            return perturbable_count > 0
            
        except Exception as e:
            print(f"⚠️ 检查样本扰动性时出错: {e}")
            return False
    
    def _is_sample_perturbable_relaxed(self, sample: Dict[str, Any]) -> bool:
        """宽松检查样本是否可以被扰动（用于补充样本）"""
        try:
            # 提取样本信息
            question = sample.get('generated_question', sample.get('question', ''))
            context = sample.get('context', '')
            
            if not question or not context:
                return False
            
            # 检查文本长度是否足够
            if len(context) < 10:
                return False
            
            # 检查是否包含可扰动的内容
            has_numbers = any(char.isdigit() for char in context)
            has_years = any(year in context for year in ['2020', '2021', '2022', '2023', '2024', '2025'])
            has_financial_terms = any(term in context for term in ['股票', '收益', '利润', '收入', '成本', '价格', '市场'])
            
            return has_numbers or has_years or has_financial_terms
            
        except Exception as e:
            print(f"⚠️ 宽松检查样本扰动性时出错: {e}")
            return False
    
    def _is_perturbation_successful(self, result: PerturbationResult) -> bool:
        """检查扰动是否成功"""
        try:
            # 检查扰动详情
            if not result.perturbation_detail:
                return False
            
            # 检查原始文本和扰动后文本是否不同
            original_text = result.perturbation_detail.original_text
            perturbed_text = result.perturbation_detail.perturbed_text
            
            if original_text == perturbed_text:
                print(f"⚠️ 扰动前后文本相同，扰动失败")
                return False
            
            # 检查扰动后答案是否生成成功
            if not result.perturbed_answer or result.perturbed_answer.startswith("扰动后答案生成失败"):
                print(f"⚠️ 扰动后答案生成失败")
                return False
            
            # 检查答案是否有意义的变化
            if len(result.perturbed_answer) < 10:
                print(f"⚠️ 扰动后答案过短")
                return False
            
            print(f"✅ 扰动成功: 原始文本长度 {len(original_text)}, 扰动后文本长度 {len(perturbed_text)}")
            return True
            
        except Exception as e:
            print(f"⚠️ 检查扰动成功性时出错: {e}")
            return False
    
    def apply_perturbation(self, text: str, perturber_name: str) -> Tuple[str, PerturbationDetail]:
        """应用扰动到文本"""
        if perturber_name not in self.perturbers:
            raise ValueError(f"未知的扰动器: {perturber_name}")
        
        perturber = self.perturbers[perturber_name]
        perturbations = perturber.perturb(text)
        
        if not perturbations:
            # 如果没有扰动，返回原始文本
            return text, PerturbationDetail(
                original_text=text,
                perturbed_text=text,
                perturbation_type=perturber_name
            )
        
        # 使用第一个扰动结果
        perturbation = perturbations[0]
        perturbed_text = perturbation.get('perturbed_text', text)
        
        detail = PerturbationDetail(
            original_text=text,
            perturbed_text=perturbed_text,
            perturbation_type=perturber_name
        )
        
        return perturbed_text, detail
    
    def get_perturbed_answer(self, question: str, context: str, perturber_name: str) -> Tuple[str, PerturbationDetail]:
        """获取扰动后的答案"""
        # 对context应用扰动
        perturbed_context, perturbation_detail = self.apply_perturbation(context, perturber_name)
        
        # 使用扰动后的context生成答案
        try:
            # 使用RAG系统适配器进行检索和生成
            rag_output = self.rag_system.get_ranked_documents_for_evaluation(
                query=question,
                top_k=10,
                mode="reranker",
                use_prefilter=True
            )
            
            # 构建扰动后的上下文
            if rag_output:
                # 对检索到的文档内容应用扰动
                retrieved_contexts = []
                for i, result in enumerate(rag_output[:3]):
                    content = result.get('content', '')
                    if content:
                        # 对检索到的内容应用扰动
                        perturber = self.perturbers.get(perturber_name)
                        if perturber:
                            perturbations = perturber.perturb(content)
                            if perturbations and len(perturbations) > 0:
                                perturbed_content = perturbations[0].get('perturbed_text', content)
                                retrieved_contexts.append(f"文档{i+1}: {perturbed_content}")
                            else:
                                retrieved_contexts.append(f"文档{i+1}: {content}")
                        else:
                            retrieved_contexts.append(f"文档{i+1}: {content}")
                
                if retrieved_contexts:
                    combined_context = "\n\n".join(retrieved_contexts)
                    # 使用RAG系统生成答案
                    perturbed_answer = self._generate_answer_with_rag(question, combined_context)
                else:
                    # 如果没有检索到文档，使用扰动后的原始context
                    perturbed_answer = self._generate_answer_with_rag(question, perturbed_context)
            else:
                # 如果检索失败，使用扰动后的原始context
                perturbed_answer = self._generate_answer_with_rag(question, perturbed_context)
                
        except Exception as e:
            print(f"⚠️ 生成扰动后答案失败: {e}")
            perturbed_answer = f"扰动后答案生成失败: {perturbed_context[:100]}..."
        
        return perturbed_answer, perturbation_detail
    
    def _init_generator(self):
        """初始化LLM生成器"""
        try:
            from xlm.components.generator.local_llm_generator import LocalLLMGenerator
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU内存清理完成")
            
            generator = LocalLLMGenerator(
                model_name=self.config.generator.model_name,
                cache_dir=self.config.generator.cache_dir,
                device=self.config.generator.device,
                use_quantization=self.config.generator.use_quantization,
                quantization_type=self.config.generator.quantization_type,
                use_flash_attention=self.config.generator.use_flash_attention
            )
            
            print(f"✅ LLM生成器初始化完成: {self.config.generator.model_name}")
            return generator
            
        except Exception as e:
            print(f"❌ LLM生成器初始化失败: {e}")
            return None
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            from langdetect import detect
            lang = detect(text)
            # 检查是否包含中文字符
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
            
            # 如果包含中文字符且中文比例超过30%，或者langdetect检测为中文，则认为是中文
            if chinese_chars > 0 and (chinese_chars / total_chars > 0.3 or lang.startswith('zh')):
                return 'zh'
            else:
                return 'en'
        except ImportError:
            # 如果langdetect失败，使用字符检测
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            return 'zh' if chinese_chars > 0 else 'en'
        except Exception as e:
            print(f"语言检测失败: {e}，使用字符检测")
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            return 'zh' if chinese_chars > 0 else 'en'
    
    def _should_enable_stock_prediction(self, question: str) -> bool:
        """检测是否应该启用股票预测模式"""
        # 移除所有股票预测检测逻辑，不再基于问题内容检测
        return False
    
    def _build_stock_prediction_instruction(self, question: str) -> str:
        """构建股票预测instruction"""
        return f"请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。\n\n请严格按照以下格式输出：\n这个股票的下月最终收益结果是:'涨/跌',上涨/下跌概率:极大/较大/中上/一般\n\n问题：{question}"
    
    def _generate_answer_with_rag(self, question: str, context: str) -> str:
        """使用RAG系统生成答案"""
        try:
            if self.generator is None:
                return f"生成器未初始化，使用上下文: {context[:100]}..."
            
            # 检测语言
            language = self._detect_language(question)
            print(f"检测到的语言: {language}")
            
            # 检测是否需要启用股票预测模式
            stock_prediction_mode = self._should_enable_stock_prediction(question)
            
            if language == 'zh' and stock_prediction_mode:
                print("🔮 检测到中文股票相关问题，启用股票预测模式")
                # 构建股票预测instruction
                instruction = self._build_stock_prediction_instruction(question)
                print(f"📋 股票预测instruction: {instruction[:100]}...")
                
                # 使用多阶段中文模板
                try:
                    answer = self.generator.generate_hybrid_answer(
                        question=question,
                        table_context=context,
                        text_context=context,
                        hybrid_decision="multi_stage_chinese"
                    )
                    print("✅ 使用多阶段中文模板生成答案")
                except Exception as e:
                    print(f"⚠️ 多阶段中文模板失败: {e}")
                    # 回退到股票预测模式
                    prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n{instruction}\n\n回答："
                    responses = self.generator.generate([prompt])
                    if responses and len(responses) > 0:
                        answer = responses[0]
                    else:
                        answer = f"股票预测生成失败: {context[:100]}..."
                
            elif language == 'zh':
                print("📝 检测到中文查询，使用多阶段中文模板")
                try:
                    answer = self.generator.generate_hybrid_answer(
                        question=question,
                        table_context=context,
                        text_context=context,
                        hybrid_decision="multi_stage_chinese"
                    )
                    print("✅ 使用多阶段中文模板生成答案")
                except Exception as e:
                    print(f"⚠️ 多阶段中文模板失败: {e}")
                    # 回退到简单prompt
                    prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}\n\n回答："
                    responses = self.generator.generate([prompt])
                    if responses and len(responses) > 0:
                        answer = responses[0]
                    else:
                        answer = f"生成失败: {context[:100]}..."
            
            else:
                print("📝 检测到英文查询，使用标准模板")
                # 构建标准prompt
                prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
                
                # 使用LLM生成器生成答案
                responses = self.generator.generate([prompt])
                
                if responses and len(responses) > 0:
                    answer = responses[0]
                else:
                    answer = f"Generation failed: {context[:100]}..."
            
            # 清理答案
            answer = self._clean_response(answer)
            return answer
            
        except Exception as e:
            print(f"⚠️ RAG答案生成失败: {e}")
            return f"答案生成失败: {context[:50]}..."
    
    def _clean_response(self, response: str) -> str:
        """清理生成的响应"""
        if not response:
            return ""
        
        # 移除多余的空白字符
        response = response.strip()
        
        # 如果响应太长，截取前500个字符
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response
    
    def run_single_experiment(self, sample: Dict[str, Any], perturber_name: str) -> PerturbationResult:
        """运行单个扰动实验"""
        print(f"🔬 运行扰动实验: {perturber_name}")
        
        # 提取样本信息
        sample_id = sample.get('id', f"sample_{time.time()}")
        # 优先使用 generated_question，如果没有则使用 question
        question = sample.get('generated_question', sample.get('question', ''))
        context = sample.get('context', '')
        expected_answer = sample.get('expected_answer', '')
        
        print(f"  问题: {question[:100]}...")
        print(f"  上下文长度: {len(context)} 字符")
        
        # 步骤1: 生成原始答案
        print("📝 生成原始答案...")
        try:
            # 使用RAG系统适配器进行检索和生成
            rag_output = self.rag_system.get_ranked_documents_for_evaluation(
                query=question,
                top_k=10,
                mode="reranker",
                use_prefilter=True
            )
            
            if rag_output:
                # 构建原始上下文
                retrieved_contexts = []
                for i, result in enumerate(rag_output[:3]):
                    content = result.get('content', '')
                    if content:
                        retrieved_contexts.append(f"文档{i+1}: {content}")
                
                if retrieved_contexts:
                    combined_context = "\n\n".join(retrieved_contexts)
                    original_answer = self._generate_answer_with_rag(question, combined_context)
                else:
                    original_answer = self._generate_answer_with_rag(question, context)
            else:
                original_answer = self._generate_answer_with_rag(question, context)
                
        except Exception as e:
            print(f"⚠️ 生成原始答案失败: {e}")
            original_answer = f"原始答案生成失败: {context[:100]}..."
        
        print(f"  原始答案: {original_answer[:200]}...")
        
        # 步骤2: 应用扰动并生成扰动后答案
        print(f"🔄 应用 {perturber_name} 扰动...")
        perturbed_answer, perturbation_detail = self.get_perturbed_answer(question, context, perturber_name)
        print(f"  扰动后答案: {perturbed_answer[:200]}...")
        
        # 步骤3: 创建结果对象
        result = PerturbationResult(
            sample_id=sample_id,
            question=question,
            context=context,
            expected_answer=expected_answer,
            perturber_name=perturber_name,
            perturbation_detail=perturbation_detail,
            original_answer=original_answer,
            perturbed_answer=perturbed_answer
        )
        
        print(f"✅ 扰动实验完成: {perturber_name}")
        return result
    
    def run_batch_experiments(self, samples: List[Dict[str, Any]], perturber_names: Optional[List[str]] = None, dataset_path: Optional[str] = None) -> List[PerturbationResult]:
        """批量运行扰动实验 - 每个扰动器7个样本，总共21个实验，如果不够则补充样本"""
        if perturber_names is None:
            perturber_names = ['term', 'year', 'trend']
        
        results = []
        target_per_perturber = 7  # 每个扰动器目标7个样本
        total_target = len(perturber_names) * target_per_perturber  # 总共21个实验
        
        print(f"🚀 开始批量扰动实验: 目标 {len(perturber_names)} 种扰动器 × {target_per_perturber} 个样本/扰动器 = {total_target} 个实验")
        
        current_experiment = 0
        available_samples = samples.copy()  # 可用样本池
        
        for perturber_name in perturber_names:
            print(f"\n🔬 处理 {perturber_name} 扰动器...")
            
            perturber_results = 0
            perturber_samples_used = 0
            
            # 尝试为当前扰动器生成7个样本
            while perturber_results < target_per_perturber and available_samples:
                current_experiment += 1
                perturber_samples_used += 1
                
                # 选择一个样本
                sample = available_samples.pop(0)
                print(f"\n📊 实验 {current_experiment}/{total_target}: {perturber_name} 扰动器 - 样本 {perturber_results + 1}/{target_per_perturber}")
                
                try:
                    result = self.run_single_experiment(sample, perturber_name)
                    
                    # 检查扰动是否成功
                    if self._is_perturbation_successful(result):
                        results.append(result)
                        perturber_results += 1
                        print(f"✅ 实验 {current_experiment}/{total_target} 完成 ({perturber_name}: {perturber_results}/{target_per_perturber})")
                        
                        # 每完成一个实验就保存一次
                        self.save_results_incremental(results, "perturbation_results_incremental.json")
                    else:
                        print(f"⚠️ 实验 {current_experiment}/{total_target} 扰动失败，跳过此样本")
                        # 将样本放回池中，尝试其他样本
                        available_samples.append(sample)
                    
                except Exception as e:
                    print(f"❌ 实验 {current_experiment}/{total_target} 失败: {str(e)}")
                    logger.error(f"实验失败: {str(e)}", exc_info=True)
                    # 失败时减少计数，因为样本被消耗了但没有成功
                    current_experiment -= 1
            
            print(f"✅ {perturber_name} 扰动器完成: {perturber_results}/{target_per_perturber} 个实验")
            
            # 如果样本不够，记录但继续
            if perturber_results < target_per_perturber:
                print(f"⚠️ {perturber_name} 扰动器样本不足: {perturber_results}/{target_per_perturber}")
        
        print(f"\n🎉 批量扰动实验完成: {len(results)}/{total_target} 个实验成功")
        print(f"📊 实验结果统计:")
        for perturber_name in perturber_names:
            perturber_count = sum(1 for r in results if r.perturber_name == perturber_name)
            print(f"  {perturber_name}: {perturber_count} 个实验")
        
        return results
    
    def save_results_incremental(self, results: List[PerturbationResult], filename: str):
        """增量保存结果"""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            result_dict = result.to_dict()
            serializable_results.append(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 统计每个扰动器的数量
        perturber_counts = {}
        for result in results:
            perturber_name = result.perturber_name
            perturber_counts[perturber_name] = perturber_counts.get(perturber_name, 0) + 1
        
        print(f"💾 增量保存结果到: {output_path} (总计: {len(results)} 个实验)")
        for perturber_name, count in perturber_counts.items():
            print(f"  {perturber_name}: {count} 个实验")
    
    def save_results(self, results: List[PerturbationResult], output_dir: str = 'perturbation_results'):
        """保存实验结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"perturbation_results_{timestamp}.json"
        filepath = output_path / filename
        
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            result_dict = result.to_dict()
            serializable_results.append(result_dict)
        
        # 保存结果
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到: {filepath}")
        return filepath
    
    def analyze_results(self, results: List[PerturbationResult]) -> Dict[str, Any]:
        """分析实验结果"""
        print("📊 分析实验结果...")
        
        if not results:
            print("⚠️ 没有结果可分析")
            return {}
        
        # 按扰动器分组
        perturber_results = {}
        for result in results:
            perturber_name = result.perturber_name
            if perturber_name not in perturber_results:
                perturber_results[perturber_name] = []
            perturber_results[perturber_name].append(result)
        
        # 计算统计信息
        analysis = {
            'total_experiments': len(results),
            'perturber_stats': {},
            'overall_stats': {
                'avg_original_answer_length': sum(len(r.original_answer) for r in results) / len(results),
                'avg_perturbed_answer_length': sum(len(r.perturbed_answer) for r in results) / len(results),
                'answer_change_rate': sum(1 for r in results if r.original_answer != r.perturbed_answer) / len(results)
            }
        }
        
        # 按扰动器分析
        for perturber_name, perturber_results_list in perturber_results.items():
            if not perturber_results_list:
                continue
                
            analysis['perturber_stats'][perturber_name] = {
                'count': len(perturber_results_list),
                'avg_original_answer_length': sum(len(r.original_answer) for r in perturber_results_list) / len(perturber_results_list),
                'avg_perturbed_answer_length': sum(len(r.perturbed_answer) for r in perturber_results_list) / len(perturber_results_list),
                'answer_change_rate': sum(1 for r in perturber_results_list if r.original_answer != r.perturbed_answer) / len(perturber_results_list)
            }
        
        # 打印分析结果
        print("\n📊 实验结果分析:")
        print(f"  总实验数: {analysis['total_experiments']}")
        print(f"  平均原始答案长度: {analysis['overall_stats']['avg_original_answer_length']:.1f} 字符")
        print(f"  平均扰动后答案长度: {analysis['overall_stats']['avg_perturbed_answer_length']:.1f} 字符")
        print(f"  答案变化率: {analysis['overall_stats']['answer_change_rate']:.2%}")
        
        for perturber_name, stats in analysis['perturber_stats'].items():
            print(f"\n  {perturber_name} 扰动器:")
            print(f"    实验数: {stats['count']}")
            print(f"    平均原始答案长度: {stats['avg_original_answer_length']:.1f} 字符")
            print(f"    平均扰动后答案长度: {stats['avg_perturbed_answer_length']:.1f} 字符")
            print(f"    答案变化率: {stats['answer_change_rate']:.2%}")
        
        return analysis
    
    def run_integrated_experiment(self, dataset_path: str, num_samples: int = 21, output_dir: str = 'perturbation_results'):
        """运行完整的集成实验"""
        print("🚀 开始集成扰动实验...")
        
        # 步骤1: 选择样本
        samples = self.select_perturbation_samples(dataset_path, num_samples)
        
        # 步骤2: 运行批量实验
        results = self.run_batch_experiments(samples, dataset_path=dataset_path)
        
        # 步骤3: 保存结果
        output_file = self.save_results(results, output_dir)
        
        # 步骤4: 分析结果
        analysis = self.analyze_results(results)
        
        # 保存分析结果
        analysis_file = Path(output_dir) / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 集成实验完成!")
        print(f"  结果文件: {output_file}")
        print(f"  分析文件: {analysis_file}")
        
        return results, analysis

def main():
    """主函数"""
    print("🔬 RAG扰动实验启动...")
    
    # 创建实验实例
    experiment = PerturbationExperiment()
    
    # 运行实验 - 21个样本（每个扰动器7个）
    dataset_path = "data/alphafin/alphafin_eval_samples_updated.jsonl"
    results, analysis = experiment.run_integrated_experiment(
        dataset_path=dataset_path,
        num_samples=21,
        output_dir='perturbation_results'
    )
    
    print("✅ 实验完成!")

if __name__ == "__main__":
    main() 