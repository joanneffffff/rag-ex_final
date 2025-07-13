#!/usr/bin/env python3
"""
RAG系统多语言端到端测试脚本
支持分别测试中文和英文数据集，支持数据采样
每10个数据保存一次原始数据
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np
import jieba
import random

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入RAG系统组件
from xlm.ui.optimized_rag_ui import OptimizedRagUI
from config.parameters import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e2e_test_multilingual.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MultilingualRagSystemAdapter:
    """
    多语言RAG系统适配器，支持中文和英文数据集的评估
    """
    
    # 类级别的缓存，避免重复初始化
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """单例模式，确保只有一个RAG系统实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, enable_reranker: bool = True, enable_stock_prediction: bool = False):
        """
        初始化多语言RAG系统适配器
        
        Args:
            enable_reranker: 是否启用重排序器
            enable_stock_prediction: 是否启用股票预测模式
        """
        # 如果已经初始化过，直接返回
        if self._initialized:
            return
            
        self.enable_reranker = enable_reranker
        self.enable_stock_prediction = enable_stock_prediction
        self.rag_ui = None
        self.initialized = False
        
    def initialize(self):
        """初始化RAG系统"""
        # 如果已经初始化过，直接返回
        if self._initialized:
            logger.info("✅ RAG系统已经初始化，跳过重复初始化")
            return
            
        try:
            logger.info("🔄 正在初始化多语言RAG系统...")
            
            # 初始化RAG UI系统（会自动调用_init_components）
            self.rag_ui = OptimizedRagUI(
                enable_reranker=self.enable_reranker,
                use_existing_embedding_index=True  # 使用现有索引以加快测试
            )
            
            # 不需要手动调用_init_components()，因为OptimizedRagUI在初始化时会自动调用
            
            self.initialized = True
            self._initialized = True  # 设置类级别的初始化标志
            logger.info("✅ 多语言RAG系统初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 多语言RAG系统初始化失败: {e}")
            raise
    
    def process_query(self, query: str, datasource: str = "auto", enable_stock_prediction_override: Optional[bool] = None) -> Dict[str, Any]:
        """
        处理用户查询，返回完整的RAG系统响应
        
        Args:
            query: 用户查询
            datasource: 数据源（auto表示自动检测）
            enable_stock_prediction_override: 临时覆盖股票预测模式设置
            
        Returns:
            包含答案和性能指标的字典
        """
        if not self.initialized:
            raise RuntimeError("RAG系统未初始化，请先调用initialize()")
        
        start_time = time.time()
        
        try:
            # 检查RAG UI是否已初始化
            if self.rag_ui is None:
                raise RuntimeError("RAG UI未初始化")
            
            # 记录生成开始时间
            generation_start_time = time.time()
            
            # 使用与RAG系统完全相同的处理逻辑
            # 如果提供了覆盖参数，使用覆盖值；否则使用默认设置
            stock_prediction_checkbox = self.enable_stock_prediction
            if enable_stock_prediction_override is not None:
                stock_prediction_checkbox = enable_stock_prediction_override
            
            answer, html_content = self.rag_ui._process_question(
                question=query,
                datasource=datasource,
                reranker_checkbox=self.enable_reranker,
                stock_prediction_checkbox=stock_prediction_checkbox
            )
            
            # 记录生成结束时间和计算Token数
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time
            
            # 计算Token数（需要检测语言）
            try:
                from langdetect import detect
                language = detect(query)
                is_chinese = language.startswith('zh')
            except:
                # 如果语言检测失败，根据字符判断
                chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
                is_chinese = chinese_chars > 0
            
            token_count = count_tokens(answer, "chinese" if is_chinese else "english")
            
            # 提取摘要和智能选择的上下文（从HTML内容中解析）
            summary_context = self._extract_summary_and_context(html_content)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "query": query,
                "answer": answer,
                "html_content": html_content,
                "summary_context": summary_context,  # 新增：摘要和智能选择的上下文
                "processing_time": processing_time,
                "generation_time": generation_time,
                "token_count": token_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ 查询处理失败: {e}")
            return {
                "query": query,
                "answer": f"处理失败: {str(e)}",
                "html_content": "",
                "summary_context": "",
                "processing_time": time.time() - start_time,
                "generation_time": 0.0,
                "token_count": 0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_summary_and_context(self, html_content: str) -> str:
        """
        从HTML内容中提取摘要和智能选择的上下文
        
        Args:
            html_content: HTML格式的上下文内容
            
        Returns:
            提取的摘要和上下文
        """
        if not html_content:
            return ""
        
        try:
            # 简单的文本提取，移除HTML标签
            import re
            # 移除HTML标签
            text_content = re.sub(r'<[^>]+>', '', html_content)
            # 移除多余的空白字符
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # 限制长度，避免文件过大
            if len(text_content) > 2000:
                text_content = text_content[:2000] + "..."
            
            return text_content
        except Exception as e:
            logger.warning(f"提取上下文失败: {e}")
            return html_content[:1000] if html_content else ""


def normalize_answer_chinese(s: str) -> str:
    """标准化中文答案"""
    if not s:
        return ""
    
    # 移除"解析"及其后面的内容
    import re
    # 查找"解析"的位置，移除它及其后面的所有内容
    parse_index = s.find("解析")
    if parse_index != -1:
        s = s[:parse_index]
    
    s = ' '.join(s.split())
    s = re.sub(r'[^\u4e00-\u9fff\w\s]', '', s)
    return s.strip()


def normalize_answer_english(s: str) -> str:
    """标准化英文答案"""
    if not s:
        return ""
    s = ' '.join(s.split())
    import re
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip().lower()


def get_tokens_chinese(s: str) -> List[str]:
    """使用jieba分词获取中文token列表"""
    return list(jieba.cut(s))


def get_tokens_english(s: str) -> List[str]:
    """获取英文token列表"""
    return s.split()


def calculate_f1_score(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """计算F1-score，支持中文和英文"""
    if language == "chinese":
        pred_tokens = set(get_tokens_chinese(normalize_answer_chinese(prediction)))
        gt_tokens = set(get_tokens_chinese(normalize_answer_chinese(ground_truth)))
    else:
        pred_tokens = set(get_tokens_english(normalize_answer_english(prediction)))
        gt_tokens = set(get_tokens_english(normalize_answer_english(ground_truth)))
    
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, ground_truth: str, language: str = "chinese") -> float:
    """计算Exact Match，支持中文和英文"""
    if language == "chinese":
        pred_normalized = normalize_answer_chinese(prediction)
        gt_normalized = normalize_answer_chinese(ground_truth)
    else:
        pred_normalized = normalize_answer_english(prediction)
        gt_normalized = normalize_answer_english(ground_truth)
    
    return 1.0 if pred_normalized == gt_normalized else 0.0


def count_tokens(text: str, language: str = "chinese") -> int:
    """
    计算文本的Token数量
    
    Args:
        text: 要计算的文本
        language: 语言类型
        
    Returns:
        Token数量
    """
    if not text:
        return 0
    
    if language == "chinese":
        # 中文使用jieba分词
        tokens = get_tokens_chinese(text)
    else:
        # 英文使用空格分词
        tokens = get_tokens_english(text)
    
    return len(tokens)


def is_stock_prediction_query(test_item: Dict[str, Any]) -> bool:
    """
    检测数据项是否为股票预测指令
    
    Args:
        test_item: 测试数据项
        
    Returns:
        是否为股票预测查询
    """
    # 检查instruction字段
    instruction = test_item.get("instruction", "")
    
    # 只有当instruction等于特定字符串时，才认为是股票预测查询
    stock_prediction_instruction = "请根据下方提供的该股票相关研报与数据，对该股票的下个月的涨跌，进行预测，请给出明确的答案，\"涨\" 或者 \"跌\"。同时给出这个股票下月的涨跌概率，分别是:极大，较大，中上，一般。"
    
    if instruction.strip() == stock_prediction_instruction:
        return True
    
    return False


def load_test_dataset(data_path: str, sample_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    加载测试数据集
    
    Args:
        data_path: 数据文件路径
        sample_size: 采样数量，None表示使用全部数据
        
    Returns:
        (测试数据列表, 语言)
    """
    logger.info(f"📂 加载测试数据集: {data_path}")
    
    # 检测语言
    if "alphafin" in data_path.lower():
        language = "chinese"
    elif "tatqa" in data_path.lower():
        language = "english"
    else:
        language = "unknown"
    
    logger.info(f"🌍 检测到语言: {language}")
    
    dataset = []
    
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    
    if sample_size and sample_size < len(dataset):
        random.seed(42)  # 固定随机种子以确保可重复性
        dataset = random.sample(dataset, sample_size)
        logger.info(f"📊 随机采样 {sample_size} 个样本")
    
    logger.info(f"✅ 加载完成，共 {len(dataset)} 个测试样本")
    return dataset, language


def save_raw_data_batch(raw_data_batch: List[Dict[str, Any]], data_path: str, batch_num: int):
    """
    保存原始数据批次
    
    Args:
        raw_data_batch: 原始数据批次
        data_path: 数据文件路径
        batch_num: 批次编号
    """
    # 从数据路径生成输出文件名
    data_name = Path(data_path).stem
    output_dir = f"raw_data_{data_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 添加时间戳到文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = Path(output_dir) / f"batch_{batch_num:03d}_{timestamp}.json"
    
    # 构建包含时间戳的数据结构
    output_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_num": batch_num,
        "data_path": data_path,
        "total_samples": len(raw_data_batch),
        "successful_samples": sum(1 for item in raw_data_batch if item.get("success", False)),
        "failed_samples": sum(1 for item in raw_data_batch if not item.get("success", True)),
        "data": raw_data_batch
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📁 保存原始数据批次 {batch_num} 到: {output_file}")
    logger.info(f"⏰ 时间戳: {output_data['timestamp']}")
    logger.info(f"📊 批次统计: 成功 {output_data['successful_samples']} 个, 失败 {output_data['failed_samples']} 个")


def test_single_dataset(
    data_path: str,
    sample_size: Optional[int] = None,
    enable_reranker: bool = True,
    enable_stock_prediction: bool = False
) -> Dict[str, Any]:
    """
    测试单个数据集
    
    Args:
        data_path: 数据文件路径
        sample_size: 采样数量
        enable_reranker: 是否启用重排序器
        enable_stock_prediction: 是否启用股票预测模式
        
    Returns:
        测试结果摘要
    """
    logger.info(f"🚀 开始测试数据集: {data_path}")
    logger.info(f"重排序器: {'启用' if enable_reranker else '禁用'}")
    logger.info(f"股票预测: {'启用' if enable_stock_prediction else '禁用'}")
    
    # 2. 加载测试数据
    dataset, language = load_test_dataset(data_path, sample_size)
    
    # 3. 预检测股票预测查询，决定是否启用股票预测模式
    stock_prediction_queries = set()
    should_enable_stock_prediction = enable_stock_prediction
    
    if "alphafin" in data_path.lower() and language == "chinese":
        for i, test_item in enumerate(dataset):
            if is_stock_prediction_query(test_item):
                stock_prediction_queries.add(i)
        
        if stock_prediction_queries and not enable_stock_prediction:
            logger.info(f"🔮 检测到 {len(stock_prediction_queries)} 个股票预测查询，将自动启用股票预测模式")
            should_enable_stock_prediction = True
    
    # 1. 初始化RAG系统适配器（只初始化一次，使用最终确定的配置）
    rag_adapter = MultilingualRagSystemAdapter(
        enable_reranker=enable_reranker,
        enable_stock_prediction=should_enable_stock_prediction
    )
    rag_adapter.initialize()
    
    # 4. 运行测试
    results = []
    total_processing_time = 0.0
    raw_data_batch = []
    batch_num = 1
    
    logger.info(f"🔄 开始处理测试样本...")
    
    for i, test_item in enumerate(tqdm(dataset, desc=f"处理样本")):
        # 获取查询和标准答案
        # 优先使用generated_question，与RAG系统保持一致
        query = test_item.get("generated_question", "") or test_item.get("query", "") or test_item.get("question", "")
        ground_truth = test_item.get("answer", "") or test_item.get("expected_answer", "")
        
        if not query:
            logger.warning(f"⚠️ 样本 {i} 缺少查询，跳过")
            continue
        
        # 检查是否为股票预测查询（使用预检测结果）
        auto_stock_prediction = i in stock_prediction_queries
        if auto_stock_prediction:
            logger.info(f"🔮 处理股票预测查询: {query[:50]}...")
        
        # 处理查询（使用统一的适配器）
        # 如果当前样本是股票预测查询，动态启用股票预测模式
        if auto_stock_prediction:
            result = rag_adapter.process_query(query, enable_stock_prediction_override=True)
        else:
            result = rag_adapter.process_query(query, enable_stock_prediction_override=False)
        
        # 计算评估指标
        if result["success"] and ground_truth:
            f1_score = calculate_f1_score(result["answer"], ground_truth, language)
            exact_match = calculate_exact_match(result["answer"], ground_truth, language)
        else:
            f1_score = 0.0
            exact_match = 0.0
        
        # 记录结果
        test_result = {
            "sample_id": i,
            "query": query,
            "ground_truth": ground_truth,
            "predicted_answer": result["answer"],
            "f1_score": f1_score,
            "exact_match": exact_match,
            "processing_time": result["processing_time"],
            "generation_time": result.get("generation_time", 0.0),
            "token_count": result.get("token_count", 0),
            "success": result["success"],
            "language": language,
            "auto_stock_prediction": auto_stock_prediction
        }
        
        if not result["success"]:
            test_result["error"] = result.get("error", "未知错误")
        
        results.append(test_result)
        total_processing_time += result["processing_time"]
        
        # 构建原始数据记录
        raw_data_record = {
            "sample_id": i,
            "query": query,
            "summary_context": result.get("summary_context", ""),  # 使用摘要和智能选择的上下文
            "answer": result["answer"],
            "expected_answer": ground_truth,
            "em": exact_match,
            "f1": f1_score,
            "processing_time": result["processing_time"],
            "generation_time": result.get("generation_time", 0.0),
            "token_count": result.get("token_count", 0),
            "success": result["success"],
            "language": language,
            "auto_stock_prediction": auto_stock_prediction
        }
        
        if not result["success"]:
            raw_data_record["error"] = result.get("error", "未知错误")
        
        raw_data_batch.append(raw_data_record)
        
        # 每处理10个样本保存一次原始数据
        if len(raw_data_batch) >= 10:
            save_raw_data_batch(raw_data_batch, data_path, batch_num)
            raw_data_batch = []
            batch_num += 1
        
        # 每处理10个样本输出一次进度
        if (i + 1) % 10 == 0:
            avg_time = total_processing_time / (i + 1)
            avg_f1 = np.mean([r["f1_score"] for r in results])
            avg_em = np.mean([r["exact_match"] for r in results])
            logger.info(f"📊 进度: {i+1}/{len(dataset)}, 平均F1: {avg_f1:.4f}, 平均EM: {avg_em:.4f}, 平均时间: {avg_time:.2f}s")
    
    # 保存剩余的原始数据
    if raw_data_batch:
        save_raw_data_batch(raw_data_batch, data_path, batch_num)
    
    # 计算统计
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        avg_f1 = np.mean([r["f1_score"] for r in successful_results])
        avg_em = np.mean([r["exact_match"] for r in successful_results])
        avg_time = np.mean([r["processing_time"] for r in successful_results])
        total_time = sum([r["processing_time"] for r in successful_results])
        avg_generation_time = np.mean([r["generation_time"] for r in successful_results])
        avg_token_count = np.mean([r["token_count"] for r in successful_results])
        total_tokens = sum([r["token_count"] for r in successful_results])
    else:
        avg_f1 = avg_em = avg_time = total_time = avg_generation_time = avg_token_count = total_tokens = 0.0
    
    # 统计股票预测检测情况
    stock_prediction_detected = sum(1 for r in results if r.get("auto_stock_prediction", False))
    
    # 生成结果摘要
    summary = {
        "data_path": data_path,
        "language": language,
        "total_samples": len(dataset),
        "successful_samples": len(successful_results),
        "success_rate": len(successful_results) / len(dataset) if dataset else 0.0,
        "average_f1_score": avg_f1,
        "average_exact_match": avg_em,
        "average_processing_time": avg_time,
        "total_processing_time": total_time,
        "average_generation_time": avg_generation_time,
        "average_token_count": avg_token_count,
        "total_tokens": total_tokens,
        "enable_reranker": enable_reranker,
        "enable_stock_prediction": should_enable_stock_prediction,
        "stock_prediction_detected": stock_prediction_detected,
        "detailed_results": results
    }
    
    # 输出摘要
    logger.info("🎉 数据集测试完成！")
    print_dataset_summary(summary)
    
    return summary


def print_dataset_summary(summary: Dict[str, Any]):
    """打印数据集测试摘要"""
    print("\n" + "="*80)
    print(f"🎯 数据集测试结果: {summary['data_path']}")
    print("="*80)
    
    print(f"📊 测试指标:")
    print(f"   数据路径: {summary['data_path']}")
    print(f"   语言: {summary['language']}")
    print(f"   总样本数: {summary['total_samples']}")
    print(f"   成功样本数: {summary['successful_samples']}")
    print(f"   成功率: {summary['success_rate']:.2%}")
    print(f"   平均F1-score: {summary['average_f1_score']:.4f}")
    print(f"   平均Exact Match: {summary['average_exact_match']:.4f}")
    print(f"   平均处理时间: {summary['average_processing_time']:.2f}秒")
    print(f"   总处理时间: {summary['total_processing_time']:.2f}秒")
    print(f"   平均生成时间: {summary['average_generation_time']:.2f}秒")
    print(f"   平均Token数: {summary['average_token_count']:.1f}")
    print(f"   总Token数: {summary['total_tokens']}")
    print(f"   重排序器: {'启用' if summary['enable_reranker'] else '禁用'}")
    print(f"   股票预测: {'启用' if summary['enable_stock_prediction'] else '禁用'}")
    print(f"   自动检测股票预测: {summary.get('stock_prediction_detected', 0)} 个")
    
    print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统单数据集端到端测试")
    parser.add_argument("--data_path", type=str, required=True,
                       help="测试数据文件路径")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="采样数量 (默认使用全部数据)")
    parser.add_argument("--disable_reranker", action="store_true",
                       help="禁用重排序器")
    parser.add_argument("--enable_stock_prediction", action="store_true",
                       help="启用股票预测模式")
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not Path(args.data_path).exists():
        print(f"❌ 数据文件不存在: {args.data_path}")
        return
    
    # 运行单数据集测试
    summary = test_single_dataset(
        data_path=args.data_path,
        sample_size=args.sample_size,
        enable_reranker=not args.disable_reranker,
        enable_stock_prediction=args.enable_stock_prediction
    )


if __name__ == "__main__":
    main() 