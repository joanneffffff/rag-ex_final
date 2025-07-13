#!/usr/bin/env python3
"""
RAG系统端到端测试脚本
模拟真实用户与RAG系统的完整交互流程，评估整体性能
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
        logging.FileHandler('e2e_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RagSystemAdapter:
    """
    RAG系统适配器，提供统一的接口来测试整个RAG系统
    """
    
    def __init__(self, enable_reranker: bool = True, enable_stock_prediction: bool = False):
        """
        初始化RAG系统适配器
        
        Args:
            enable_reranker: 是否启用重排序器
            enable_stock_prediction: 是否启用股票预测模式
        """
        self.enable_reranker = enable_reranker
        self.enable_stock_prediction = enable_stock_prediction
        self.rag_ui = None
        self.initialized = False
        
    def initialize(self):
        """初始化RAG系统"""
        try:
            logger.info("🔄 正在初始化RAG系统...")
            
            # 初始化RAG UI系统
            self.rag_ui = OptimizedRagUI(
                enable_reranker=self.enable_reranker,
                use_existing_embedding_index=True  # 使用现有索引以加快测试
            )
            
            # 初始化组件
            self.rag_ui._init_components()
            
            # 验证逻辑一致性
            if self.verify_rag_logic_consistency():
                self.initialized = True
                logger.info("✅ RAG系统初始化完成，逻辑一致性验证通过")
            else:
                logger.warning("⚠️ RAG系统初始化完成，但逻辑一致性验证失败")
                self.initialized = True  # 仍然允许使用，但记录警告
            
        except Exception as e:
            logger.error(f"❌ RAG系统初始化失败: {e}")
            raise
    
    def process_query(self, query: str, datasource: str = "auto") -> Dict[str, Any]:
        """
        处理用户查询，返回完整的RAG系统响应
        使用与RAG系统完全相同的逻辑
        
        Args:
            query: 用户查询
            datasource: 数据源（auto表示自动检测）
            
        Returns:
            包含答案和性能指标的字典
        """
        if not self.initialized:
            raise RuntimeError("RAG系统未初始化，请先调用initialize()")
        
        start_time = time.time()
        
        try:
            # 使用与RAG系统完全相同的处理逻辑
            # 直接调用RAG系统的_process_question方法，它会自动处理：
            # 1. 语言检测
            # 2. 根据语言选择处理方式（中文用多阶段检索，英文用传统RAG）
            # 3. 股票预测模式的处理
            # 4. 重排序器的处理
            answer, html_content = self.rag_ui._process_question(
                question=query,
                datasource=datasource,
                reranker_checkbox=self.enable_reranker,
                stock_prediction_checkbox=self.enable_stock_prediction
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 提取性能指标（这里需要根据实际RAG系统的日志来获取）
            # 由于RAG系统内部没有直接暴露这些指标，我们需要从日志中解析
            performance_metrics = self._extract_performance_metrics()
            
            return {
                "query": query,
                "answer": answer,
                "html_content": html_content,
                "processing_time": processing_time,
                "performance_metrics": performance_metrics,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ 查询处理失败: {e}")
            return {
                "query": query,
                "answer": f"处理失败: {str(e)}",
                "html_content": "",
                "processing_time": time.time() - start_time,
                "performance_metrics": {},
                "success": False,
                "error": str(e)
            }
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """
        从RAG系统日志中提取性能指标
        这里返回一个示例，实际实现需要根据RAG系统的日志格式来解析
        """
        return {
            "retrieval_time": 0.0,  # 检索时间
            "generation_time": 0.0,  # 生成时间
            "total_tokens": 0,  # 总token数
            "retrieved_docs": 0,  # 检索到的文档数
            "reranker_enabled": self.enable_reranker,
            "stock_prediction_enabled": self.enable_stock_prediction
        }
    
    def verify_rag_logic_consistency(self) -> bool:
        """
        验证RagSystemAdapter是否与RAG系统使用相同的逻辑
        """
        if not self.initialized:
            logger.warning("⚠️ RAG系统未初始化，无法验证逻辑一致性")
            return False
        
        try:
            # 检查RAG系统是否包含所有必要的组件
            required_components = [
                'retriever',
                'generator', 
                'reranker',
                'chinese_retrieval_system',
                'config'
            ]
            
            missing_components = []
            for component in required_components:
                if not hasattr(self.rag_ui, component):
                    missing_components.append(component)
            
            if missing_components:
                logger.warning(f"⚠️ 缺少RAG系统组件: {missing_components}")
                return False
            
            # 检查是否包含所有必要的方法
            required_methods = [
                '_process_question',
                '_unified_rag_processing_with_prompt',
                '_unified_rag_processing',
                '_generate_answer_with_context'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(self.rag_ui, method):
                    missing_methods.append(method)
            
            if missing_methods:
                logger.warning(f"⚠️ 缺少RAG系统方法: {missing_methods}")
                return False
            
            logger.info("✅ RAG系统逻辑一致性验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG系统逻辑一致性验证失败: {e}")
            return False


def normalize_answer_chinese(s: str) -> str:
    """
    标准化中文答案，用于计算F1-score和Exact Match
    """
    if not s:
        return ""
    
    # 移除"解析"及其后面的内容
    import re
    # 查找"解析"的位置，移除它及其后面的所有内容
    parse_index = s.find("解析")
    if parse_index != -1:
        s = s[:parse_index]
    
    # 移除多余的空白字符
    s = ' '.join(s.split())
    
    # 移除标点符号（保留中文标点）
    s = re.sub(r'[^\u4e00-\u9fff\w\s]', '', s)
    
    return s.strip()


def get_tokens_chinese(s: str) -> List[str]:
    """
    使用jieba分词获取中文token列表
    """
    return list(jieba.cut(s))


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    计算F1-score
    """
    pred_tokens = set(get_tokens_chinese(normalize_answer_chinese(prediction)))
    gt_tokens = set(get_tokens_chinese(normalize_answer_chinese(ground_truth)))
    
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    计算Exact Match
    """
    pred_normalized = normalize_answer_chinese(prediction)
    gt_normalized = normalize_answer_chinese(ground_truth)
    
    return 1.0 if pred_normalized == gt_normalized else 0.0


def load_test_dataset(data_path: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载测试数据集
    
    Args:
        data_path: 数据文件路径
        sample_size: 采样数量，None表示使用全部数据
        
    Returns:
        测试数据列表
    """
    logger.info(f"📂 加载测试数据集: {data_path}")
    
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
        import random
        random.seed(42)  # 固定随机种子以确保可重复性
        dataset = random.sample(dataset, sample_size)
        logger.info(f"📊 随机采样 {sample_size} 个样本")
    
    logger.info(f"✅ 加载完成，共 {len(dataset)} 个测试样本")
    return dataset


def run_e2e_test(
    data_path: str,
    output_path: str,
    sample_size: Optional[int] = None,
    enable_reranker: bool = True,
    enable_stock_prediction: bool = False
) -> Dict[str, Any]:
    """
    运行端到端测试
    
    Args:
        data_path: 测试数据文件路径
        output_path: 结果输出文件路径
        sample_size: 采样数量
        enable_reranker: 是否启用重排序器
        enable_stock_prediction: 是否启用股票预测模式
        
    Returns:
        测试结果摘要
    """
    logger.info("🚀 开始端到端测试")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"重排序器: {'启用' if enable_reranker else '禁用'}")
    logger.info(f"股票预测: {'启用' if enable_stock_prediction else '禁用'}")
    
    # 1. 加载测试数据
    test_dataset = load_test_dataset(data_path, sample_size)
    
    # 2. 初始化RAG系统适配器
    rag_adapter = RagSystemAdapter(
        enable_reranker=enable_reranker,
        enable_stock_prediction=enable_stock_prediction
    )
    rag_adapter.initialize()
    
    # 3. 运行测试
    results = []
    total_processing_time = 0.0
    
    logger.info("🔄 开始处理测试样本...")
    
    for i, test_item in enumerate(tqdm(test_dataset, desc="处理测试样本")):
        # 获取查询和标准答案
        query = test_item.get("query", "") or test_item.get("question", "") or test_item.get("generated_question", "")
        ground_truth = test_item.get("answer", "") or test_item.get("expected_answer", "")
        
        if not query:
            logger.warning(f"⚠️ 样本 {i} 缺少查询，跳过")
            continue
        
        # 处理查询
        result = rag_adapter.process_query(query)
        
        # 计算评估指标
        if result["success"] and ground_truth:
            f1_score = calculate_f1_score(result["answer"], ground_truth)
            exact_match = calculate_exact_match(result["answer"], ground_truth)
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
            "success": result["success"],
            "performance_metrics": result["performance_metrics"]
        }
        
        if not result["success"]:
            test_result["error"] = result.get("error", "未知错误")
        
        results.append(test_result)
        total_processing_time += result["processing_time"]
        
        # 每处理10个样本输出一次进度
        if (i + 1) % 10 == 0:
            avg_time = total_processing_time / (i + 1)
            avg_f1 = np.mean([r["f1_score"] for r in results])
            avg_em = np.mean([r["exact_match"] for r in results])
            logger.info(f"📊 进度: {i+1}/{len(test_dataset)}, 平均F1: {avg_f1:.4f}, 平均EM: {avg_em:.4f}, 平均时间: {avg_time:.2f}s")
    
    # 4. 计算总体统计
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        avg_f1 = np.mean([r["f1_score"] for r in successful_results])
        avg_em = np.mean([r["exact_match"] for r in successful_results])
        avg_time = np.mean([r["processing_time"] for r in successful_results])
        total_time = sum([r["processing_time"] for r in successful_results])
    else:
        avg_f1 = avg_em = avg_time = total_time = 0.0
    
    # 5. 生成测试摘要
    test_summary = {
        "test_config": {
            "data_path": data_path,
            "sample_size": sample_size,
            "enable_reranker": enable_reranker,
            "enable_stock_prediction": enable_stock_prediction
        },
        "overall_metrics": {
            "total_samples": len(test_dataset),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(test_dataset) if test_dataset else 0.0,
            "average_f1_score": avg_f1,
            "average_exact_match": avg_em,
            "average_processing_time": avg_time,
            "total_processing_time": total_time
        },
        "detailed_results": results
    }
    
    # 6. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)
    
    # 7. 输出摘要
    logger.info("🎉 端到端测试完成！")
    logger.info(f"📊 测试摘要:")
    logger.info(f"   总样本数: {len(test_dataset)}")
    logger.info(f"   成功样本数: {len(successful_results)}")
    logger.info(f"   成功率: {test_summary['overall_metrics']['success_rate']:.2%}")
    logger.info(f"   平均F1-score: {avg_f1:.4f}")
    logger.info(f"   平均Exact Match: {avg_em:.4f}")
    logger.info(f"   平均处理时间: {avg_time:.2f}秒")
    logger.info(f"   总处理时间: {total_time:.2f}秒")
    logger.info(f"📁 详细结果已保存到: {output_path}")
    
    return test_summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统端到端测试")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="测试数据文件路径 (jsonl或json格式)")
    parser.add_argument("--output_path", type=str, default="e2e_test_results.json",
                       help="结果输出文件路径")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="采样数量 (默认使用全部数据)")
    parser.add_argument("--disable_reranker", action="store_true",
                       help="禁用重排序器")
    parser.add_argument("--enable_stock_prediction", action="store_true",
                       help="启用股票预测模式")
    
    args = parser.parse_args()
    
    # 运行端到端测试
    test_summary = run_e2e_test(
        data_path=args.data_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        enable_reranker=not args.disable_reranker,
        enable_stock_prediction=args.enable_stock_prediction
    )
    
    # 输出最终结果
    print("\n" + "="*50)
    print("🎯 端到端测试最终结果")
    print("="*50)
    print(f"平均F1-score: {test_summary['overall_metrics']['average_f1_score']:.4f}")
    print(f"平均Exact Match: {test_summary['overall_metrics']['average_exact_match']:.4f}")
    print(f"成功率: {test_summary['overall_metrics']['success_rate']:.2%}")
    print(f"平均处理时间: {test_summary['overall_metrics']['average_processing_time']:.2f}秒")
    print("="*50)


if __name__ == "__main__":
    main() 