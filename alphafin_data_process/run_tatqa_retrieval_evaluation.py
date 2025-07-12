#!/usr/bin/env python3
"""
TatQA英文数据集检索评估脚本
支持baseline（纯FAISS）和reranker（FAISS+重排序）两种模式
"""

import json
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def setup_logging(output_dir: Path) -> logging.Logger:
    """设置日志记录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"tatqa_evaluation_log_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"TatQA评估日志文件: {log_file}")
    return logger

def run_tatqa_evaluation(
    eval_data_path: str,
    output_dir: str,
    modes: Optional[List[str]] = None,
    top_k_list: Optional[List[int]] = None,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    运行TatQA检索评估
    
    Args:
        eval_data_path: 评测数据集路径
        output_dir: 输出目录
        modes: 检索模式列表（baseline, reranker）
        top_k_list: Top-K列表
        max_samples: 最大测试样本数（用于快速测试）
        
    Returns:
        评测结果字典
    """
    # 设置默认参数
    if modes is None:
        modes = ['baseline', 'reranker']
    if top_k_list is None:
        top_k_list = [1, 3, 5, 10]
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_path)
    
    logger.info("=" * 80)
    logger.info("开始TatQA英文数据集检索评估")
    logger.info("=" * 80)
    logger.info(f"评测数据: {eval_data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"检索模式: {modes}")
    logger.info(f"Top-K列表: {top_k_list}")
    if max_samples:
        logger.info(f"最大样本数: {max_samples}")
    
    # 加载评测数据
    try:
        logger.info("加载TatQA评测数据...")
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = []
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        
        # 限制样本数量（用于快速测试）
        if max_samples and max_samples < len(eval_data):
            eval_data = eval_data[:max_samples]
            logger.info(f"限制样本数量为: {max_samples}")
        
        logger.info(f"加载TatQA评测数据: {len(eval_data)} 个样本")
    except Exception as e:
        logger.error(f"加载TatQA评测数据失败: {e}")
        return {}
    
    # 初始化RAG系统适配器
    try:
        logger.info("初始化RAG系统适配器...")
        adapter = RagSystemAdapter()
        logger.info("RAG系统适配器初始化成功")
    except Exception as e:
        logger.error(f"RAG系统适配器初始化失败: {e}")
        return {}
    
    # 运行评测
    all_results = {}
    
    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始评测模式: {mode}")
        logger.info(f"{'='*50}")
        
        mode_results = {}
        
        for top_k in top_k_list:
            logger.info(f"\n评测 Top-{top_k}...")
            
            try:
                start_time = time.time()
                
                raw_results = adapter.evaluate_retrieval_performance(
                    eval_dataset=eval_data,
                    top_k=top_k,
                    mode=mode
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # 保存原始数据
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_data_file = output_path / f"raw_results_{mode}_{timestamp}.json"
                with open(raw_data_file, 'w', encoding='utf-8') as f:
                    json.dump(raw_results, f, ensure_ascii=False, indent=2)
                logger.info(f"原始数据已保存到: {raw_data_file}")
                
                # 计算指标
                from alphafin_data_process.run_retrieval_evaluation_background import calculate_metrics_from_raw_results
                metrics = calculate_metrics_from_raw_results(raw_results, top_k)
                
                # 构建结果字典
                results = {
                    'MRR': metrics.get('MRR', 0.0),
                    f'Hit@{top_k}': metrics.get(f'Hit@{top_k}', 0.0),
                    'total_samples': metrics.get('total_samples', 0),
                    'duration_seconds': duration,
                    'mode': mode,
                    'top_k': top_k
                }
                
                mode_results[f"top_{top_k}"] = results
                logger.info(f"Top-{top_k} 评测完成，耗时: {duration:.2f}秒")
                
            except Exception as e:
                logger.error(f"Top-{top_k} 评测失败: {e}")
                mode_results[f"top_{top_k}"] = {
                    'MRR': 0.0,
                    f'Hit@{top_k}': 0.0,
                    'total_samples': 0,
                    'error': str(e),
                    'duration_seconds': 0.0,
                    'mode': mode,
                    'top_k': top_k
                }
        
        all_results[mode] = mode_results
        
        # 保存模式结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_file = output_path / f"tatqa_results_{mode}_{timestamp}.json"
        with open(mode_file, 'w', encoding='utf-8') as f:
            json.dump(mode_results, f, ensure_ascii=False, indent=2)
        logger.info(f"模式 {mode} 结果已保存到: {mode_file}")
    
    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    complete_file = output_path / f"tatqa_complete_results_{timestamp}.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"完整结果已保存到: {complete_file}")
    
    # 生成评估报告
    generate_tatqa_report(all_results, output_path, logger)
    
    # 打印汇总结果
    logger.info(f"\n{'='*80}")
    logger.info("TatQA评测汇总结果:")
    logger.info(f"{'='*80}")
    
    for mode in modes:
        logger.info(f"\n模式: {mode}")
        logger.info("-" * 40)
        for top_k in top_k_list:
            if mode in all_results and f"top_{top_k}" in all_results[mode]:
                result = all_results[mode][f"top_{top_k}"]
                mrr = result.get('MRR', 0.0)
                hitk = result.get(f'Hit@{top_k}', 0.0)
                duration = result.get('duration_seconds', 0.0)
                logger.info(f"  Top-{top_k}: MRR={mrr:.4f}, Hit@{top_k}={hitk:.4f}, 耗时={duration:.2f}秒")
    
    logger.info(f"\n{'='*80}")
    logger.info("TatQA评测完成！")
    logger.info(f"{'='*80}")
    
    return all_results

def generate_tatqa_report(results: Dict[str, Any], output_path: Path, logger: logging.Logger):
    """生成TatQA评估报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"tatqa_evaluation_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TatQA英文数据集检索评估报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("实验设计:\n")
            f.write("  实验ⅰ: Encoder + FAISS 检索 (baseline模式)\n")
            f.write("  实验ⅱ: Encoder + FAISS + Reranker 检索 (reranker模式)\n\n")
            
            for mode, mode_results in results.items():
                f.write(f"模式: {mode}\n")
                f.write("-" * 40 + "\n")
                
                if mode == "baseline":
                    f.write("实验ⅰ: Encoder + FAISS 检索\n")
                elif mode == "reranker":
                    f.write("实验ⅱ: Encoder + FAISS + Reranker 检索\n")
                
                f.write("\n")
                
                for top_k_name, top_k_results in mode_results.items():
                    if 'error' in top_k_results:
                        f.write(f"{top_k_name}: 评测失败 - {top_k_results['error']}\n")
                    else:
                        f.write(f"{top_k_name}:\n")
                        f.write(f"  MRR: {top_k_results['MRR']:.4f}\n")
                        hit_key = f'Hit@{top_k_results["top_k"]}'
                        f.write(f"  Hit@{top_k_results['top_k']}: {top_k_results[hit_key]:.4f}\n")
                        f.write(f"  样本数: {top_k_results['total_samples']}\n")
                        f.write(f"  耗时: {top_k_results.get('duration_seconds', 0):.2f}秒\n")
                
                f.write("\n")
            
            # 添加对比分析
            f.write("对比分析:\n")
            f.write("-" * 40 + "\n")
            
            if 'baseline' in results and 'reranker' in results:
                for top_k in [1, 3, 5, 10]:
                    baseline_key = f"top_{top_k}"
                    reranker_key = f"top_{top_k}"
                    
                    if baseline_key in results['baseline'] and reranker_key in results['reranker']:
                        baseline_result = results['baseline'][baseline_key]
                        reranker_result = results['reranker'][reranker_key]
                        
                        if 'error' not in baseline_result and 'error' not in reranker_result:
                            baseline_mrr = baseline_result['MRR']
                            reranker_mrr = reranker_result['MRR']
                            mrr_improvement = reranker_mrr - baseline_mrr
                            
                            baseline_hit = baseline_result[f'Hit@{top_k}']
                            reranker_hit = reranker_result[f'Hit@{top_k}']
                            hit_improvement = reranker_hit - baseline_hit
                            
                            f.write(f"Top-{top_k}:\n")
                            f.write(f"  MRR提升: {baseline_mrr:.4f} → {reranker_mrr:.4f} (+{mrr_improvement:.4f})\n")
                            f.write(f"  Hit@{top_k}提升: {baseline_hit:.4f} → {reranker_hit:.4f} (+{hit_improvement:.4f})\n")
                            f.write(f"  相对提升: MRR +{mrr_improvement/baseline_mrr*100:.1f}%, Hit@{top_k} +{hit_improvement/baseline_hit*100:.1f}%\n\n")
        
        logger.info(f"TatQA评估报告已生成: {report_file}")
        
    except Exception as e:
        logger.error(f"生成TatQA评估报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TatQA英文数据集检索评估脚本")
    parser.add_argument('--eval_data_path', type=str, 
                       default='evaluate_mrr/tatqa_eval_enhanced.jsonl',
                       help='TatQA评测数据集路径')
    parser.add_argument('--output_dir', type=str, 
                       default='alphafin_data_process/tatqa_evaluation_results',
                       help='输出目录')
    parser.add_argument('--modes', nargs='+', 
                       default=['baseline', 'reranker'],
                       help='检索模式列表 (baseline, reranker)')
    parser.add_argument('--top_k_list', nargs='+', type=int,
                       default=[1, 3, 5, 10],
                       help='Top-K列表')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数（用于快速测试）')
    
    args = parser.parse_args()
    
    try:
        # 运行TatQA评测
        results = run_tatqa_evaluation(
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            modes=args.modes,
            top_k_list=args.top_k_list,
            max_samples=args.max_samples
        )
        
        print("TatQA评测完成！")
        return 0
        
    except Exception as e:
        print(f"TatQA评测失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 