#!/usr/bin/env python3
"""
后台运行检索评测脚本
支持日志记录，不使用shell
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from utils.data_loader import load_json_or_jsonl

def setup_logging(log_file: str) -> logging.Logger:
    """设置日志记录"""
    # 创建日志目录
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_evaluation_background(
    eval_data_path: str,
    output_dir: str,
    modes: list = None,
    top_k_list: list = None
) -> Dict[str, Any]:
    """
    后台运行检索评测
    
    Args:
        eval_data_path: 评测数据集路径
        output_dir: 输出目录
        modes: 检索模式列表
        top_k_list: Top-K列表
        
    Returns:
        评测结果字典
    """
    # 设置默认参数
    if modes is None:
        modes = ['baseline', 'prefilter', 'reranker']
    if top_k_list is None:
        top_k_list = [1, 3, 5, 10]
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"retrieval_evaluation_{timestamp}.log"
    logger = setup_logging(str(log_file))
    
    logger.info("=" * 60)
    logger.info("开始后台检索评测")
    logger.info(f"评测数据集: {eval_data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"检索模式: {modes}")
    logger.info(f"Top-K列表: {top_k_list}")
    logger.info("=" * 60)
    
    try:
        # 加载评测数据
        logger.info("加载评测数据集...")
        eval_dataset = load_json_or_jsonl(eval_data_path)
        if not eval_dataset:
            raise ValueError("没有加载到任何有效数据")
        logger.info(f"加载完成，共 {len(eval_dataset)} 个样本")
        
        # 初始化适配器
        logger.info("初始化RAG系统适配器...")
        adapter = RagSystemAdapter()
        logger.info("适配器初始化完成")
        
        # 存储所有结果
        all_results = {}
        
        # 对每种模式进行评测
        for mode in modes:
            logger.info(f"\n开始评测模式: {mode}")
            mode_results = {}
            
            for top_k in top_k_list:
                logger.info(f"评测 Top-{top_k}...")
                start_time = time.time()
                
                try:
                    # 执行评测
                    results = adapter.evaluate_retrieval_performance(
                        eval_dataset=eval_dataset,
                        top_k=top_k,
                        mode=mode
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # 添加时间信息
                    results['duration_seconds'] = duration
                    results['timestamp'] = datetime.now().isoformat()
                    
                    mode_results[f'top_{top_k}'] = results
                    
                    logger.info(f"Top-{top_k} 评测完成")
                    logger.info(f"MRR: {results['mrr']:.4f}")
                    logger.info(f"Hit@{top_k}: {results[f'hit@{top_k}']:.4f}")
                    logger.info(f"耗时: {duration:.2f}秒")
                    
                except Exception as e:
                    logger.error(f"Top-{top_k} 评测失败: {e}")
                    mode_results[f'top_{top_k}'] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            
            all_results[mode] = mode_results
            
            # 保存模式结果
            mode_output_file = output_path / f"results_{mode}_{timestamp}.json"
            with open(mode_output_file, 'w', encoding='utf-8') as f:
                json.dump(mode_results, f, ensure_ascii=False, indent=2)
            logger.info(f"模式 {mode} 结果已保存到: {mode_output_file}")
        
        # 保存完整结果
        complete_output_file = output_path / f"complete_results_{timestamp}.json"
        with open(complete_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 生成总结报告
        summary_file = output_path / f"evaluation_summary_{timestamp}.txt"
        generate_summary_report(all_results, summary_file, logger)
        
        logger.info("=" * 60)
        logger.info("后台检索评测完成")
        logger.info(f"完整结果: {complete_output_file}")
        logger.info(f"总结报告: {summary_file}")
        logger.info(f"详细日志: {log_file}")
        logger.info("=" * 60)
        
        return all_results
        
    except Exception as e:
        logger.error(f"评测过程中发生错误: {e}")
        raise

def generate_summary_report(results: Dict[str, Any], output_file: Path, logger: logging.Logger):
    """生成总结报告"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("检索评测总结报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for mode, mode_results in results.items():
                f.write(f"模式: {mode}\n")
                f.write("-" * 30 + "\n")
                
                for top_k_name, top_k_results in mode_results.items():
                    if 'error' in top_k_results:
                        f.write(f"{top_k_name}: 评测失败 - {top_k_results['error']}\n")
                    else:
                        f.write(f"{top_k_name}:\n")
                        f.write(f"  MRR: {top_k_results['mrr']:.4f}\n")
                        hit_key = f"hit@{top_k_results['mode']}"
                        f.write(f"  Hit@{top_k_results['mode']}: {top_k_results[hit_key]:.4f}\n")
                        f.write(f"  样本数: {top_k_results['total_samples']}\n")
                        f.write(f"  耗时: {top_k_results.get('duration_seconds', 0):.2f}秒\n")
                
                f.write("\n")
        
        logger.info(f"总结报告已生成: {output_file}")
        
    except Exception as e:
        logger.error(f"生成总结报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="后台运行检索评测")
    parser.add_argument('--eval_data_path', type=str, 
                       default='data/alphafin/alphafin_eval_samples.json',
                       help='评测数据集路径')
    parser.add_argument('--output_dir', type=str, 
                       default='alphafin_data_process/evaluation_results',
                       help='输出目录')
    parser.add_argument('--modes', nargs='+', 
                       default=['baseline', 'prefilter', 'reranker'],
                       help='检索模式列表')
    parser.add_argument('--top_k_list', nargs='+', type=int,
                       default=[1, 3, 5, 10],
                       help='Top-K列表')
    
    args = parser.parse_args()
    
    try:
        # 运行后台评测
        results = run_evaluation_background(
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            modes=args.modes,
            top_k_list=args.top_k_list
        )
        
        print("后台评测完成！")
        return 0
        
    except Exception as e:
        print(f"后台评测失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 