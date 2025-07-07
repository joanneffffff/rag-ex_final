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
from typing import Dict, Any, Optional

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
    modes: Optional[list] = None,
    top_k_list: Optional[list] = None,
    use_prefilter: bool = True,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    后台运行检索评测
    
    Args:
        eval_data_path: 评测数据集路径
        output_dir: 输出目录
        modes: 检索模式列表
        top_k_list: Top-K列表
        use_prefilter: 是否使用预过滤（baseline模式可以控制，使用时会自动启用映射功能）
        max_samples: 最大测试样本数（用于快速测试）
        
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
    log_file = output_path / f"evaluation_log_{timestamp}.txt"
    
    print(f"开始检索评测...")
    print(f"评测数据: {eval_data_path}")
    print(f"输出目录: {output_dir}")
    print(f"检索模式: {modes}")
    print(f"Top-K列表: {top_k_list}")
    print(f"预过滤开关: {'开启' if use_prefilter else '关闭'}")
    if use_prefilter:
        print("预过滤模式下自动启用股票代码和公司名称映射")
    if max_samples:
        print(f"最大样本数: {max_samples}")
    
    # 加载评测数据
    try:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f]
        
        # 限制样本数量（用于快速测试）
        if max_samples and max_samples < len(eval_data):
            eval_data = eval_data[:max_samples]
            print(f"限制样本数量为: {max_samples}")
        
        print(f"加载评测数据: {len(eval_data)} 个样本")
    except Exception as e:
        print(f"加载评测数据失败: {e}")
        return {}
    
    # 初始化RAG系统适配器
    try:
        adapter = RagSystemAdapter()
        print("RAG系统适配器初始化成功")
    except Exception as e:
        print(f"RAG系统适配器初始化失败: {e}")
        return {}
    
    # 运行评测
    all_results = {}
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"开始评测模式: {mode}")
        print(f"{'='*50}")
        
        mode_results = {}
        
        for top_k in top_k_list:
            print(f"\n评测 Top-{top_k}...")
            
            try:
                results = adapter.evaluate_retrieval_performance(
                    eval_dataset=eval_data,
                    top_k=top_k,
                    mode=mode,
                    use_prefilter=use_prefilter
                )
                
                mode_results[f"top_{top_k}"] = results
                print(f"Top-{top_k} 评测完成")
                
            except Exception as e:
                print(f"Top-{top_k} 评测失败: {e}")
                mode_results[f"top_{top_k}"] = {
                    'MRR': 0.0,
                    f'Hit@{top_k}': 0.0,
                    'total_samples': 0,
                    'error': str(e)
                }
        
        all_results[mode] = mode_results
        
        # 保存模式结果
        mode_file = output_path / f"results_{mode}_{timestamp}.json"
        with open(mode_file, 'w', encoding='utf-8') as f:
            json.dump(mode_results, f, ensure_ascii=False, indent=2)
        print(f"模式 {mode} 结果已保存到: {mode_file}")
    
    # 保存完整结果
    complete_file = output_path / f"complete_results_{timestamp}.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"完整结果已保存到: {complete_file}")
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("评测汇总结果:")
    print(f"{'='*80}")
    
    for mode in modes:
        print(f"\n模式: {mode}")
        print("-" * 40)
        for top_k in top_k_list:
            if mode in all_results and f"top_{top_k}" in all_results[mode]:
                result = all_results[mode][f"top_{top_k}"]
                mrr = result.get('MRR', 0.0)
                hitk = result.get(f'Hit@{top_k}', 0.0)
                print(f"  Top-{top_k}: MRR={mrr:.4f}, Hit@{top_k}={hitk:.4f}")
    
    return all_results

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
    parser.add_argument('--use_prefilter', action='store_true',
                       help='是否使用预过滤（baseline模式可以控制）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数（用于快速测试）')
    
    args = parser.parse_args()
    
    try:
        # 运行后台评测
        results = run_evaluation_background(
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            modes=args.modes,
            top_k_list=args.top_k_list,
            use_prefilter=args.use_prefilter,
            max_samples=args.max_samples
        )
        
        print("后台评测完成！")
        return 0
        
    except Exception as e:
        print(f"后台评测失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 