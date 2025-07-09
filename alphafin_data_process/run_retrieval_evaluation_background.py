#!/usr/bin/env python3
"""
后台运行检索评测脚本 - 优化版本
支持一次检索，多指标计算，以节省GPU资源
"""

import sys
import os
import json
import logging
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any, Optional, List

# 允许使用所有GPU (如果需要，可以取消注释并配置)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# --- 修改开始 ---
# 默认导入 config/parameters.py，而不是 parameters_evaluation_light
# 确保 'config.parameters' 是指向您项目根目录下 config 文件夹中 parameters.py 文件的正确路径
from config.parameters import config as default_config
# --- 修改结束 ---

from alphafin_data_process.rag_system_adapter import RagSystemAdapter
from utils.data_loader import load_json_or_jsonl

def save_intermediate_results(
    results: Dict[str, Any], 
    output_path: Path, 
    timestamp: str, 
    mode: str,
    sample_count: int = 0
):
    """
    保存中间结果，防止数据丢失
    
    Args:
        results: 当前结果字典
        output_path: 输出目录
        timestamp: 时间戳
        mode: 当前模式
        sample_count: 已处理的样本数量
    """
    try:
        # 保存当前模式的结果
        mode_file = output_path / f"results_{mode}_{timestamp}.json"
        with open(mode_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存进度信息
        progress_file = output_path / f"progress_{timestamp}.json"
        progress_info = {
            'timestamp': timestamp,
            'current_mode': mode,
            'processed_samples': sample_count,
            'last_update': datetime.now().isoformat(),
            'status': 'running'
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_info, f, ensure_ascii=False, indent=2)
            
        print(f"中间结果已保存 - 模式: {mode}, 样本数: {sample_count}")
        
    except Exception as e:
        print(f"保存中间结果失败: {e}")

def calculate_metrics_from_raw_results(
    raw_results: List[Dict[str, Any]], 
    top_k: int
) -> Dict[str, float]:
    """
    从原始检索结果计算指标
    
    Args:
        raw_results: 原始检索结果列表，每个元素包含：
            - query_text: str - 查询文本
            - ground_truth_doc_ids: List[str] - 正确答案的文档ID列表
            - retrieved_doc_ids_ranked: List[str] - 检索到的文档ID列表（按排序结果）
        top_k: 要计算的top-k值
        
    Returns:
        指标字典，包含MRR、Hit@k等
    """
    mrr_total = 0.0
    hitk_total = 0
    total = 0
    
    for i, result in enumerate(raw_results):
        query_text = result.get('query_text', '')
        ground_truth_doc_ids = result.get('ground_truth_doc_ids', [])
        retrieved_doc_ids = result.get('retrieved_doc_ids_ranked', [])
        
        if not query_text or not ground_truth_doc_ids:
            continue
        
        # 计算MRR和Hit@k - 支持多个相关文档ID
        found_rank = None
        for rank, doc_id in enumerate(retrieved_doc_ids[:top_k], 1):
            if doc_id in ground_truth_doc_ids:
                found_rank = rank
                break
        
        if found_rank is not None:
            mrr_total += 1.0 / found_rank
        else:
            # 如果没有找到相关文档，MRR贡献为0
            pass 
            
        # Hit@k判断，只要找到一个相关文档就算命中
        hit = False
        for doc_id in retrieved_doc_ids[:top_k]:
            if doc_id in ground_truth_doc_ids:
                hit = True
                break
        if hit:
            hitk_total += 1
        
        total += 1
    
    # 计算最终指标
    if total > 0:
        mrr = mrr_total / total
        hitk = hitk_total / total
    else:
        mrr = 0.0
        hitk = 0.0
    
    return {
        'MRR': mrr,
        f'Hit@{top_k}': hitk,
        'total_samples': total
    }

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
    max_samples: Optional[int] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    后台运行检索评测 - 优化版本：一次检索，多指标计算
    
    Args:
        eval_data_path: 评测数据集路径
        output_dir: 输出目录
        modes: 检索模式列表
        top_k_list: Top-K列表
        use_prefilter: 是否使用预过滤（baseline模式可以控制，使用时会自动启用映射功能）
        max_samples: 最大测试样本数（用于快速测试）
        config_path: 配置文件路径（可选，默认使用 config/parameters.py）
        
    Returns:
        评测结果字典
    """
    # 设置默认参数
    if modes is None:
        modes = ['baseline', 'prefilter', 'reranker', 'reranker_no_prefilter']
    if top_k_list is None:
        top_k_list = [1, 3, 5, 10]
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"evaluation_log_{timestamp}.txt"
    
    print(f"开始检索评测（优化版本）...")
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
        print(f"正在加载评测数据: {eval_data_path}")
        
        # 使用 utils/data_loader.py 中的智能加载函数
        eval_data = load_json_or_jsonl(eval_data_path)
        
        # 限制样本数量（用于快速测试）
        if max_samples and max_samples < len(eval_data):
            eval_data = eval_data[:max_samples]
            print(f"限制样本数量为: {max_samples}")
        
        print(f"最终加载评测数据: {len(eval_data)} 个样本")
        
    except Exception as e:
        print(f"❌ 加载评测数据失败: {e}")
        print(f"请检查文件格式是否正确，支持JSON数组格式和JSONL格式")
        return {}
    
    # 初始化RAG系统适配器
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
        
        # 根据配置路径动态导入配置，如果未指定则使用默认导入的 default_config
        if config_path:
            config_module_name = Path(config_path).stem
            config_spec = importlib.util.spec_from_file_location(config_module_name, config_path)
            if config_spec is None:
                raise ImportError(f"无法加载配置文件: {config_path}")
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)
            selected_config = config_module.config
            print(f"使用指定配置初始化RAG系统: {config_path}")
        else:
            # 默认使用上方导入的 config/parameters.py
            selected_config = default_config
            print("使用默认配置文件 (config/parameters.py) 初始化RAG系统...")
        
        adapter = RagSystemAdapter(config=selected_config)
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
        
        # 确定最大检索深度
        max_top_k = max(top_k_list)
        print(f"使用最大检索深度: {max_top_k}")
        
        # 一次性检索 - 记录开始时间
        start_time = time.time()
        try:
            # 这里的 use_prefilter 逻辑保持不变，因为它在 RagSystemAdapter 内部已经被模式参数覆盖
            # 确保传递给 evaluate_retrieval_performance 的 use_prefilter 参数的初始值是命令行指定的
            # 但最终的预过滤行为仍由 RagSystemAdapter 根据 mode 决定
            raw_results = adapter.evaluate_retrieval_performance(
                eval_dataset=eval_data,
                top_k=max_top_k,  # 使用最大深度进行检索
                mode=mode,
                use_prefilter=use_prefilter # 保持这个参数传递方式不变
            )
            retrieval_time = time.time() - start_time
            print(f"检索完成，耗时: {retrieval_time:.2f}秒")
            
            # 检索完成后立即保存原始结果
            raw_results_file = output_path / f"raw_results_{mode}_{timestamp}.json"
            with open(raw_results_file, 'w', encoding='utf-8') as f:
                json.dump(raw_results, f, ensure_ascii=False, indent=2)
            print(f"原始检索结果已保存: {raw_results_file}")
            
        except Exception as e:
            print(f"检索失败: {e}")
            logging.error(f"检索模式 {mode} 失败: {e}", exc_info=True)
            continue
        
        mode_results = {}
        
        # 对每个top_k值计算指标
        for top_k in top_k_list:
            print(f"\n计算 Top-{top_k} 指标...")
            
            try:
                # 使用原始检索结果计算指标
                results = calculate_metrics_from_raw_results(raw_results, top_k)
                results['retrieval_time_seconds'] = retrieval_time  # 添加检索耗时
                
                mode_results[f"top_{top_k}"] = results
                print(f"Top-{top_k} 指标计算完成")
                print(f"   MRR: {results['MRR']:.4f}")
                print(f"   Hit@{top_k}: {results[f'Hit@{top_k}']:.4f}")
                
                # 每个top_k计算完成后保存中间结果
                save_intermediate_results(mode_results, output_path, timestamp, mode, len(eval_data))
                
            except Exception as e:
                print(f"Top-{top_k} 指标计算失败: {e}")
                logging.error(f"Top-{top_k} 指标计算失败: {e}", exc_info=True)
                mode_results[f"top_{top_k}"] = {
                    'MRR': 0.0,
                    f'Hit@{top_k}': 0.0,
                    'total_samples': 0,
                    'retrieval_time_seconds': retrieval_time,
                    'error': str(e)
                }
        
        all_results[mode] = mode_results
        
        # 保存模式结果
        mode_file = output_path / f"results_{mode}_{timestamp}.json"
        with open(mode_file, 'w', encoding='utf-8') as f:
            json.dump(mode_results, f, ensure_ascii=False, indent=2)
        print(f"模式 {mode} 结果已保存到: {mode_file}")
        
        # 同时更新完整结果文件
        complete_file = output_path / f"complete_results_{timestamp}.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"完整结果已更新: {complete_file}")
    
    # 保存完整结果
    complete_file = output_path / f"complete_results_{timestamp}.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"完整结果已保存到: {complete_file}")
    
    # 保存最终状态
    final_progress_file = output_path / f"progress_{timestamp}.json"
    final_progress_info = {
        'timestamp': timestamp,
        'status': 'completed',
        'total_modes': len(modes),
        'completed_modes': list(all_results.keys()),
        'completion_time': datetime.now().isoformat(),
        'total_samples': len(eval_data)
    }
    with open(final_progress_file, 'w', encoding='utf-8') as f:
        json.dump(final_progress_info, f, ensure_ascii=False, indent=2)
    print(f"最终状态已保存: {final_progress_file}")
    
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
                retrieval_time = result.get('retrieval_time_seconds', 0.0)
                print(f"   Top-{top_k}: MRR={mrr:.4f}, Hit@{top_k}={hitk:.4f}, 检索耗时={retrieval_time:.2f}s")
    
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
                        # 从 top_k_name 中解析出 k 值，例如 "top_1" -> 1
                        try:
                            k_value = int(top_k_name.split('_')[1])
                        except (IndexError, ValueError):
                            k_value = 'UNKNOWN' # 处理解析失败的情况
                        
                        f.write(f"Top-{k_value}:\n")
                        f.write(f"   MRR: {top_k_results['MRR']:.4f}\n")
                        f.write(f"   Hit@{k_value}: {top_k_results[f'Hit@{k_value}']:.4f}\n") 
                        f.write(f"   样本数: {top_k_results['total_samples']}\n")
                        f.write(f"   耗时: {top_k_results.get('retrieval_time_seconds', 0):.2f}秒\n") 
                
                f.write("\n")
        
        logger.info(f"总结报告已生成: {output_file}")
        
    except Exception as e:
        logger.error(f"生成总结报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="后台运行检索评测")
    parser.add_argument('--eval_data_path', type=str, 
                        default='data/alphafin/alphafin_eval_samples.jsonl',
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
                        help='是否使用预过滤（注意：baseline模式下此参数会被RagSystemAdapter强制忽略）') 
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数（用于快速测试）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（可选，默认使用 config/parameters.py）') 
    
    args = parser.parse_args()
    
    # 设置日志
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"evaluation_log_{timestamp}.txt"
    logger = setup_logging(log_file)

    try:
        # 运行后台评测
        results = run_evaluation_background(
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            modes=args.modes,
            top_k_list=args.top_k_list,
            use_prefilter=args.use_prefilter,
            max_samples=args.max_samples,
            config_path=args.config
        )
        
        # 生成总结报告
        summary_file = output_path / f"summary_report_{timestamp}.txt"
        generate_summary_report(results, summary_file, logger)

        logger.info("后台评测完成！")
        return 0
        
    except Exception as e:
        logger.error(f"后台评测失败: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())