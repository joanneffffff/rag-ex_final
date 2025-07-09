#!/usr/bin/env python3
"""
三种检索模式实验脚本
后台运行baseline、prefilter、reranker三种模式的对比实验
包含完整的日志记录和结果保存功能
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import signal
import psutil

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.run_retrieval_evaluation_background import run_evaluation_background

class ExperimentRunner:
    """实验运行器类"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 实验配置
        self.experiment_config = {
            'modes': ['baseline', 'prefilter', 'reranker', 'reranker_no_prefilter'],
            'top_k_list': [1, 3, 5, 10],
            'use_prefilter': True,
            'max_samples': None  # 可以设置为数字进行快速测试
        }
    
    def setup_logging(self):
        """设置日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"experiment_log_{timestamp}.txt"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"实验日志文件: {log_file}")
    
    def run_single_experiment(self, 
                            eval_data_path: str,
                            experiment_name: str,
                            modes: Optional[List[str]] = None,
                            top_k_list: Optional[List[int]] = None,
                            use_prefilter: bool = True,
                            max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            eval_data_path: 评测数据集路径
            experiment_name: 实验名称
            modes: 检索模式列表
            top_k_list: Top-K列表
            use_prefilter: 是否使用预过滤
            max_samples: 最大测试样本数
            
        Returns:
            实验结果字典
        """
        # 使用默认配置或传入的配置
        if modes is None:
            modes = self.experiment_config['modes']
        if top_k_list is None:
            top_k_list = self.experiment_config['top_k_list']
        
        # 创建实验输出目录
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始实验: {experiment_name}")
        self.logger.info(f"评测数据: {eval_data_path}")
        self.logger.info(f"输出目录: {experiment_dir}")
        self.logger.info(f"检索模式: {modes}")
        self.logger.info(f"Top-K列表: {top_k_list}")
        self.logger.info(f"预过滤开关: {'开启' if use_prefilter else '关闭'}")
        if max_samples:
            self.logger.info(f"最大样本数: {max_samples}")
        
        try:
            # 运行评测
            start_time = time.time()
            results = run_evaluation_background(
                eval_data_path=eval_data_path,
                output_dir=str(experiment_dir),
                modes=modes,
                top_k_list=top_k_list,
                use_prefilter=use_prefilter,
                max_samples=max_samples
            )
            end_time = time.time()
            
            # 记录实验时间
            experiment_time = end_time - start_time
            self.logger.info(f"实验完成，耗时: {experiment_time:.2f}秒")
            
            # 保存实验配置和结果
            experiment_summary = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': experiment_time,
                'config': {
                    'eval_data_path': eval_data_path,
                    'modes': modes,
                    'top_k_list': top_k_list,
                    'use_prefilter': use_prefilter,
                    'max_samples': max_samples
                },
                'results': results
            }
            
            # 保存实验摘要
            summary_file = experiment_dir / "experiment_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"实验摘要已保存到: {summary_file}")
            
            return experiment_summary
            
        except Exception as e:
            self.logger.error(f"实验失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def run_comprehensive_experiments(self, eval_data_path: str) -> Dict[str, Any]:
        """
        运行全面的对比实验
        
        Args:
            eval_data_path: 评测数据集路径
            
        Returns:
            所有实验结果
        """
        self.logger.info("=" * 80)
        self.logger.info("开始运行全面的三种模式对比实验")
        self.logger.info("=" * 80)
        
        all_experiments = {}
        
        # 实验1: 标准对比实验（baseline vs prefilter vs reranker）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验1: 标准对比实验")
        self.logger.info("=" * 60)
        
        experiment1 = self.run_single_experiment(
            eval_data_path=eval_data_path,
            experiment_name="standard_comparison",
            modes=['baseline', 'prefilter', 'reranker'],
            top_k_list=[1, 3, 5, 10],
            use_prefilter=True,
            max_samples=None
        )
        all_experiments['standard_comparison'] = experiment1
        
        # 实验2: 真正的baseline对比（无预过滤的baseline vs 有预过滤的prefilter）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验2: 真正的baseline对比")
        self.logger.info("=" * 60)
        
        experiment2 = self.run_single_experiment(
            eval_data_path=eval_data_path,
            experiment_name="true_baseline_comparison",
            modes=['baseline', 'prefilter', 'reranker'],
            top_k_list=[1, 3, 5, 10],
            use_prefilter=False,  # baseline不使用预过滤
            max_samples=None
        )
        all_experiments['true_baseline_comparison'] = experiment2
        
        # 实验3: 快速验证实验（小样本测试）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验3: 快速验证实验")
        self.logger.info("=" * 60)
        
        experiment3 = self.run_single_experiment(
            eval_data_path=eval_data_path,
            experiment_name="quick_validation",
            modes=['baseline', 'prefilter', 'reranker'],
            top_k_list=[1, 3, 5, 10],
            use_prefilter=True,
            max_samples=100  # 快速测试
        )
        all_experiments['quick_validation'] = experiment3
        
        # 保存所有实验结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results_file = self.output_dir / f"all_experiments_{timestamp}.json"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_experiments, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n所有实验结果已保存到: {all_results_file}")
        
        # 生成实验报告
        self.generate_experiment_report(all_experiments)
        
        return all_experiments
    
    def generate_experiment_report(self, all_experiments: Dict[str, Any]):
        """生成实验报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"experiment_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("三种检索模式对比实验报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for exp_name, exp_data in all_experiments.items():
                if 'error' in exp_data:
                    f.write(f"实验: {exp_name}\n")
                    f.write(f"状态: 失败 - {exp_data['error']}\n\n")
                    continue
                
                f.write(f"实验: {exp_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"时间: {exp_data['timestamp']}\n")
                f.write(f"耗时: {exp_data['duration_seconds']:.2f}秒\n")
                
                config = exp_data['config']
                f.write(f"配置:\n")
                f.write(f"  评测数据: {config['eval_data_path']}\n")
                f.write(f"  检索模式: {config['modes']}\n")
                f.write(f"  Top-K列表: {config['top_k_list']}\n")
                f.write(f"  预过滤开关: {'开启' if config['use_prefilter'] else '关闭'}\n")
                f.write(f"  最大样本数: {config['max_samples'] or '无限制'}\n")
                
                results = exp_data['results']
                f.write(f"结果:\n")
                for mode in config['modes']:
                    if mode in results:
                        f.write(f"  {mode}模式:\n")
                        for top_k in config['top_k_list']:
                            top_k_key = f"top_{top_k}"
                            if top_k_key in results[mode]:
                                result = results[mode][top_k_key]
                                mrr = result.get('MRR', 0.0)
                                hitk = result.get(f'Hit@{top_k}', 0.0)
                                f.write(f"    Top-{top_k}: MRR={mrr:.4f}, Hit@{top_k}={hitk:.4f}\n")
                
                f.write("\n")
        
        self.logger.info(f"实验报告已生成: {report_file}")
    
    def run_background(self, eval_data_path: str, daemon: bool = False):
        """
        后台运行实验
        
        Args:
            eval_data_path: 评测数据集路径
            daemon: 是否以守护进程模式运行
        """
        if daemon:
            # 创建守护进程
            pid = os.fork()
            if pid != 0:
                # 父进程退出
                sys.exit(0)
            
            # 子进程继续运行
            os.setsid()
            os.umask(0)
            
            # 重定向标准输出和错误输出
            sys.stdout.flush()
            sys.stderr.flush()
            
            with open('/dev/null', 'r') as dev_null:
                os.dup2(dev_null.fileno(), sys.stdin.fileno())
            
            with open(self.output_dir / 'experiment_stdout.log', 'a') as stdout:
                os.dup2(stdout.fileno(), sys.stdout.fileno())
            
            with open(self.output_dir / 'experiment_stderr.log', 'a') as stderr:
                os.dup2(stderr.fileno(), sys.stderr.fileno())
        
        # 设置信号处理
        def signal_handler(signum, frame):
            self.logger.info(f"收到信号 {signum}，正在优雅退出...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 运行实验
            results = self.run_comprehensive_experiments(eval_data_path)
            
            self.logger.info("=" * 80)
            self.logger.info("所有实验完成！")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"实验运行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="三种检索模式对比实验脚本")
    parser.add_argument('--eval_data_path', type=str, 
                       default='data/alphafin/alphafin_eval_samples.jsonl',
                       help='评测数据集路径')
    parser.add_argument('--output_dir', type=str, 
                       default='experiment_results',
                       help='输出目录')
    parser.add_argument('--daemon', action='store_true',
                       help='以守护进程模式运行（后台运行）')
    parser.add_argument('--quick_test', action='store_true',
                       help='快速测试模式（只运行小样本实验）')
    parser.add_argument('--pid_file', type=str,
                       help='PID文件路径（用于守护进程）')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.eval_data_path):
        print(f"错误: 评测数据文件不存在: {args.eval_data_path}")
        sys.exit(1)
    
    # 创建实验运行器
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # 如果指定了PID文件，写入PID
    if args.pid_file:
        with open(args.pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    try:
        if args.quick_test:
            # 快速测试模式
            print("运行快速测试模式...")
            result = runner.run_single_experiment(
                eval_data_path=args.eval_data_path,
                experiment_name="quick_test",
                max_samples=50
            )
        else:
            # 完整实验模式
            result = runner.run_background(
                eval_data_path=args.eval_data_path,
                daemon=args.daemon
            )
        
        if result:
            print("实验完成！")
            sys.exit(0)
        else:
            print("实验失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断实验")
        sys.exit(0)
    except Exception as e:
        print(f"实验运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 