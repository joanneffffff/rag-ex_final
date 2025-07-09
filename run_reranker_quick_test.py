"""
Reranker模式快速测试脚本
只运行reranker和reranker_no_prefilter两个模式，使用少量样本
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

from alphafin_data_process.run_retrieval_evaluation_background import run_evaluation_background

def main():
    """运行reranker快速测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reranker模式快速测试")
    parser.add_argument('--modes', nargs='+', 
                       default=['reranker', 'reranker_no_prefilter'],
                       help='评测模式列表')
    parser.add_argument('--eval_data_path', type=str,
                       default="data/alphafin/alphafin_eval_samples.jsonl",
                       help='评测数据路径')
    parser.add_argument('--output_dir', type=str,
                       default="alphafin_data_process/evaluation_results",
                       help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数，None表示全部数据')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Reranker模式快速测试")
    print("=" * 80)
    
    # 评测参数
    eval_data_path = args.eval_data_path
    output_dir = args.output_dir
    modes = args.modes
    top_k_list = [1, 3, 5, 10]
    max_samples = args.max_samples
    
    print(f"评测数据: {eval_data_path}")
    print(f"输出目录: {output_dir}")
    print(f"评测模式: {modes}")
    print(f"Top-K列表: {top_k_list}")
    print(f"最大样本数: {max_samples}")
    
    # 检查数据文件是否存在
    if not os.path.exists(eval_data_path):
        print(f"❌ 评测数据文件不存在: {eval_data_path}")
        return
    
    # 运行评测
    try:
        # 添加内存监控
        import torch
        if torch.cuda.is_available():
            print(f"GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB")
        
        results = run_evaluation_background(
            eval_data_path=eval_data_path,
            output_dir=output_dir,
            modes=modes,
            top_k_list=top_k_list,
            use_prefilter=True,  # 这个参数会被mode覆盖
            max_samples=max_samples
        )
        
        if results:
            print("\n" + "=" * 80)
            print("快速测试完成！")
            print("=" * 80)
            
            # 打印结果摘要
            for mode in modes:
                if mode in results:
                    print(f"\n{mode}模式结果:")
                    for top_k in top_k_list:
                        top_k_key = f"top_{top_k}"
                        if top_k_key in results[mode]:
                            result = results[mode][top_k_key]
                            mrr = result.get('MRR', 0.0)
                            hitk = result.get(f'Hit@{top_k}', 0.0)
                            print(f"  Top-{top_k}: MRR={mrr:.4f}, Hit@{top_k}={hitk:.4f}")
        else:
            print("❌ 评测失败")
            
    except Exception as e:
        print(f"❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 