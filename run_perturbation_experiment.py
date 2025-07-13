#!/usr/bin/env python3
"""
扰动实验运行脚本
支持配置化运行不同类型的扰动实验
"""

import json
import argparse
from typing import List, Dict, Any
from unified_perturbation_experiment import UnifiedPerturbationExperiment

def load_test_samples(file_path: str) -> List[Dict[str, Any]]:
    """从文件加载测试样本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种数据格式
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'samples' in data:
            return data['samples']
        else:
            raise ValueError("不支持的数据格式")
            
    except Exception as e:
        print(f"❌ 加载测试样本失败: {e}")
        return []

def create_default_samples() -> List[Dict[str, Any]]:
    """创建默认测试样本"""
    return [
        {
            'id': 'sample_1',
            'query': '首钢股份在2023年上半年的业绩表现如何？',
            'answer': '首钢股份在2023年上半年业绩表现良好，营收增长15%，净利润增长20%'
        },
        {
            'id': 'sample_2', 
            'query': '中国平安的财务状况怎么样？',
            'answer': '中国平安财务状况稳健，总资产超过10万亿元，净利润持续增长'
        },
        {
            'id': 'sample_3',
            'query': '腾讯控股的营收增长情况如何？',
            'answer': '腾讯控股营收保持稳定增长，游戏业务和广告业务表现良好'
        }
    ]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG扰动实验运行脚本')
    parser.add_argument('--samples', type=str, help='测试样本文件路径')
    parser.add_argument('--output', type=str, default='perturbation_results.json', 
                       help='输出结果文件路径')
    parser.add_argument('--perturbers', type=str, nargs='+', 
                       choices=['leave_one_out', 'reorder', 'trend', 'year', 'term'],
                       help='指定要测试的扰动器')
    parser.add_argument('--max_samples', type=int, default=10, 
                       help='最大测试样本数')
    
    args = parser.parse_args()
    
    print("🧪 RAG扰动实验运行脚本")
    print("=" * 50)
    
    # 加载测试样本
    if args.samples:
        samples = load_test_samples(args.samples)
        if not samples:
            print("⚠️ 无法加载样本文件，使用默认样本")
            samples = create_default_samples()
    else:
        print("📝 使用默认测试样本")
        samples = create_default_samples()
    
    # 限制样本数量
    if len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        print(f"📊 限制测试样本数为 {args.max_samples}")
    
    print(f"📋 测试样本数: {len(samples)}")
    
    # 创建实验实例
    experiment = UnifiedPerturbationExperiment()
    
    # 如果指定了扰动器，只测试指定的
    if args.perturbers:
        experiment.perturbers = {
            name: experiment.perturbers[name] 
            for name in args.perturbers 
            if name in experiment.perturbers
        }
        print(f"🎯 指定测试扰动器: {list(experiment.perturbers.keys())}")
    
    # 运行实验
    print(f"\n🚀 开始运行扰动实验...")
    results = experiment.run_comprehensive_experiment(samples)
    
    # 保存结果
    experiment.save_results(results, args.output)
    
    # 分析结果
    analysis = experiment.analyze_results(results)
    
    print(f"\n📊 实验分析结果:")
    print(f"总实验数: {analysis['total_experiments']}")
    print(f"扰动器统计:")
    for perturber_name, stats in analysis['perturber_stats'].items():
        print(f"  {perturber_name}: {stats['count']} 个实验")
        print(f"    平均F1分数: {stats['avg_f1_score']:.3f}")
        print(f"    平均准确性分数: {stats['avg_accuracy_score']:.3f}")
    
    print(f"\n🎉 实验完成！结果已保存到 {args.output}")

if __name__ == "__main__":
    main() 