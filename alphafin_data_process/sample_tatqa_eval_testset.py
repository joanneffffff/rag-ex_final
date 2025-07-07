#!/usr/bin/env python3
"""
从TatQA评估数据集中等量抽取三类样本生成测试集
按answer_from字段分组：text/table/table+text，每组等量抽取，总计100条
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def sample_tatqa_eval_testset(
    input_file: str,
    output_file: str,
    total_samples: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    从TatQA评估数据集中等量抽取样本生成测试集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        total_samples: 总样本数
        seed: 随机种子
        
    Returns:
        统计信息字典
    """
    # 设置随机种子
    random.seed(seed)
    
    print(f"开始从 {input_file} 抽取测试集...")
    print(f"目标样本数: {total_samples}")
    print(f"随机种子: {seed}")
    
    # 读取原始数据
    data_by_type = defaultdict(list)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        answer_from = item.get('answer_from', 'unknown')
                        data_by_type[answer_from].append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告: 第{line_num}行JSON解析失败: {e}")
                        continue
        
        print(f"原始数据统计:")
        for data_type, items in data_by_type.items():
            print(f"  {data_type}: {len(items)} 条")
        
    except FileNotFoundError:
        print(f"错误: 输入文件不存在: {input_file}")
        return {}
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        return {}
    
    # 确定目标数据类型
    target_types = ['text', 'table', 'table+text']
    available_types = [t for t in target_types if t in data_by_type]
    
    if not available_types:
        print("错误: 未找到目标数据类型")
        return {}
    
    print(f"\n目标数据类型: {available_types}")
    
    # 计算每组样本数
    samples_per_type = total_samples // len(available_types)
    remainder = total_samples % len(available_types)
    
    print(f"每组样本数: {samples_per_type}")
    if remainder > 0:
        print(f"额外分配: {remainder} 条")
    
    # 抽取样本
    sampled_data = []
    stats = {}
    
    for i, data_type in enumerate(available_types):
        items = data_by_type[data_type]
        
        # 计算当前类型应抽取的样本数
        current_samples = samples_per_type
        if i < remainder:  # 优先分配给前面的类型
            current_samples += 1
        
        # 确保不超过可用样本数
        current_samples = min(current_samples, len(items))
        
        # 随机抽取
        sampled_items = random.sample(items, current_samples)
        sampled_data.extend(sampled_items)
        
        stats[data_type] = {
            'available': len(items),
            'sampled': current_samples
        }
        
        print(f"  {data_type}: 可用{len(items)}条 → 抽取{current_samples}条")
    
    # 打乱样本顺序
    random.shuffle(sampled_data)
    
    # 保存到输出文件
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n测试集已保存到: {output_file}")
        print(f"总样本数: {len(sampled_data)}")
        
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")
        return {}
    
    # 验证输出
    print(f"\n输出验证:")
    output_stats = defaultdict(int)
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answer_from = item.get('answer_from', 'unknown')
                output_stats[answer_from] += 1
    
    for data_type, count in output_stats.items():
        print(f"  {data_type}: {count} 条")
    
    # 返回统计信息
    result_stats = {
        'input_file': input_file,
        'output_file': output_file,
        'total_samples': len(sampled_data),
        'target_samples': total_samples,
        'sampling_stats': stats,
        'output_stats': dict(output_stats)
    }
    
    return result_stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从TatQA评估数据集中等量抽取样本生成测试集")
    parser.add_argument('--input_file', type=str, 
                       default='evaluate_mrr/tatqa_eval_enhanced.jsonl',
                       help='输入文件路径')
    parser.add_argument('--output_file', type=str, 
                       default='evaluate_mrr/tatqa_eval_test_100.jsonl',
                       help='输出文件路径')
    parser.add_argument('--total_samples', type=int, default=100,
                       help='总样本数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    try:
        # 执行抽样
        stats = sample_tatqa_eval_testset(
            input_file=args.input_file,
            output_file=args.output_file,
            total_samples=args.total_samples,
            seed=args.seed
        )
        
        if stats:
            print(f"\n✅ 测试集生成成功!")
            print(f"📁 输出文件: {stats['output_file']}")
            print(f"📊 样本总数: {stats['total_samples']}")
        else:
            print("❌ 测试集生成失败!")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 