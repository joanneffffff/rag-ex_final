#!/usr/bin/env python3
"""
通用数据加载工具
支持JSON和JSONL格式的智能检测和加载
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

def load_json_or_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    智能加载JSON或JSONL格式的数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        加载的数据列表
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持或解析失败
    """
    file_path = Path(file_path)
    
    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"📖 正在从 {file_path} 加载数据...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取第一个字符来判断格式
        first_char = f.read(1)
        f.seek(0)  # 重置文件指针
        
        if first_char == '[':
            # 标准JSON数组格式
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"✅ 成功加载为JSON数组，样本数: {len(data)}")
                    return data
                else:
                    raise ValueError(f"文件以[开头但不是数组格式")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON数组解析失败: {e}")
        else:
            # 尝试JSONL格式（每行一个JSON对象）
            eval_data = []
            try:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            eval_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                            continue
                
                if eval_data:
                    print(f"✅ 成功加载为JSONL格式，样本数: {len(eval_data)}")
                    return eval_data
                else:
                    raise ValueError("JSONL解析失败，没有有效数据")
            except Exception as e:
                raise ValueError(f"JSONL解析失败: {e}")

def save_json_or_jsonl(data: List[Dict[str, Any]], 
                      file_path: Union[str, Path], 
                      format: str = "auto") -> None:
    """
    保存数据为JSON或JSONL格式
    
    Args:
        data: 要保存的数据列表
        file_path: 保存路径
        format: 保存格式 ("json", "jsonl", "auto")
    """
    file_path = Path(file_path)
    
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "auto":
        # 根据文件扩展名自动选择格式
        if file_path.suffix.lower() == ".jsonl":
            format = "jsonl"
        else:
            format = "json"
    
    if format == "jsonl":
        # 保存为JSONL格式（每行一个JSON对象）
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ 数据已保存为JSONL格式: {file_path}")
    else:
        # 保存为JSON格式
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 数据已保存为JSON格式: {file_path}")

def convert_format(input_path: Union[str, Path], 
                  output_path: Union[str, Path], 
                  target_format: str) -> None:
    """
    转换数据格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_format: 目标格式 ("json" 或 "jsonl")
    """
    # 加载数据
    data = load_json_or_jsonl(input_path)
    
    # 保存为目标格式
    save_json_or_jsonl(data, output_path, target_format)

def validate_data_format(data: List[Dict[str, Any]], 
                        required_fields: List[str] = None) -> bool:
    """
    验证数据格式是否正确
    
    Args:
        data: 数据列表
        required_fields: 必需字段列表
        
    Returns:
        是否格式正确
    """
    if not isinstance(data, list):
        print("❌ 数据不是列表格式")
        return False
    
    if len(data) == 0:
        print("⚠️ 数据列表为空")
        return True
    
    # 检查第一个样本的字段
    first_sample = data[0]
    if not isinstance(first_sample, dict):
        print("❌ 数据样本不是字典格式")
        return False
    
    # 检查必需字段
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in first_sample:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            return False
    
    print(f"✅ 数据格式验证通过，共 {len(data)} 个样本")
    return True

def sample_data(data: List[Dict[str, Any]], 
                sample_size: int = None, 
                random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    从数据中采样
    
    Args:
        data: 原始数据列表
        sample_size: 采样数量，None表示全部
        random_seed: 随机种子
        
    Returns:
        采样后的数据列表
    """
    if sample_size is None or sample_size >= len(data):
        return data
    
    import numpy as np
    np.random.seed(random_seed)
    indices = np.random.choice(len(data), sample_size, replace=False)
    sampled_data = [data[i] for i in indices]
    
    print(f"✅ 随机采样 {len(sampled_data)} 个样本")
    return sampled_data

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据加载工具")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径（用于格式转换）")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], help="目标格式")
    parser.add_argument("--sample", type=int, help="采样数量")
    parser.add_argument("--validate", action="store_true", help="验证数据格式")
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        data = load_json_or_jsonl(args.input)
        
        # 验证格式
        if args.validate:
            validate_data_format(data)
        
        # 采样
        if args.sample:
            data = sample_data(data, args.sample)
        
        # 格式转换
        if args.output and args.format:
            save_json_or_jsonl(data, args.output, args.format)
        
        print(f"✅ 处理完成，共 {len(data)} 个样本")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        exit(1) 