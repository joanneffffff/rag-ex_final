#!/usr/bin/env python3
"""
清理AlphaFin数据，去除模板化内容
保留原始财务数据，去除"你是一个股票分析师"等模板化内容
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

def clean_alphafin_context(context: str) -> str:
    """
    清理AlphaFin的context，只删除特定的模板化词汇
    保留大部分原始内容，只去除"你是一个股票分析师"等模板化前缀
    
    Args:
        context: 原始context字符串
        
    Returns:
        清理后的context字符串
    """
    if not context or not isinstance(context, str):
        return context
    
    # 只删除特定的模板化前缀，保留数据内容
    patterns_to_remove = [
        # 删除"你是一个股票分析师"开头的模板化内容
        r"^你是一个股票分析师.*?如下：\s*",
        r"^你是一个股票分析师.*?数据如下：\s*",
        # 删除"以下数据是...你是一个股票分析师"的模板化前缀
        r"^以下数据是.*?时间为.*?，\s*你是一个股票分析师.*?如下：\s*",
        r"^以下数据是.*?时间为.*?，\s*你是一个股票分析师.*?数据如下：\s*",
        # 删除"你是一个股票分析师，我将给你提供一份"的模板化前缀
        r"^你是一个股票分析师.*?我将给你提供一份.*?数据，如下：\s*",
        # 删除复杂的模板化前缀（包含问题部分）
        r"^你是一个股票分析师.*?我将给你提供一份.*?数据表格，这是一份股票名为.*?，股票代码为.*?的最新时间为.*?的数据，【问题】：.*?\s*数据如下：\s*",
    ]
    
    cleaned_context = context
    for pattern in patterns_to_remove:
        cleaned_context = re.sub(pattern, "", cleaned_context, flags=re.DOTALL)
    
    # 删除【问题】和【答案】标记，但保留内容
    cleaned_context = re.sub(r"【问题】：", "", cleaned_context)
    cleaned_context = re.sub(r"【答案】：", "", cleaned_context)
    
    # 去除多余的空格和换行，但保留基本格式
    cleaned_context = re.sub(r'\s+', ' ', cleaned_context).strip()
    
    return cleaned_context

def clean_alphafin_data(input_path: str, output_path: str):
    """
    清理AlphaFin数据文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"🔄 清理AlphaFin数据...")
    print(f"📖 输入文件: {input_path}")
    print(f"💾 输出文件: {output_path}")
    
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 原始数据样本数: {len(data)}")
    
    # 清理数据
    cleaned_data = []
    removed_count = 0
    
    for item in tqdm(data, desc="清理数据"):
        # 清理original_context
        original_context = item.get('original_context', '')
        cleaned_context = clean_alphafin_context(original_context)
        
        # 如果清理后内容为空或太短，跳过
        if not cleaned_context or len(cleaned_context) < 10:
            removed_count += 1
            continue
        
        # 创建清理后的数据项
        cleaned_item = item.copy()
        cleaned_item['original_context'] = cleaned_context
        
        # 可选：清理其他字段
        if 'summary' in cleaned_item:
            cleaned_item['summary'] = clean_alphafin_context(cleaned_item['summary'])
        
        cleaned_data.append(cleaned_item)
    
    print(f"✅ 清理完成:")
    print(f"   📊 保留样本数: {len(cleaned_data)}")
    print(f"   🗑️ 移除样本数: {removed_count}")
    print(f"   📈 保留率: {len(cleaned_data)/len(data)*100:.1f}%")
    
    # 保存清理后的数据
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 清理后的数据已保存到: {output_path}")
    
    # 显示一些示例
    print(f"\n📋 清理示例:")
    for i in range(min(3, len(cleaned_data))):
        item = cleaned_data[i]
        print(f"\n示例 {i+1}:")
        print(f"  公司: {item.get('company_name', 'N/A')}")
        print(f"  股票代码: {item.get('stock_code', 'N/A')}")
        print(f"  清理前长度: {len(item.get('original_context', ''))}")
        print(f"  清理后长度: {len(item['original_context'])}")
        print(f"  清理后内容: {item['original_context'][:100]}...")

def main():
    """主函数"""
    # 输入和输出文件路径
    input_file = "data/alphafin/alphafin_merged_generated_qa_full_dedup.json"
    output_file = "data/alphafin/alphafin_cleaned.json"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
    # 清理数据
    clean_alphafin_data(input_file, output_file)
    
    print(f"\n🎉 数据清理完成！")
    print(f"📝 建议更新配置文件中的中文数据路径为: {output_file}")

if __name__ == "__main__":
    main() 