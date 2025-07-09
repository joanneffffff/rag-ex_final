#!/usr/bin/env python3
import pandas as pd
import re

def normalize_stock_code(code):
    """标准化股票代码格式，补全前导零"""
    if pd.isna(code):
        return code
    
    code_str = str(code).strip()
    
    # 如果是纯数字，补全到6位
    if re.match(r'^\d+$', code_str):
        return code_str.zfill(6)
    
    return code_str

def fix_stock_mapping():
    """修复股票代码映射文件"""
    input_file = 'data/astock_code_company_name.csv'
    output_file = 'data/astock_code_company_name_fixed.csv'
    
    print(f"正在修复映射文件: {input_file}")
    
    # 读取原始文件
    df = pd.read_csv(input_file)
    print(f"原始记录数: {len(df)}")
    
    # 检查股票代码格式
    print("\n检查股票代码格式:")
    code_lengths = df['stock_code'].astype(str).str.len().value_counts().sort_index()
    for length, count in code_lengths.items():
        print(f"  {length}位数字: {count}条")
    
    # 标准化股票代码
    print("\n正在标准化股票代码...")
    df['stock_code'] = df['stock_code'].apply(normalize_stock_code)
    
    # 检查修复后的格式
    print("\n修复后的股票代码格式:")
    fixed_code_lengths = df['stock_code'].astype(str).str.len().value_counts().sort_index()
    for length, count in fixed_code_lengths.items():
        print(f"  {length}位数字: {count}条")
    
    # 检查是否有重复
    duplicates = df[df.duplicated(subset=['stock_code'], keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠️  发现重复股票代码: {len(duplicates)}条")
        for _, row in duplicates.head().iterrows():
            print(f"    {row['stock_code']} -> {row['company_name']}")
    else:
        print("\n✅ 无重复股票代码")
    
    # 保存修复后的文件
    df.to_csv(output_file, index=False)
    print(f"\n修复后的文件已保存: {output_file}")
    
    # 验证修复效果
    print("\n验证修复效果:")
    test_codes = ['2812', '002812', '600481', '601618']
    for code in test_codes:
        normalized = normalize_stock_code(code)
        print(f"  {code} -> {normalized}")
    
    return df

if __name__ == "__main__":
    fixed_df = fix_stock_mapping() 