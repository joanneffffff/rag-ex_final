#!/usr/bin/env python3
"""
元数据过滤流程可视化示例
展示从查询提取到元数据过滤的完整过程
"""

from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import re

def extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """从查询中提取股票代码和公司名称"""
    stock_code = None
    company_name = None
    
    # 1. 括号格式匹配
    bracket_patterns = [
        r'([^，。？\s]+)[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 公司名(股票代码)
        r'[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 纯(股票代码)
    ]
    
    for pattern in bracket_patterns:
        match = re.search(pattern, query)
        if match:
            if len(match.groups()) == 2:
                company_name = match.group(1)
                stock_code = match.group(2)
            else:
                stock_code = match.group(1)
            break
    
    # 2. 无括号格式匹配
    if not stock_code:
        no_bracket_patterns = [
            r'([^，。？\s]+)\s*(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码
            r'(\d{6}(?:\.(?:SZ|SH))?)',  # 纯股票代码
        ]
        
        for pattern in no_bracket_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 2:
                    company_name = match.group(1)
                    stock_code = match.group(2)
                else:
                    stock_code = match.group(1)
                break
    
    # 3. 清理股票代码
    if stock_code:
        pure_code_match = re.search(r'(\d{6})', stock_code)
        if pure_code_match:
            stock_code = pure_code_match.group(1)
    
    return company_name, stock_code

def build_metadata_index(data: List[Dict]) -> Dict:
    """构建元数据索引"""
    metadata_index = {
        'company_name': defaultdict(list),
        'stock_code': defaultdict(list),
        'company_stock': defaultdict(list)
    }
    
    for idx, record in enumerate(data):
        # 公司名称索引
        if record.get('company_name'):
            company_name = record['company_name'].strip().lower()
            metadata_index['company_name'][company_name].append(idx)
        
        # 股票代码索引
        if record.get('stock_code'):
            stock_code = str(record['stock_code']).strip().lower()
            metadata_index['stock_code'][stock_code].append(idx)
        
        # 组合索引
        if record.get('company_name') and record.get('stock_code'):
            company_name = record['company_name'].strip().lower()
            stock_code = str(record['stock_code']).strip().lower()
            key = f"{company_name}_{stock_code}"
            metadata_index['company_stock'][key].append(idx)
    
    return metadata_index

def pre_filter(metadata_index: Dict, 
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               max_candidates: int = 1000,
               total_records: int = 0) -> List[int]:
    """基于元数据进行预过滤"""
    
    print(f"  元数据过滤条件:")
    print(f"    公司名称: {company_name}")
    print(f"    股票代码: {stock_code}")
    
    # 优先使用组合索引
    if company_name and stock_code:
        company_name_lower = company_name.strip().lower()
        stock_code_lower = stock_code.strip().lower()
        key = f"{company_name_lower}_{stock_code_lower}"
        
        if key in metadata_index['company_stock']:
            indices = metadata_index['company_stock'][key]
            print(f"    ✅ 组合过滤成功: 找到 {len(indices)} 条匹配记录")
            return indices[:max_candidates]
        else:
            print(f"    ❌ 组合过滤失败: 无匹配记录")
            return []
    
    # 单一过滤
    elif company_name:
        company_name_lower = company_name.strip().lower()
        if company_name_lower in metadata_index['company_name']:
            indices = metadata_index['company_name'][company_name_lower]
            print(f"    ✅ 公司名称过滤成功: 找到 {len(indices)} 条匹配记录")
            return indices[:max_candidates]
        else:
            print(f"    ❌ 公司名称过滤失败: 无匹配记录")
            return []
    
    elif stock_code:
        stock_code_lower = stock_code.strip().lower()
        if stock_code_lower in metadata_index['stock_code']:
            indices = metadata_index['stock_code'][stock_code_lower]
            print(f"    ✅ 股票代码过滤成功: 找到 {len(indices)} 条匹配记录")
            return indices[:max_candidates]
        else:
            print(f"    ❌ 股票代码过滤失败: 无匹配记录")
            return []
    
    # 无过滤条件
    print(f"    ⚠️  无过滤条件，返回所有记录")
    return list(range(total_records))

def demonstrate_metadata_filtering():
    """演示元数据过滤流程"""
    print("=== 元数据过滤流程演示 ===\n")
    
    # 1. 模拟数据
    print("1. 模拟数据:")
    data = [
        {'company_name': '德赛电池', 'stock_code': '000049', 'content': '德赛电池2021年财报...'},
        {'company_name': '德赛电池', 'stock_code': '000049', 'content': '德赛电池2022年财报...'},
        {'company_name': '用友网络', 'stock_code': '600588', 'content': '用友网络2021年财报...'},
        {'company_name': '中国平安', 'stock_code': '601318', 'content': '中国平安2021年财报...'},
        {'company_name': '腾讯控股', 'stock_code': '00700', 'content': '腾讯控股2021年财报...'},
    ]
    
    for i, record in enumerate(data):
        print(f"   记录 {i}: {record['company_name']}({record['stock_code']}) - {record['content'][:20]}...")
    
    print("\n2. 构建元数据索引:")
    metadata_index = build_metadata_index(data)
    
    print("   公司名称索引:")
    for company, indices in metadata_index['company_name'].items():
        print(f"     {company}: {indices}")
    
    print("   股票代码索引:")
    for stock, indices in metadata_index['stock_code'].items():
        print(f"     {stock}: {indices}")
    
    print("   组合索引:")
    for key, indices in metadata_index['company_stock'].items():
        print(f"     {key}: {indices}")
    
    # 3. 测试查询
    test_queries = [
        "德赛电池（000049）的业绩如何？",
        "德赛电池(000049.SZ)的财务表现？",
        "000049的股价走势",
        "用友网络（600588）的营收情况？",
        "什么是财务报表？",
    ]
    
    print("\n3. 查询处理和元数据过滤:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        
        # 步骤1: 正则表达式提取
        print("  步骤1: 正则表达式提取")
        company_name, stock_code = extract_stock_info(query)
        print(f"    提取结果: 公司={company_name}, 股票={stock_code}")
        
        # 步骤2: 元数据过滤
        print("  步骤2: 元数据过滤")
        filtered_indices = pre_filter(metadata_index, company_name, stock_code, 1000, len(data))
        
        # 步骤3: 显示过滤结果
        print("  步骤3: 过滤结果")
        if filtered_indices:
            print(f"    找到 {len(filtered_indices)} 条相关记录:")
            for idx in filtered_indices[:3]:  # 只显示前3条
                record = data[idx]
                print(f"      - {record['company_name']}({record['stock_code']}): {record['content'][:30]}...")
            if len(filtered_indices) > 3:
                print(f"      ... 还有 {len(filtered_indices) - 3} 条记录")
        else:
            print("    未找到匹配记录")
        
        print("-" * 60)

def main():
    """主函数"""
    demonstrate_metadata_filtering()
    
    print("\n=== 总结 ===")
    print("元数据过滤流程:")
    print("1. 使用正则表达式从用户查询中提取股票信息")
    print("2. 将提取的信息与预构建的元数据索引进行匹配")
    print("3. 根据匹配结果过滤出相关的文档记录")
    print("4. 将过滤后的记录传递给后续的向量检索和重排序")
    print("\n优势:")
    print("- 提高检索精度：只检索相关公司的文档")
    print("- 减少计算量：避免在全量数据中搜索")
    print("- 提升用户体验：返回更相关的答案")

if __name__ == "__main__":
    main() 