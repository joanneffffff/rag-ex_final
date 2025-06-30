#!/usr/bin/env python3
"""
股票信息提取工具
支持多种格式的股票代码和公司名称提取，用于多阶段检索系统的元数据过滤
"""

import re
from typing import Tuple, Optional


def extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从查询中提取股票代码和公司名称
    
    支持的格式：
    1. 德赛电池(000049) - 英文括号
    2. 德赛电池（000049） - 中文括号
    3. 德赛电池000049 - 无括号
    4. 000049 - 纯数字
    5. 德赛电池(000049.SZ) - 带交易所后缀
    6. 德赛电池（000049.SH） - 中文括号+后缀
    7. 德赛电池000049.SZ - 无括号+后缀
    8. 德赛电池 000049 - 空格分隔
    
    Args:
        query: 查询文本
        
    Returns:
        (company_name, stock_code): 公司名称和股票代码的元组
    """
    stock_code = None
    company_name = None
    
    # 1. 首先尝试匹配带括号的格式（中英文括号）
    bracket_patterns = [
        r'([^，。？\s]+)[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 公司名(股票代码)
        r'[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # 纯(股票代码)
    ]
    
    for pattern in bracket_patterns:
        match = re.search(pattern, query)
        if match:
            if len(match.groups()) == 2:
                # 第一个模式：公司名(股票代码)
                company_name = match.group(1)
                stock_code = match.group(2)
            else:
                # 第二个模式：纯(股票代码)
                stock_code = match.group(1)
            break
    
    # 2. 如果没有找到括号格式，尝试无括号格式
    if not stock_code:
        no_bracket_patterns = [
            r'([^，。？\s]+)\s*(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码（支持空格）
            r'([^，。？\s]+)(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码（无空格）
            r'(\d{6}(?:\.(?:SZ|SH))?)',  # 纯股票代码
        ]
        
        for pattern in no_bracket_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 2:
                    # 前两个模式：公司名+股票代码
                    company_name = match.group(1)
                    stock_code = match.group(2)
                else:
                    # 第三个模式：纯股票代码
                    stock_code = match.group(1)
                break
    
    # 3. 如果还没有找到公司名，尝试从股票代码前后提取
    if stock_code and not company_name:
        # 在股票代码前查找可能的公司名
        stock_code_escaped = re.escape(stock_code)
        company_patterns = [
            rf'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))\s*{stock_code_escaped}',
            rf'([^，。？\s]+)\s*{stock_code_escaped}',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query)
            if match:
                company_name = match.group(1)
                break
    
    # 4. 清理股票代码（移除交易所后缀用于索引匹配）
    if stock_code:
        # 提取纯6位数字代码
        pure_code_match = re.search(r'(\d{6})', stock_code)
        if pure_code_match:
            stock_code = pure_code_match.group(1)
    
    return company_name, stock_code


def extract_stock_info_simple(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    简化版本的股票信息提取函数，用于向后兼容
    
    Args:
        query: 查询文本
        
    Returns:
        (company_name, stock_code): 公司名称和股票代码的元组
    """
    stock_code = None
    company_name = None
    
    # 提取股票代码 - 匹配6位数字（支持中英文括号）
    stock_patterns = [
        r'[（(](\d{6})[）)]',  # 中英文括号
        r'(\d{6})',  # 纯数字
    ]
    
    for pattern in stock_patterns:
        stock_match = re.search(pattern, query)
        if stock_match:
            stock_code = stock_match.group(1)
            break
    
    # 提取公司名称 - 匹配公司后缀
    company_patterns = [
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险)[^，。？\s]*)'
    ]
    
    for pattern in company_patterns:
        company_match = re.search(pattern, query)
        if company_match:
            company_name = company_match.group(1)
            break
    
    return company_name, stock_code


def extract_report_date(query: str) -> Optional[str]:
    """
    从查询中提取年份/季度/日期等信息，适配元数据report_date字段
    支持格式：2021、2021年、2021年度、2021Q1、2021年第一季度、2021-03-31等
    """
    # 年份
    m = re.search(r'(20\d{2})[年度]?', query)
    if m:
        return m.group(1)
    # 年+季度
    m = re.search(r'(20\d{2})[年\s]*[第]?(1|2|3|4)[季度Q]', query)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r'(20\d{2})Q([1-4])', query)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    # 年-月-日
    m = re.search(r'(20\d{2}-\d{1,2}-\d{1,2})', query)
    if m:
        return m.group(1)
    # 只提取年
    m = re.search(r'(20\d{2})', query)
    if m:
        return m.group(1)
    return None


def test_extraction():
    """测试提取函数"""
    test_queries = [
        "德赛电池(000049)的下一季度收益预测如何？",
        "德赛电池（000049）2021年利润持续增长的主要原因是什么？",
        "德赛电池(000049.SZ)的业绩如何？",
        "德赛电池（000049.SH）的财务表现？",
        "000049的股价走势",
        "德赛电池000049的营收情况",
        "德赛电池 000049 的财务数据",
        "德赛电池000049.SZ的业绩",
        "用友网络(600588)的财务表现如何？",
        "中国平安（601318）的保险业务发展情况？",
    ]
    
    print("=== 测试股票信息提取 ===")
    print("支持的格式：")
    print("1. 公司名(股票代码) - 英文括号")
    print("2. 公司名（股票代码） - 中文括号")
    print("3. 公司名股票代码 - 无括号")
    print("4. 股票代码 - 纯数字")
    print("5. 带交易所后缀的各种格式")
    print()
    
    for query in test_queries:
        company_name, stock_code = extract_stock_info(query)
        print(f"查询: {query}")
        print(f"  公司名称: {company_name}")
        print(f"  股票代码: {stock_code}")
        print()


if __name__ == "__main__":
    test_extraction() 