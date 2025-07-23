#!/usr/bin/env python3
"""
Stock Information Extraction Tool
Supports extraction of stock codes and company names in various formats, used for metadata filtering in multi-stage retrieval systems
"""

import re
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict


def load_stock_company_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load stock code and company name mapping file
    
    Returns:
        (stock_company_mapping, company_stock_mapping): Bidirectional mapping dictionaries
    """
    stock_company_mapping = {}
    company_stock_mapping = {}
    
    # Try to load mapping file from multiple paths
    possible_paths = [
        Path("data/astock_code_company_name.csv"),
        Path(__file__).parent.parent.parent / "data" / "astock_code_company_name.csv",
        Path(__file__).parent.parent / "data" / "astock_code_company_name.csv"
    ]
    
    mapping_path = None
    for path in possible_paths:
        if path.exists():
            mapping_path = path
            break
    
    if mapping_path:
        try:
            df = pd.read_csv(mapping_path, encoding='utf-8')
            
            # Build bidirectional mapping
            for _, row in df.iterrows():
                stock_code = str(row['stock_code']).strip()
                company_name = str(row['company_name']).strip()
                
                if stock_code and company_name:
                    # Stock code -> Company name
                    stock_company_mapping[stock_code] = company_name
                    # Company name -> Stock code
                    company_stock_mapping[company_name] = stock_code
            
            print(f"Successfully loaded stock code and company name mapping file: {mapping_path}")
            print(f"Stock code mapping count: {len(stock_company_mapping)}")
            print(f"Company name mapping count: {len(company_stock_mapping)}")
            
        except Exception as e:
            print(f"Failed to load stock code and company name mapping file: {e}")
            print(f"File path: {mapping_path}")
    else:
        print("Stock code and company name mapping file does not exist")
        print(f"Attempted paths: {[str(p) for p in possible_paths]}")
    
    return stock_company_mapping, company_stock_mapping


def extract_stock_info_with_mapping(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Stock information extraction function with mapping file priority
    
    Strategy:
    1. First use mapping file to find known company names and stock codes
    2. Then use regular expressions to extract information not in mapping
    3. Prioritize accurate information from mapping file
    
    Args:
        query: Query text
        
    Returns:
        (company_name, stock_code): Tuple of company name and stock code
    """
    # Load mapping file
    stock_company_mapping, company_stock_mapping = load_stock_company_mapping()
    
    stock_code = None
    company_name = None
    
    # Step 1: Use mapping file to search
    print(f"Using mapping file to search for company information in query...")
    
    # 1.1 Search for stock codes
    stock_patterns = [
        r'[（(](\d{6})[）)]',  # Chinese and English brackets
        r'(\d{6}(?:\.(?:SZ|SH))?)',  # With exchange suffix
        r'(\d{6})',  # Pure numbers
    ]
    
    for pattern in stock_patterns:
        match = re.search(pattern, query)
        if match:
            found_stock_code = match.group(1)
            # Clean stock code (remove exchange suffix)
            pure_code_match = re.search(r'(\d{6})', found_stock_code)
            if pure_code_match:
                found_stock_code = pure_code_match.group(1)
            
            # Check if it's in the mapping file
            if found_stock_code in stock_company_mapping:
                stock_code = found_stock_code
                mapped_company = stock_company_mapping[found_stock_code]
                print(f"Found stock code through mapping file: {found_stock_code} -> Company: {mapped_company}")
                break
    
    # 1.2 Search for company names
    if not company_name:
        # Search for company names mentioned in query within mapping file
        for mapped_company in company_stock_mapping.keys():
            if mapped_company in query:
                company_name = mapped_company
                mapped_stock = company_stock_mapping[mapped_company]
                print(f"Found company name through mapping file: {mapped_company} -> Stock: {mapped_stock}")
                # If stock code not found yet, use mapped stock code
                if not stock_code:
                    stock_code = mapped_stock
                break
    
    # Step 2: If mapping file doesn't find complete information, use regular expressions
    if not stock_code or not company_name:
        print("Mapping file didn't find complete information, using regular expressions to extract...")
        
        # 2.1 Extract stock code (if not found yet)
        if not stock_code:
            for pattern in stock_patterns:
                match = re.search(pattern, query)
                if match:
                    found_stock_code = match.group(1)
                    pure_code_match = re.search(r'(\d{6})', found_stock_code)
                    if pure_code_match:
                        stock_code = pure_code_match.group(1)
                        print(f"Regular expression extracted stock code: {stock_code}")
                        break
        
        # 2.2 Extract company name (if not found yet)
        if not company_name:
            # Prioritize company names with bracket format
            bracket_patterns = [
                r'([^，。？\s]+)[（(]\d{6}(?:\.(?:SZ|SH))?[）)]',  # Company name (stock code)
            ]
            
            for pattern in bracket_patterns:
                match = re.search(pattern, query)
                if match:
                    company_name = match.group(1).strip()
                    print(f"Regular expression extracted company name: {company_name}")
                    break
            
            # If not found, try other formats
            if not company_name:
                # Search for company suffixes
                company_patterns = [
                    r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
                ]
                
                for pattern in company_patterns:
                    match = re.search(pattern, query)
                    if match:
                        potential_company = match.group(1).strip()
                        # Validate if extracted company name is reasonable (length, content, etc.)
                        if len(potential_company) >= 2 and len(potential_company) <= 20:
                            # Check if it contains obvious company suffixes
                            if any(suffix in potential_company for suffix in ['股份', '集团', '公司', '有限', '科技', '网络', '银行', '证券', '保险']):
                                company_name = potential_company
                                print(f"🔍 Regular expression extracted company name: {company_name}")
                                break
    
    # Step 3: Validate and clean results
    if company_name:
        # Clean company name
        company_name = company_name.strip()
        # Remove possible punctuation
        company_name = re.sub(r'[，。？\s]+$', '', company_name)
        
        # Validate if company name is reasonable
        if len(company_name) < 2 or len(company_name) > 20:
            print(f"Company name length is unreasonable: {company_name}")
            company_name = None
    
    if stock_code:
        # Validate stock code format - Fix: Allow 6 digits, don't force leading zeros
        if not re.match(r'^\d{6}$', stock_code):
            # Try to complete leading zeros
            if re.match(r'^\d{1,6}$', stock_code):
                stock_code = stock_code.zfill(6)
                print(f"Completed stock code leading zeros: {stock_code}")
            else:
                print(f"Stock code format is incorrect: {stock_code}")
                stock_code = None
    
    return company_name, stock_code


def extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract stock code and company name from query
    
    支持的格式：
    1. 德赛电池(000049) - 英文括号
    2. 德赛电池（000049） - 中文括号
    3. 德赛电池000049 - 无括号
    4. 000049 - 纯数字
    5. 德赛电池(000049.SZ) - 带交易所后缀
    6. 德赛电池（000049.SH） - 中文括号+后缀
    7. 德赛电池000049.SZ - 无括号+后缀
    8. 德赛电池 000049 - 空格分隔
    9. 首钢股份的业绩表现如何？ - 仅公司名称，无股票代码
    
    Args:
        query: Query text
        
    Returns:
        (company_name, stock_code): Tuple of company name and stock code
    """
    stock_code = None
    company_name = None
    
    # 1. First try to match bracket format (Chinese and English brackets)
    bracket_patterns = [
        r'([^，。？\s]+)[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # Company name (stock code)
        r'[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]',  # Pure (stock code)
    ]
    
    for pattern in bracket_patterns:
        match = re.search(pattern, query)
        if match:
            if len(match.groups()) == 2:
                # First pattern: Company name (stock code)
                company_name = match.group(1)
                stock_code = match.group(2)
            else:
                # Second pattern: Pure (stock code)
                stock_code = match.group(1)
            break
    
    # 2. If bracket format not found, try no-bracket format
    if not stock_code:
        no_bracket_patterns = [
            r'([^，。？\s]+)\s*(\d{6}(?:\.(?:SZ|SH))?)',  # Company name + stock code (supports space)
            r'([^，。？\s]+)(\d{6}(?:\.(?:SZ|SH))?)',  # Company name + stock code (no space)
            r'(\d{6}(?:\.(?:SZ|SH))?)',  # Pure stock code
        ]
        
        for pattern in no_bracket_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 2:
                    # First two patterns: Company name + stock code
                    company_name = match.group(1)
                    stock_code = match.group(2)
                else:
                    # Third pattern: Pure stock code
                    stock_code = match.group(1)
                break
    
    # 3. If company name not found yet, try to extract from before/after stock code
    if stock_code and not company_name:
        # Search for possible company name before stock code
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
    
    # 4. If stock code not found, try to extract company name only (new step)
    if not company_name:
        company_patterns = [
            r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
            r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险)[^，。？\s]*)'
        ]
        
        for pattern in company_patterns:
            company_match = re.search(pattern, query)
            if company_match:
                company_name = company_match.group(1)
                break
    
    # 5. Clean stock code (remove exchange suffix for index matching)
    if stock_code:
        # Extract pure 6-digit code
        pure_code_match = re.search(r'(\d{6})', stock_code)
        if pure_code_match:
            stock_code = pure_code_match.group(1)
    
    return company_name, stock_code


def extract_stock_info_simple(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Simplified version of stock information extraction function, for backward compatibility
    
    Args:
        query: Query text
        
    Returns:
        (company_name, stock_code): Tuple of company name and stock code
    """
    stock_code = None
    company_name = None
    
    # Extract stock code - Match 6 digits (supports Chinese and English brackets)
    stock_patterns = [
        r'[（(](\d{6})[）)]',  # Chinese and English brackets
        r'(\d{6})',  # Pure numbers
    ]
    
    for pattern in stock_patterns:
        stock_match = re.search(pattern, query)
        if stock_match:
            stock_code = stock_match.group(1)
            break
    
    # Extract company name - Match company suffixes
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
    Extract year/quarter/date information from query, adapts to metadata report_date field
    Supports formats: 2021, 2021年, 2021年度, 2021Q1, 2021年第一季度, 2021-03-31, etc.
    """
    # Year
    m = re.search(r'(20\d{2})[年度]?', query)
    if m:
        return m.group(1)
    # Year + quarter
    m = re.search(r'(20\d{2})[年\s]*[第]?(1|2|3|4)[季度Q]', query)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r'(20\d{2})Q([1-4])', query)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    # Year-month-day
    m = re.search(r'(20\d{2}-\d{1,2}-\d{1,2})', query)
    if m:
        return m.group(1)
    # Extract year only
    m = re.search(r'(20\d{2})', query)
    if m:
        return m.group(1)
    return None


def test_extraction():
    """Test extraction functions"""
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