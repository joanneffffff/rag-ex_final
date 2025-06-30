#!/usr/bin/env python3
"""
测试股票代码提取问题
分析为什么第一个查询能找到股票代码，第二个查询不能
并提供兼容不同条件的解决方案
"""

import re

def test_stock_code_extraction():
    """测试股票代码提取"""
    print("=== 测试股票代码提取 ===")
    
    # 测试查询 - 包含各种不同格式
    test_queries = [
        "德赛电池(000049)的下一季度收益预测如何？",  # 英文括号
        "德赛电池（000049）2021年利润持续增长的主要原因是什么？",  # 中文括号
        "德赛电池(000049)的业绩如何？",  # 英文括号
        "德赛电池（000049）的财务表现？",  # 中文括号
        "000049的股价走势",  # 无括号，纯数字
        "德赛电池000049的营收情况",  # 无括号，公司名+数字
        "德赛电池 000049 的财务数据",  # 空格分隔
        "德赛电池000049.SZ的业绩",  # 带交易所后缀
        "德赛电池(000049.SZ)的股价",  # 英文括号+后缀
        "德赛电池（000049.SH）的营收",  # 中文括号+后缀
        "德赛电池000049.SH的财务表现",  # 无括号+后缀
        "000049.SZ的股价走势",  # 纯数字+后缀
    ]
    
    print("当前正则表达式: r'\\((\d{6})\\)'")
    print("问题：只匹配英文括号()，不匹配中文括号（）和其他格式")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"查询 {i}: {query}")
        
        # 当前的正则表达式（只匹配英文括号）
        current_pattern = r'\((\d{6})\)'
        stock_match = re.search(current_pattern, query)
        
        if stock_match:
            stock_code = stock_match.group(1)
            print(f"  ✅ 当前正则匹配成功: {stock_code}")
        else:
            print(f"  ❌ 当前正则匹配失败")
        
        # 修复后的正则表达式（兼容多种格式）
        fixed_pattern = r'[（(](\d{6}(?:\.(?:SZ|SH))?)[）)]'
        stock_match_fixed = re.search(fixed_pattern, query)
        
        if stock_match_fixed:
            stock_code_fixed = stock_match_fixed.group(1)
            print(f"  ✅ 修复后正则匹配成功: {stock_code_fixed}")
        else:
            print(f"  ❌ 修复后正则匹配失败")
        
        print()

def test_company_name_extraction():
    """测试公司名称提取"""
    print("=== 测试公司名称提取 ===")
    
    test_queries = [
        "德赛电池(000049)的下一季度收益预测如何？",
        "德赛电池（000049）2021年利润持续增长的主要原因是什么？",
        "用友网络(600588)的财务表现如何？",
        "中国平安（601318）的保险业务发展情况？",
    ]
    
    # 当前的公司名称提取模式
    company_patterns = [
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险)[^，。？\s]*)'
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"查询 {i}: {query}")
        
        company_name = None
        for pattern in company_patterns:
            company_match = re.search(pattern, query)
            if company_match:
                company_name = company_match.group(1)
                break
        
        if company_name:
            print(f"  ✅ 提取到公司名称: {company_name}")
        else:
            print(f"  ❌ 未提取到公司名称")
        
        print()

def create_compatible_extraction_function():
    """创建兼容不同条件的提取函数"""
    print("=== 兼容不同条件的提取函数 ===")
    
    def extract_stock_info_compatible(query: str):
        """
        兼容不同条件的股票代码和公司名称提取
        
        支持的格式：
        1. 德赛电池(000049) - 英文括号
        2. 德赛电池（000049） - 中文括号
        3. 德赛电池000049 - 无括号
        4. 000049 - 纯数字
        5. 德赛电池(000049.SZ) - 带交易所后缀
        6. 德赛电池（000049.SH） - 中文括号+后缀
        7. 德赛电池000049.SZ - 无括号+后缀
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
                r'([^，。？\s]+)(\d{6}(?:\.(?:SZ|SH))?)',  # 公司名+股票代码
                r'(\d{6}(?:\.(?:SZ|SH))?)',  # 纯股票代码
            ]
            
            for pattern in no_bracket_patterns:
                match = re.search(pattern, query)
                if match:
                    if len(match.groups()) == 2:
                        # 第一个模式：公司名+股票代码
                        company_name = match.group(1)
                        stock_code = match.group(2)
                    else:
                        # 第二个模式：纯股票代码
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
    
    # 测试兼容函数
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
    
    print("兼容不同条件的提取函数:")
    print("支持格式：")
    print("1. 公司名(股票代码) - 英文括号")
    print("2. 公司名（股票代码） - 中文括号")
    print("3. 公司名股票代码 - 无括号")
    print("4. 股票代码 - 纯数字")
    print("5. 带交易所后缀的各种格式")
    print()
    
    for query in test_queries:
        company_name, stock_code = extract_stock_info_compatible(query)
        print(f"查询: {query}")
        print(f"  公司名称: {company_name}")
        print(f"  股票代码: {stock_code}")
        print()
    
    return extract_stock_info_compatible

def generate_compatible_fix_code():
    """生成兼容不同条件的修复代码"""
    print("=== 兼容不同条件的修复代码 ===")
    
    fix_code = '''
def extract_stock_info_compatible(query: str):
    """
    兼容不同条件的股票代码和公司名称提取
    
    支持的格式：
    1. 德赛电池(000049) - 英文括号
    2. 德赛电池（000049） - 中文括号
    3. 德赛电池000049 - 无括号
    4. 000049 - 纯数字
    5. 德赛电池(000049.SZ) - 带交易所后缀
    6. 德赛电池（000049.SH） - 中文括号+后缀
    7. 德赛电池000049.SZ - 无括号+后缀
    """
    stock_code = None
    company_name = None
    
    # 1. 首先尝试匹配带括号的格式（中英文括号）
    bracket_patterns = [
        r'([^，。？\\s]+)[（(](\\d{6}(?:\\.(?:SZ|SH))?)[）)]',  # 公司名(股票代码)
        r'[（(](\\d{6}(?:\\.(?:SZ|SH))?)[）)]',  # 纯(股票代码)
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
            r'([^，。？\\s]+)(\\d{6}(?:\\.(?:SZ|SH))?)',  # 公司名+股票代码
            r'(\\d{6}(?:\\.(?:SZ|SH))?)',  # 纯股票代码
        ]
        
        for pattern in no_bracket_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 2:
                    # 第一个模式：公司名+股票代码
                    company_name = match.group(1)
                    stock_code = match.group(2)
                else:
                    # 第二个模式：纯股票代码
                    stock_code = match.group(1)
                break
    
    # 3. 如果还没有找到公司名，尝试从股票代码前后提取
    if stock_code and not company_name:
        # 在股票代码前查找可能的公司名
        stock_code_escaped = re.escape(stock_code)
        company_patterns = [
            rf'([^，。？\\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))\\s*{stock_code_escaped}',
            rf'([^，。？\\s]+)\\s*{stock_code_escaped}',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query)
            if match:
                company_name = match.group(1)
                break
    
    # 4. 清理股票代码（移除交易所后缀用于索引匹配）
    if stock_code:
        # 提取纯6位数字代码
        pure_code_match = re.search(r'(\\d{6})', stock_code)
        if pure_code_match:
            stock_code = pure_code_match.group(1)
    
    return company_name, stock_code

# 在UI中使用：
# company_name, stock_code = extract_stock_info_compatible(question)
'''
    
    print(fix_code)
    
    # 需要修改的文件
    files_to_fix = [
        "xlm/ui/optimized_rag_ui.py",
        "xlm/ui/optimized_rag_ui_with_multi_stage.py"
    ]
    
    print("需要修改的文件:")
    for file in files_to_fix:
        print(f"  - {file}")
    
    return fix_code

def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===")
    
    edge_cases = [
        "德赛电池",  # 只有公司名
        "000049",  # 只有股票代码
        "德赛电池的业绩",  # 没有股票代码
        "德赛电池(000049)和用友网络(600588)的对比",  # 多个股票代码
        "德赛电池（000049）的业绩如何？用友网络呢？",  # 混合格式
        "德赛电池000049.SZ和用友网络600588.SH",  # 多个带后缀
    ]
    
    extract_func = create_compatible_extraction_function()
    
    for query in edge_cases:
        company_name, stock_code = extract_func(query)
        print(f"查询: {query}")
        print(f"  公司名称: {company_name}")
        print(f"  股票代码: {stock_code}")
        print()

def main():
    """主函数"""
    print("开始股票代码提取问题分析和兼容性解决方案...")
    
    # 1. 测试当前提取逻辑
    test_stock_code_extraction()
    
    # 2. 测试公司名称提取
    test_company_name_extraction()
    
    # 3. 创建兼容不同条件的提取函数
    extract_func = create_compatible_extraction_function()
    
    # 4. 测试边界情况
    test_edge_cases()
    
    # 5. 生成兼容的修复代码
    generate_compatible_fix_code()
    
    print("=== 问题总结和解决方案 ===")
    print("问题原因：")
    print("1. 当前股票代码提取正则表达式 r'\\((\d{6})\\)' 只匹配英文括号()")
    print("2. 不支持中文括号（）、无括号格式、带交易所后缀等")
    print("3. 缺乏对不同输入格式的兼容性")
    print()
    print("解决方案：")
    print("1. 使用多层匹配策略，优先匹配括号格式，再匹配无括号格式")
    print("2. 支持中英文括号、带交易所后缀、纯数字等多种格式")
    print("3. 提供健壮的公司名称提取逻辑")
    print("4. 清理股票代码，统一为6位数字格式用于索引匹配")

if __name__ == "__main__":
    main() 