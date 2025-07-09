#!/usr/bin/env python3
import sys
sys.path.append('.')

from xlm.utils.stock_info_extractor import extract_stock_info_with_mapping, load_stock_company_mapping

# 测试查询
test_queries = [
    "中国中冶(601618.SH)在2017年6月30日的资产负债表中，其净资产是多少？",
    "阿里巴巴集团2024财年一季度财报中，菜鸟集团的营收情况如何？",
    "双良节能宣布设立包头双良并投资单晶硅棒、硅片项目后，其股价的短期涨跌趋势如何？",
    "瀚蓝环境（600323）在2020年度财报中的关键业绩指标有哪些？",
    "恩捷股份（002812）在2021年半年报中表现出哪些业绩亮点？"
]

print("=== 测试元数据提取和映射效果 ===\n")

# 加载映射文件
stock_company_mapping, company_stock_mapping = load_stock_company_mapping()
print(f"映射文件统计:")
print(f"  股票代码映射数量: {len(stock_company_mapping)}")
print(f"  公司名称映射数量: {len(company_stock_mapping)}")

# 检查特定股票代码
test_stocks = ['601618', '600323', '002812']
print(f"\n检查测试股票代码:")
for stock in test_stocks:
    if stock in stock_company_mapping:
        print(f"  {stock} -> {stock_company_mapping[stock]}")
    else:
        print(f"  {stock} -> 未找到")

print(f"\n=== 测试查询提取效果 ===")
for i, query in enumerate(test_queries, 1):
    print(f"\n查询 {i}: {query}")
    company_name, stock_code = extract_stock_info_with_mapping(query)
    print(f"  提取结果: 公司={company_name}, 股票={stock_code}")
    
    # 检查映射
    if stock_code and stock_code in stock_company_mapping:
        mapped_company = stock_company_mapping[stock_code]
        print(f"  映射验证: {stock_code} -> {mapped_company}")
    elif company_name and company_name in company_stock_mapping:
        mapped_stock = company_stock_mapping[company_name]
        print(f"  映射验证: {company_name} -> {mapped_stock}")
    else:
        print(f"  映射验证: 未找到对应映射") 