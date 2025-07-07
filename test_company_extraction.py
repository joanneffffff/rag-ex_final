#!/usr/bin/env python3
"""
测试公司名称提取逻辑
"""

from xlm.utils.stock_info_extractor import extract_stock_info

def test_company_extraction():
    """测试公司名称提取"""
    
    test_queries = [
        "中兴通讯在AI时代如何布局通信能力提升，以及其对公司未来业绩的影响是什么？",
        "中兴通讯（000063）在AI时代如何布局通信能力提升？",
        "中兴通讯(000063)在AI时代如何布局通信能力提升？",
        "中兴通讯000063在AI时代如何布局通信能力提升？",
        "林洋能源（601222）在2020年上半年业绩表现如何？",
        "以及其对公司未来业绩的影响是什么？",  # 这个可能被误匹配
    ]
    
    print("=" * 80)
    print("公司名称提取测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. 查询: {query}")
        
        company_name, stock_code = extract_stock_info(query)
        
        print(f"   提取结果:")
        print(f"     公司名称: {company_name}")
        print(f"     股票代码: {stock_code}")
        
        # 分析提取逻辑
        if company_name:
            print(f"   ✅ 成功提取公司名称: {company_name}")
        else:
            print(f"   ❌ 未提取到公司名称")
            
        if stock_code:
            print(f"   ✅ 成功提取股票代码: {stock_code}")
        else:
            print(f"   ❌ 未提取到股票代码")
        
        print()

if __name__ == "__main__":
    test_company_extraction() 