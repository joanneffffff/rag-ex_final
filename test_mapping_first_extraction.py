#!/usr/bin/env python3
"""
测试映射优先的公司信息提取函数
"""

from xlm.utils.stock_info_extractor import extract_stock_info_with_mapping, extract_stock_info

def test_mapping_first_extraction():
    """测试映射优先的提取函数"""
    
    test_queries = [
        # 测试用例1: 映射文件中存在的公司
        "中兴通讯在AI时代如何布局通信能力提升，以及其对公司未来业绩的影响是什么？",
        "中兴通讯（000063）在AI时代如何布局通信能力提升？",
        "中兴通讯(000063)在AI时代如何布局通信能力提升？",
        "中兴通讯000063在AI时代如何布局通信能力提升？",
        
        # 测试用例2: 映射文件中存在的公司
        "林洋能源（601222）在2020年上半年业绩表现如何？",
        "林洋能源在2020年上半年业绩表现如何？",
        
        # 测试用例3: 映射文件中存在的公司
        "宝莱特（300246）的财务状况如何？",
        "宝莱特的财务状况如何？",
        
        # 测试用例4: 不存在的公司（应该被过滤掉）
        "以及其对公司未来业绩的影响是什么？",
        "这个公司的表现如何？",
        
        # 测试用例5: 映射文件中存在的公司
        "海融科技（300915）的发展前景如何？",
        "海融科技的发展前景如何？",
    ]
    
    print("=" * 80)
    print("映射优先的公司信息提取测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 查询: {query}")
        print("   " + "-" * 60)
        
        # 使用新的映射优先函数
        print("   【映射优先提取】:")
        company_name, stock_code = extract_stock_info_with_mapping(query)
        
        if company_name:
            print(f"   ✅ 公司名称: {company_name}")
        else:
            print(f"   ❌ 未提取到公司名称")
            
        if stock_code:
            print(f"   ✅ 股票代码: {stock_code}")
        else:
            print(f"   ❌ 未提取到股票代码")
        
        # 使用原来的函数作为对比
        print("   【原函数对比】:")
        old_company_name, old_stock_code = extract_stock_info(query)
        
        if old_company_name:
            print(f"   ✅ 公司名称: {old_company_name}")
        else:
            print(f"   ❌ 未提取到公司名称")
            
        if old_stock_code:
            print(f"   ✅ 股票代码: {old_stock_code}")
        else:
            print(f"   ❌ 未提取到股票代码")
        
        # 比较结果
        if company_name != old_company_name or stock_code != old_stock_code:
            print("   🔄 结果不同！")
        else:
            print("   ✅ 结果一致")

if __name__ == "__main__":
    test_mapping_first_extraction() 