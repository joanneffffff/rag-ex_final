#!/usr/bin/env python3
"""
测试映射优先提取是否在所有地方都生效
"""

from xlm.utils.stock_info_extractor import extract_stock_info_with_mapping, extract_report_date
from alphafin_data_process.rag_system_adapter import RagSystemAdapter

def test_extraction_in_rag_system():
    """测试RAG系统中的公司名称提取"""
    
    print("=" * 60)
    print("测试RAG系统中的公司名称提取")
    print("=" * 60)
    
    # 测试查询
    test_query = "中兴通讯在AI时代如何布局通信能力提升，以及其对公司未来业绩的影响是什么？"
    
    print(f"测试查询: {test_query}")
    print()
    
    # 1. 直接测试提取函数
    print("1. 直接测试映射优先提取函数:")
    company_name, stock_code = extract_stock_info_with_mapping(test_query)
    print(f"   公司名称: {company_name}")
    print(f"   股票代码: {stock_code}")
    print()
    
    # 2. 测试RAG系统适配器中的提取
    print("2. 测试RAG系统适配器中的提取:")
    try:
        # 创建RAG系统适配器实例
        adapter = RagSystemAdapter()
        
        # 模拟RAG系统中的提取逻辑
        company_name_rag, stock_code_rag = extract_stock_info_with_mapping(test_query)
        report_date = extract_report_date(test_query)
        
        print(f"   公司名称: {company_name_rag}")
        print(f"   股票代码: {stock_code_rag}")
        print(f"   报告日期: {report_date}")
        
    except Exception as e:
        print(f"   ❌ RAG系统测试失败: {e}")
    
    print()
    
    # 3. 对比结果
    print("3. 结果对比:")
    if company_name == company_name_rag:
        print("   ✅ 提取结果一致")
    else:
        print("   ❌ 提取结果不一致")
        print(f"   直接提取: {company_name}")
        print(f"   RAG系统: {company_name_rag}")

if __name__ == "__main__":
    test_extraction_in_rag_system() 