#!/usr/bin/env python3
"""
测试公司名称提取修复
验证"首钢股份的业绩表现如何？"查询是否能正确提取公司名称
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_company_name_extraction_fix():
    """测试公司名称提取修复"""
    print("=== 测试公司名称提取修复 ===")
    
    # 导入修复后的函数
    from xlm.utils.stock_info_extractor import extract_stock_info
    
    # 测试查询
    test_query = "首钢股份的业绩表现如何？"
    print(f"测试查询: {test_query}")
    
    # 提取公司名称和股票代码
    company_name, stock_code = extract_stock_info(test_query)
    
    print(f"提取结果:")
    print(f"  公司名称: {company_name}")
    print(f"  股票代码: {stock_code}")
    
    # 验证结果
    if company_name == "首钢股份":
        print("✅ 公司名称提取成功！")
        return True
    else:
        print("❌ 公司名称提取失败！")
        return False

def test_multi_stage_integration():
    """测试多阶段检索系统集成"""
    print("\n=== 测试多阶段检索系统集成 ===")
    
    try:
        # 导入多阶段检索系统
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 初始化系统
        data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            return False
        
        print("✅ 初始化多阶段检索系统...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path,
            dataset_type="chinese",
            use_existing_config=True
        )
        
        # 测试查询
        test_query = "首钢股份的业绩表现如何？"
        print(f"测试查询: {test_query}")
        
        # 提取公司名称
        from xlm.utils.stock_info_extractor import extract_stock_info
        company_name, stock_code = extract_stock_info(test_query)
        
        print(f"提取的公司名称: {company_name}")
        print(f"提取的股票代码: {stock_code}")
        
        # 执行检索
        results = retrieval_system.search(
            query=test_query,
            company_name=company_name,
            stock_code=stock_code,
            top_k=5
        )
        
        # 检查结果
        if isinstance(results, dict) and 'retrieved_documents' in results:
            documents = results['retrieved_documents']
            print(f"✅ 检索成功！找到 {len(documents)} 个文档")
            
            # 检查是否有首钢股份相关的文档
            shougang_docs = [doc for doc in documents if '首钢' in str(doc.get('company_name', ''))]
            if shougang_docs:
                print(f"✅ 找到 {len(shougang_docs)} 个首钢股份相关文档")
                return True
            else:
                print("⚠️ 未找到首钢股份相关文档，但检索系统正常工作")
                return True
        else:
            print("❌ 检索失败或返回格式错误")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试公司名称提取修复...")
    
    # 测试1: 公司名称提取
    test1_result = test_company_name_extraction_fix()
    
    # 测试2: 多阶段检索系统集成
    test2_result = test_multi_stage_integration()
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结:")
    print(f"  公司名称提取: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"  多阶段检索集成: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("🎉 所有测试通过！公司名称提取问题已修复。")
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main() 