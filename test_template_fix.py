#!/usr/bin/env python3
"""
测试模板修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
from pathlib import Path

def test_template_fix():
    """测试模板修复效果"""
    
    print("🧪 测试模板修复效果")
    print("=" * 50)
    
    # 初始化多阶段检索系统
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    retrieval_system = MultiStageRetrievalSystem(
        data_path=data_path,
        dataset_type="chinese",
        use_existing_config=True
    )
    
    # 测试查询
    query = "德赛电池(000049)的下一季度收益预测如何？"
    
    print(f"📋 测试查询: {query}")
    
    try:
        # 执行检索
        results = retrieval_system.search(
            query=query,
            company_name="德赛电池",
            stock_code="000049",
            top_k=5
        )
        
        if 'llm_answer' in results and results['llm_answer']:
            print("✅ 模板修复成功！")
            print(f"🤖 生成的答案长度: {len(results['llm_answer'])} 字符")
            print(f"📝 答案前200字符: {results['llm_answer'][:200]}...")
        else:
            print("❌ 模板仍有问题")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_template_fix() 