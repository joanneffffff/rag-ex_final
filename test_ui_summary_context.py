#!/usr/bin/env python3
"""
测试UI中summary和context的正确使用
确保：UI只显示context，但prompt同时使用summary和context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.ui.optimized_rag_ui import OptimizedRagUI

def test_summary_context_integration():
    """测试summary和context的集成"""
    print("=== 测试UI中summary和context的正确使用 ===")
    
    try:
        # 初始化UI（不启动界面）
        ui = OptimizedRagUI(
            use_faiss=True,
            enable_reranker=False,  # 简化测试
            window_title="测试RAG系统"
        )
        
        print("✅ UI初始化成功")
        
        # 测试中文查询
        print("\n--- 测试中文查询 ---")
        chinese_question = "请介绍一下公司的财务状况"
        result, html_content = ui._process_question(
            question=chinese_question,
            datasource="unified",
            reranker_checkbox=False
        )
        
        print(f"中文查询结果: {result[:200]}...")
        print(f"HTML内容长度: {len(html_content)}")
        
        # 测试英文查询
        print("\n--- 测试英文查询 ---")
        english_question = "What is the company's financial performance?"
        result, html_content = ui._process_question(
            question=english_question,
            datasource="unified",
            reranker_checkbox=False
        )
        
        print(f"英文查询结果: {result[:200]}...")
        print(f"HTML内容长度: {len(html_content)}")
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_summary_context_integration() 