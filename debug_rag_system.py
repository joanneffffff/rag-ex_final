#!/usr/bin/env python3
"""
调试RAG系统的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.ui.optimized_rag_ui import OptimizedRagUI

def main():
    print("=== 开始调试RAG系统 ===")
    
    try:
        # 初始化RAG UI
        print("1. 初始化OptimizedRagUI...")
        rag_ui = OptimizedRagUI(
            use_faiss=True,
            enable_reranker=True,
            use_existing_embedding_index=False  # 强制重新计算嵌入
        )
        print("✅ OptimizedRagUI初始化完成")
        
        # 测试查询
        test_query = "How was internally developed software capitalised?"
        print(f"\n2. 测试查询: {test_query}")
        
        # 直接调用统一RAG处理
        answer, context = rag_ui._unified_rag_processing(
            question=test_query,
            language="en",
            reranker_checkbox=True
        )
        
        print(f"\n3. 查询结果:")
        print(f"答案: {answer}")
        print(f"上下文: {context[:200]}..." if len(context) > 200 else f"上下文: {context}")
        
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 