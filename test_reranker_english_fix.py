#!/usr/bin/env python3
"""
测试英文数据集Reranker修复的脚本
验证rerank()方法调用是否正确
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from xlm.components.retriever.reranker import QwenReranker

def test_english_reranker_fix():
    """测试英文数据集Reranker修复是否有效"""
    
    print("=== 测试英文数据集Reranker修复 ===")
    
    # 模拟英文数据集测试
    query = "What is the revenue growth rate?"
    
    # 创建测试文档，模拟TatQA数据格式
    documents = [
        "The company reported quarterly earnings.",
        "Revenue growth rate increased by 15% year-over-year.",
        "The stock price rose by 5%.",
        "Revenue growth was driven by strong sales performance.",
        "The quarterly report shows positive trends."
    ]
    
    try:
        # 初始化Reranker
        print("1. 初始化Reranker...")
        reranker = QwenReranker(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_quantization=True
        )
        print("✅ Reranker初始化成功")
        
        # 执行重排序（英文数据集调用方式）
        print("2. 执行重排序（英文数据集方式）...")
        reranked_results = reranker.rerank(
            query=query,
            documents=documents,
            batch_size=4
        )
        
        print("3. 分析重排序结果...")
        print(f"查询: {query}")
        print(f"返回结果类型: {type(reranked_results)}")
        print(f"返回结果长度: {len(reranked_results)}")
        
        print("\n重排序结果:")
        for i, (doc_text, score) in enumerate(reranked_results, 1):
            print(f"  {i}. {doc_text[:50]}... (Score: {score:.4f})")
        
        # 检查返回格式是否正确
        if isinstance(reranked_results, list) and len(reranked_results) > 0:
            first_item = reranked_results[0]
            if isinstance(first_item, tuple) and len(first_item) == 2:
                print("\n✅ 返回格式正确：List[Tuple[str, float]]")
                
                # 检查是否包含关键词的文档排在前面
                first_doc = first_item[0]
                if 'revenue' in first_doc.lower() or 'growth' in first_doc.lower():
                    print("✅ 重排序生效：相关文档排在前面")
                    return True
                else:
                    print("⚠️ 重排序可能未生效：相关文档未排在前面")
                    return False
            else:
                print(f"❌ 返回格式错误：期望Tuple[str, float]，实际{type(first_item)}")
                return False
        else:
            print("❌ 返回结果为空或格式错误")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    import torch
    success = test_english_reranker_fix()
    if success:
        print("\n🎉 英文数据集Reranker修复验证成功！")
    else:
        print("\n⚠️ 英文数据集Reranker修复验证失败，需要进一步检查") 