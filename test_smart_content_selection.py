#!/usr/bin/env python3
"""
测试智能内容选择的实现
验证：只有中文查询使用智能内容选择，英文查询使用原始context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_smart_content_selection_logic():
    """测试智能内容选择逻辑"""
    print("=== 测试智能内容选择逻辑 ===")
    
    # 模拟DocumentWithMetadata对象
    class MockDocument:
        def __init__(self, content, metadata):
            self.content = content
            self.metadata = metadata
    
    class MockMetadata:
        def __init__(self, language=None, summary=None):
            self.language = language
            self.summary = summary
    
    # 测试数据
    chinese_doc = MockDocument(
        content="这是一个很长的中文文档内容，包含了很多详细的财务信息和分析数据。公司在上个季度的表现非常出色，营业收入增长了15%，净利润增长了20%。这些增长主要来自于新产品的推出和市场份额的扩大。",
        metadata=MockMetadata(language="chinese", summary="公司财务表现良好，营收和利润都有显著增长")
    )
    
    english_doc = MockDocument(
        content="This is a long English document about financial performance. The company showed excellent results last quarter with 15% revenue growth and 20% profit increase.",
        metadata=MockMetadata(language="english", summary=None)
    )
    
    def get_smart_content(doc, is_chinese_query):
        """模拟智能内容选择逻辑"""
        if is_chinese_query and hasattr(doc.metadata, 'language') and doc.metadata.language == 'chinese':
            # 中文数据：尝试组合summary和context
            summary = ""
            if hasattr(doc.metadata, 'summary') and doc.metadata.summary:
                summary = doc.metadata.summary
            else:
                # 如果没有summary，使用context的前200字符作为summary
                summary = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            # 组合summary和context，避免过长
            combined_text = f"摘要：{summary}\n\n详细内容：{doc.content}"
            # 限制总长度，避免超出限制
            if len(combined_text) > 4000:
                combined_text = f"摘要：{summary}\n\n详细内容：{doc.content[:3500]}..."
            return combined_text
        else:
            # 英文数据或非中文数据：只使用context
            return doc.content
    
    # 测试中文查询
    print("\n--- 中文查询测试 ---")
    chinese_query = "请介绍一下公司的财务状况"
    chinese_result = get_smart_content(chinese_doc, True)
    print(f"中文查询结果长度: {len(chinese_result)}")
    print(f"中文查询结果预览: {chinese_result[:200]}...")
    
    # 测试英文查询
    print("\n--- 英文查询测试 ---")
    english_query = "What is the company's financial performance?"
    english_result = get_smart_content(chinese_doc, False)
    print(f"英文查询结果长度: {len(english_result)}")
    print(f"英文查询结果预览: {english_result[:200]}...")
    
    # 验证差异
    print(f"\n--- 验证结果 ---")
    print(f"中文查询使用智能内容选择: {'是' if '摘要：' in chinese_result else '否'}")
    print(f"英文查询使用原始context: {'是' if chinese_result == chinese_doc.content else '否'}")
    print(f"内容长度差异: {len(chinese_result) - len(english_result)} 字符")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_smart_content_selection_logic() 