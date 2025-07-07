#!/usr/bin/env python3
"""
测试语言检测功能
"""

from langdetect import detect, LangDetectException

def test_language_detection():
    """测试语言检测"""
    
    test_queries = [
        "中兴通讯在AI时代如何布局通信能力提升，以及其对公司未来业绩的影响是什么？",
        "林洋能源（601222）在2020年上半年业绩表现如何，有哪些驱动因素？",
        "What is the total assets as of June 30, 2019?",
        "AI时代的中兴通讯",
        "中兴通讯AI布局",
        "中兴通讯在AI时代",
        "AI technology in China",
        "中国AI技术发展",
        "AI and 5G technology",
        "5G和AI技术"
    ]
    
    print("=" * 80)
    print("语言检测测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        try:
            lang = detect(query)
            print(f"{i:2d}. 查询: {query}")
            print(f"    检测结果: {lang}")
            
            # 检查是否包含中文字符
            chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
            total_chars = len([char for char in query if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
            
            if total_chars > 0:
                chinese_ratio = chinese_chars / total_chars
                print(f"    中文字符数: {chinese_chars}, 总字符数: {total_chars}, 中文比例: {chinese_ratio:.2f}")
            else:
                print(f"    中文字符数: {chinese_chars}, 总字符数: {total_chars}")
            
            # 判断是否应该被识别为中文
            should_be_chinese = chinese_chars > 0 and (chinese_ratio > 0.3 if total_chars > 0 else True)
            print(f"    应该识别为中文: {should_be_chinese}")
            print(f"    当前识别正确: {lang.startswith('zh') == should_be_chinese}")
            print()
            
        except LangDetectException as e:
            print(f"{i:2d}. 查询: {query}")
            print(f"    检测失败: {e}")
            print()

if __name__ == "__main__":
    test_language_detection() 