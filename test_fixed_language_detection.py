#!/usr/bin/env python3
"""
测试修复后的语言检测逻辑
"""

from langdetect import detect, LangDetectException

def test_fixed_language_detection():
    """测试修复后的语言检测逻辑"""
    
    test_queries = [
        "5G和AI技术",  # 这个被langdetect误判为vi
        "中兴通讯在AI时代如何布局通信能力提升，以及其对公司未来业绩的影响是什么？",
        "林洋能源（601222）在2020年上半年业绩表现如何，有哪些驱动因素？",
        "AI时代的中兴通讯",
        "中兴通讯AI布局"
    ]
    
    print("=" * 80)
    print("修复后的语言检测测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. 查询: {query}")
        
        # 原始langdetect结果
        try:
            lang = detect(query)
            print(f"   原始langdetect结果: {lang}")
        except LangDetectException as e:
            print(f"   原始langdetect失败: {e}")
            lang = 'unknown'
        
        # 检查是否包含中文字符
        chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
        total_chars = len([char for char in query if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
        
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            print(f"   中文字符数: {chinese_chars}, 总字符数: {total_chars}, 中文比例: {chinese_ratio:.2f}")
        else:
            print(f"   中文字符数: {chinese_chars}, 总字符数: {total_chars}")
            chinese_ratio = 0
        
        # 修复后的逻辑
        try:
            # 如果包含中文字符且中文比例超过30%，或者langdetect检测为中文，则认为是中文
            if chinese_chars > 0 and (chinese_ratio > 0.3 or lang.startswith('zh')):
                fixed_language = 'zh'
            else:
                fixed_language = 'en'
        except:
            # 如果langdetect失败，使用字符检测
            chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')
            fixed_language = 'zh' if chinese_chars > 0 else 'en'
        
        print(f"   修复后结果: {fixed_language}")
        
        # 判断是否正确
        should_be_chinese = chinese_chars > 0 and (chinese_ratio > 0.3 if total_chars > 0 else True)
        is_correct = (fixed_language == 'zh') == should_be_chinese
        
        print(f"   应该识别为中文: {should_be_chinese}")
        print(f"   修复后识别正确: {is_correct}")
        print()

if __name__ == "__main__":
    test_fixed_language_detection() 