#!/usr/bin/env python3
"""
详细调试模板格式化问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_template_detailed():
    """详细调试模板格式化问题"""
    
    print("🔍 详细调试模板格式化问题")
    print("=" * 60)
    
    # 1. 直接读取模板文件
    print("1. 直接读取模板文件:")
    template_path = "data/prompt_templates/multi_stage_chinese_template.txt"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        print(f"✅ 文件读取成功，长度: {len(raw_content)} 字符")
        
        # 检查文件内容
        print("文件内容前200字符:")
        print("-" * 40)
        print(repr(raw_content[:200]))
        print("-" * 40)
        
        # 检查是否有特殊字符
        print("检查特殊字符:")
        special_chars = []
        for i, char in enumerate(raw_content):
            if ord(char) > 127:
                special_chars.append((i, char, ord(char)))
        print(f"发现 {len(special_chars)} 个特殊字符")
        if special_chars:
            print("前10个特殊字符:")
            for i, char, code in special_chars[:10]:
                print(f"  位置{i}: '{char}' (U+{code:04X})")
        
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return
    
    # 2. 检查模板参数
    print("\n2. 检查模板参数:")
    import re
    param_pattern = r'\{(\w+)\}'
    params = re.findall(param_pattern, raw_content)
    print(f"模板中的参数: {params}")
    
    # 3. 测试手动格式化
    print("\n3. 测试手动格式化:")
    try:
        # 清理模板内容
        cleaned_template = raw_content.strip()
        
        # 测试格式化
        test_result = cleaned_template.format(
            summary="测试摘要",
            context="测试上下文", 
            query="测试查询"
        )
        print("✅ 手动格式化成功")
        print("结果预览:")
        print("-" * 40)
        print(test_result[:300] + "..." if len(test_result) > 300 else test_result)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ 手动格式化失败: {e}")
        print(f"错误类型: {type(e)}")
        
        # 尝试定位问题
        print("尝试定位问题:")
        try:
            # 逐个参数测试
            for param in params:
                test_dict = {param: f"测试{param}"}
                result = cleaned_template.format(**test_dict)
                print(f"✅ 参数 {param} 测试成功")
        except Exception as e2:
            print(f"❌ 参数测试失败: {e2}")
    
    # 4. 检查模板加载器的问题
    print("\n4. 检查模板加载器:")
    from xlm.components.prompt_templates.template_loader import template_loader
    
    loaded_template = template_loader.get_template("multi_stage_chinese_template")
    if loaded_template:
        print(f"✅ 模板加载器加载成功，长度: {len(loaded_template)} 字符")
        
        # 比较原始内容和加载的内容
        if loaded_template == raw_content.strip():
            print("✅ 加载内容与原始内容一致")
        else:
            print("❌ 加载内容与原始内容不一致")
            print(f"原始长度: {len(raw_content.strip())}")
            print(f"加载长度: {len(loaded_template)}")
            
            # 检查差异
            if len(loaded_template) < len(raw_content.strip()):
                print("加载的内容被截断了")
            else:
                print("加载的内容有额外内容")
    else:
        print("❌ 模板加载器加载失败")

if __name__ == "__main__":
    debug_template_detailed() 