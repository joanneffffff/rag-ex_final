#!/usr/bin/env python3
"""
简单的修复效果测试
"""

import re

def test_company_name_fix():
    """测试公司名称修正功能"""
    print("=== 测试公司名称修正功能 ===")
    
    # 您提供的实际LLM回答
    actual_response = "根据公司的财报预测及其详细解释，在2019年至20年间取得了显著进展后，德赛 battery (00) 的20+ 年度收益预估显示了积极势头。关键因素之一是在苹果(A 客户群组 ) 上的新产品订单激增——特别是 iPhone +ProMax ——这不仅推动了收入流也提升了毛利率。同时 , 公司强调其非智能手机产品的扩展( 如手表/耳机 ), 这些领域正经历快速扩张期 ; 此外还提到通过并购活动完全纳入 NVT 子公司所带来的协同效应增强了内部效率 和规模经济效果; 最终这些都促成了更广泛的产品组合多样化和支持更高边际贡献率的基础架构建设 . 因此综上所述 : 新品推出成功带动市场需求上升 , 合理化生产流程提高运营效益加上多元化战略部署共同构成了未来一年内连续获利的关键驱动力源 . 答案忠实地反映了原始文档的内容而无多余推断"
    
    print(f"原始LLM回答:")
    print(f"'{actual_response}'")
    
    # 公司名称翻译映射 - 修正后的版本
    company_translations = {
        # 德赛电池相关 - 修正股票代码
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(0+\)': '德赛电池（000049）',  # 匹配任何以0开头的股票代码
        r'德赛\s*Battery\s*\(0+\)': '德赛电池（000049）',  # 匹配任何以0开头的股票代码
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
        
        # 中国平安相关
        r'中国平安\s*Ping\s*An\s*\(601318\)': '中国平安（601318）',
        r'Ping\s*An\s*Insurance\s*\(601318\)': '中国平安（601318）',
        r'中国平安\s*Ping\s*An': '中国平安',
        r'Ping\s*An\s*Insurance': '中国平安',
        
        # 比亚迪相关
        r'比亚迪\s*BYD\s*\(002594\)': '比亚迪（002594）',
        r'BYD\s*Company\s*\(002594\)': '比亚迪（002594）',
        r'比亚迪\s*BYD': '比亚迪',
        r'BYD\s*Company': '比亚迪',
    }
    
    def fix_company_name_translation(text: str) -> str:
        """修正公司名称翻译问题"""
        for pattern, replacement in company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    # 应用修正
    fixed_response = fix_company_name_translation(actual_response)
    
    print(f"\n修正后回答:")
    print(f"'{fixed_response}'")
    
    # 检查修正效果
    if "德赛 battery" in fixed_response or "德赛 Battery" in fixed_response:
        print(f"❌ 修正失败：仍包含翻译问题")
    else:
        print(f"✅ 修正成功：公司名称已正确修正")
    
    # 检查股票代码修正
    if "德赛电池（000049）" in fixed_response:
        print(f"✅ 股票代码已正确修正为（000049）")
    elif "德赛电池 (00)" in fixed_response:
        print(f"❌ 股票代码修正失败：仍为 (00)")
    elif "德赛电池" in fixed_response:
        print(f"⚠️  公司名称已修正，但股票代码可能有问题")
    else:
        print(f"❌ 缺少正确的中文公司名称")

def test_various_cases():
    """测试各种情况"""
    print("\n=== 测试各种情况 ===")
    
    test_cases = [
        "德赛 battery (00) 的业绩表现良好",
        "德赛 Battery (000049) 在2021年表现良好",
        "德赛 battery (000049) 的业绩超出预期",
        "德赛 Battery (0) 的营收增长",
        "德赛 battery (000) 的利润提升",
        "中国平安 Ping An (601318) 的保险业务发展迅速",
        "比亚迪 BYD (002594) 的电动汽车销量大幅增长",
    ]
    
    # 公司名称翻译映射
    company_translations = {
        # 德赛电池相关 - 修正股票代码
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(0+\)': '德赛电池（000049）',  # 匹配任何以0开头的股票代码
        r'德赛\s*Battery\s*\(0+\)': '德赛电池（000049）',  # 匹配任何以0开头的股票代码
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
        
        # 中国平安相关
        r'中国平安\s*Ping\s*An\s*\(601318\)': '中国平安（601318）',
        r'Ping\s*An\s*Insurance\s*\(601318\)': '中国平安（601318）',
        r'中国平安\s*Ping\s*An': '中国平安',
        r'Ping\s*An\s*Insurance': '中国平安',
        
        # 比亚迪相关
        r'比亚迪\s*BYD\s*\(002594\)': '比亚迪（002594）',
        r'BYD\s*Company\s*\(002594\)': '比亚迪（002594）',
        r'比亚迪\s*BYD': '比亚迪',
        r'BYD\s*Company': '比亚迪',
    }
    
    def fix_company_name_translation(text: str) -> str:
        """修正公司名称翻译问题"""
        for pattern, replacement in company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"  原始: {test_case}")
        
        fixed = fix_company_name_translation(test_case)
        print(f"  修正后: {fixed}")
        
        # 检查修正效果
        if "battery" in fixed.lower() and "德赛" in fixed:
            print(f"  ❌ 仍有翻译问题")
        elif "德赛电池（000049）" in fixed:
            print(f"  ✅ 完美修正")
        elif "德赛电池" in fixed:
            print(f"  ✅ 公司名称已修正")
        else:
            print(f"  ❌ 修正失败")

def test_enhanced_prompt():
    """测试增强的prompt模板"""
    print("\n=== 测试增强的Prompt模板 ===")
    
    # 读取增强的prompt模板
    try:
        with open("data/prompt_templates/multi_stage_chinese_template.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        print("✅ 成功读取增强的prompt模板")
        
        # 检查是否包含公司名称约束
        if "公司名称约束示例" in prompt_template:
            print("✅ 包含公司名称约束示例")
        else:
            print("❌ 缺少公司名称约束示例")
        
        if "绝对禁止的行为" in prompt_template:
            print("✅ 包含绝对禁止的行为说明")
        else:
            print("❌ 缺少绝对禁止的行为说明")
        
        if "德赛 battery" in prompt_template:
            print("✅ 包含具体的错误示例")
        else:
            print("❌ 缺少具体的错误示例")
        
        if "德赛电池（000049）" in prompt_template:
            print("✅ 包含正确的示例")
        else:
            print("❌ 缺少正确的示例")
        
        print(f"\nPrompt模板长度: {len(prompt_template)} 字符")
        
    except Exception as e:
        print(f"❌ 读取prompt模板失败: {e}")

if __name__ == "__main__":
    test_company_name_fix()
    test_various_cases()
    test_enhanced_prompt() 