#!/usr/bin/env python3
"""
直接测试公司名称修正功能的有效性
"""

import re

def test_company_name_fix():
    """测试公司名称修正功能"""
    print("=== 测试公司名称修正功能 ===")
    
    # 模拟LLM生成的包含翻译问题的回答
    test_responses = [
        "根据公司的财报预测及其详细解释，在2019年至20年间取得了显著进展后，德赛 battery (00) 的20+ 年度收益预估显示了积极势头。",
        "德赛 Battery (000049) 在2021年表现良好，主要得益于iPhone需求增长。",
        "德赛 battery (000049) 的业绩超出预期，主要源于A客户业务成长。",
        "中国平安 Ping An 的保险业务发展迅速。",
        "比亚迪 BYD 的电动汽车销量大幅增长。",
        "腾讯 Tencent 的营收表现强劲。",
    ]
    
    # 公司名称翻译映射
    company_translations = {
        # 德赛电池相关
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
        r'德赛\s*BATTERY': '德赛电池',
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
        
        # 其他常见公司
        r'中国平安\s*Ping\s*An': '中国平安',
        r'Ping\s*An\s*Insurance': '中国平安',
        r'比亚迪\s*BYD': '比亚迪',
        r'BYD\s*Company': '比亚迪',
        r'腾讯\s*Tencent': '腾讯',
        r'Tencent\s*Holdings': '腾讯',
        r'阿里巴巴\s*Alibaba': '阿里巴巴',
        r'Alibaba\s*Group': '阿里巴巴',
    }
    
    def fix_company_name_translation(text: str) -> str:
        """修正公司名称翻译问题"""
        # 应用修正
        for pattern, replacement in company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    print("测试修正功能...")
    for i, response in enumerate(test_responses, 1):
        print(f"\n测试 {i}:")
        print(f"原始回答: {response}")
        
        # 应用修正
        fixed_response = fix_company_name_translation(response)
        print(f"修正后回答: {fixed_response}")
        
        # 检查是否还有翻译问题
        issues = []
        if "battery" in fixed_response.lower() and "德赛" in fixed_response:
            issues.append("德赛电池仍被翻译为battery")
        if "ping an" in fixed_response.lower() and "中国平安" in fixed_response:
            issues.append("中国平安仍被翻译为Ping An")
        if "byd" in fixed_response.lower() and "比亚迪" in fixed_response:
            issues.append("比亚迪仍被翻译为BYD")
        
        if issues:
            print(f"❌ 仍有问题: {', '.join(issues)}")
        else:
            print(f"✅ 修正成功")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")

def test_specific_case():
    """测试您提供的具体案例"""
    print("\n=== 测试具体案例 ===")
    
    # 您提供的实际LLM回答
    actual_response = "根据公司的财报预测及其详细解释，在2019年至20年间取得了显著进展后，德赛 battery (00) 的20+ 年度收益预估显示了积极势头。关键因素之一是在苹果(A 客户群组 ) 上的新产品订单激增——特别是 iPhone +ProMax ——这不仅推动了收入流也提升了毛利率。同时 , 公司强调其非智能手机产品的扩展( 如手表/耳机 ), 这些领域正经历快速扩张期 ; 此外还提到通过并购活动完全纳入 NVT 子公司所带来的协同效应增强了内部效率 和规模经济效果; 最终这些都促成了更广泛的产品组合多样化和支持更高边际贡献率的基础架构建设 . 因此综上所述 : 新品推出成功带动市场需求上升 , 合理化生产流程提高运营效益加上多元化战略部署共同构成了未来一年内连续获利的关键驱动力源 . 答案忠实地反映了原始文档的内容而无多余推断"
    
    print(f"原始LLM回答:")
    print(f"'{actual_response}'")
    
    # 公司名称翻译映射
    company_translations = {
        r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
        r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
        r'德赛\s*battery': '德赛电池',
        r'德赛\s*Battery': '德赛电池',
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
    
    # 检查是否包含正确的中文名称
    if "德赛电池" in fixed_response:
        print(f"✅ 包含正确的中文公司名称")
    else:
        print(f"❌ 缺少正确的中文公司名称")

if __name__ == "__main__":
    test_company_name_fix()
    test_specific_case() 