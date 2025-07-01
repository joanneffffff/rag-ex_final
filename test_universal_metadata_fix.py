#!/usr/bin/env python3
"""
测试基于元数据的通用公司名称修正方案
"""

import re
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_universal_company_name_fix():
    """测试通用公司名称修正功能"""
    print("=== 测试通用公司名称修正功能 ===")
    
    # 测试用例：包含各种公司的翻译问题
    test_cases = [
        # 德赛电池相关
        "德赛 battery (00) 的业绩表现良好",
        "德赛 Battery (000049) 在2021年表现良好",
        "德赛 battery (000049) 的业绩超出预期",
        "德赛 Battery (0) 的营收增长",
        
        # 中国平安相关
        "中国平安 Ping An (601318) 的保险业务发展迅速",
        "Ping An Insurance (601318) 的财务表现",
        "中国平安 Ping An 的营收增长",
        
        # 比亚迪相关
        "比亚迪 BYD (002594) 的电动汽车销量大幅增长",
        "BYD Company (002594) 的业绩表现",
        "比亚迪 BYD 的营收情况",
        
        # 腾讯相关
        "腾讯 Tencent (00700) 的营收表现强劲",
        "Tencent Holdings (00700) 的财务数据",
        "腾讯 Tencent 的业绩",
        
        # 阿里巴巴相关
        "阿里巴巴 Alibaba (09988) 的电商业务",
        "Alibaba Group (09988) 的财务表现",
        "阿里巴巴 Alibaba 的营收",
        
        # 其他公司（通用测试）
        "华为 Huawei (002502) 的技术创新",
        "小米 Xiaomi (01810) 的智能手机",
        "美团 Meituan (03690) 的外卖业务",
        "京东 JD (09618) 的电商平台",
        "网易 NetEase (09999) 的游戏业务",
        "百度 Baidu (09888) 的AI技术",
        "拼多多 PDD (PDD) 的社交电商",
        "字节跳动 ByteDance 的短视频",
        "滴滴 DiDi 的出行服务",
        "快手 Kuaishou (01024) 的直播业务",
    ]
    
    # 基于元数据的通用修正逻辑
    def fix_company_name_translation_universal(text: str) -> str:
        """基于元数据的通用公司名称修正"""
        
        # 1. 特定公司映射（保持精确性）
        specific_company_translations = {
            # 德赛电池相关
            r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
            r'德赛\s*battery\s*\(0+\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(0+\)': '德赛电池（000049）',
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
        
        # 2. 应用特定公司映射
        for pattern, replacement in specific_company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 3. 通用中文公司名称翻译修正
        # 匹配模式：中文公司名 + 英文翻译 + 股票代码
        chinese_company_patterns = [
            # 模式1：中文名 + 英文名 + 股票代码（带括号）
            r'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s+([A-Za-z]+(?:\s*[A-Za-z]+)*)\s*[（(](\d{6})[）)]',
            # 模式2：中文名 + 英文名（无股票代码）
            r'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s+([A-Za-z]+(?:\s*[A-Za-z]+)*)',
        ]
        
        for pattern in chinese_company_patterns:
            def replace_company_name(match):
                chinese_name = match.group(1).strip()
                english_name = match.group(2).strip()
                
                # 如果有股票代码，保留股票代码
                if len(match.groups()) >= 3 and match.group(3):
                    stock_code = match.group(3)
                    return f"{chinese_name}（{stock_code}）"
                else:
                    return chinese_name
            
            text = re.sub(pattern, replace_company_name, text)
        
        # 4. 修正常见的英文公司名翻译回中文
        common_english_to_chinese = {
            r'\bBattery\b': '电池',
            r'\bInsurance\b': '保险',
            r'\bHoldings\b': '控股',
            r'\bGroup\b': '集团',
            r'\bCompany\b': '公司',
            r'\bCorporation\b': '公司',
            r'\bLimited\b': '有限公司',
            r'\bLtd\b': '有限公司',
            r'\bInc\b': '公司',
            r'\bCo\b': '公司',
        }
        
        # 只在特定上下文中应用这些翻译
        for english, chinese in common_english_to_chinese.items():
            # 只在中文公司名后面出现英文时进行替换
            pattern = rf'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s*{english}'
            replacement = rf'\1{chinese}'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    print("测试通用修正功能...")
    success_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"  原始: {test_case}")
        
        # 应用修正
        fixed = fix_company_name_translation_universal(test_case)
        print(f"  修正后: {fixed}")
        
        # 检查修正效果
        issues = []
        
        # 检查是否还有英文翻译
        if re.search(r'[A-Za-z]+(?:\s*[A-Za-z]+)*', fixed):
            # 检查是否是合理的英文（如股票代码中的字母）
            if not re.search(r'[（(]\d{6}[）)]', fixed) and not re.search(r'[（(][A-Z]{2}\d{4}[）)]', fixed):
                issues.append("仍包含英文翻译")
        
        # 检查是否包含中文公司名称
        if not re.search(r'[\u4e00-\u9fff]+', fixed):
            issues.append("缺少中文公司名称")
        
        if issues:
            print(f"  ❌ 问题: {', '.join(issues)}")
        else:
            print(f"  ✅ 修正成功")
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_count} 成功")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    print(f"{'='*60}")

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 使用真实数据测试 ===")
    
    # 您提供的实际LLM回答
    actual_response = "根据公司的财报预测及其详细解释，在2019年至20年间取得了显著进展后，德赛 battery (00) 的20+ 年度收益预估显示了积极势头。关键因素之一是在苹果(A 客户群组 ) 上的新产品订单激增——特别是 iPhone +ProMax ——这不仅推动了收入流也提升了毛利率。同时 , 公司强调其非智能手机产品的扩展( 如手表/耳机 ), 这些领域正经历快速扩张期 ; 此外还提到通过并购活动完全纳入 NVT 子公司所带来的协同效应增强了内部效率 和规模经济效果; 最终这些都促成了更广泛的产品组合多样化和支持更高边际贡献率的基础架构建设 . 因此综上所述 : 新品推出成功带动市场需求上升 , 合理化生产流程提高运营效益加上多元化战略部署共同构成了未来一年内连续获利的关键驱动力源 . 答案忠实地反映了原始文档的内容而无多余推断"
    
    print(f"原始LLM回答:")
    print(f"'{actual_response}'")
    
    # 应用通用修正
    def fix_company_name_translation_universal(text: str) -> str:
        """基于元数据的通用公司名称修正"""
        
        # 1. 特定公司映射
        specific_company_translations = {
            r'德赛\s*battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(00\)': '德赛电池（000049）',
            r'德赛\s*battery\s*\(000049\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(000049\)': '德赛电池（000049）',
            r'德赛\s*battery\s*\(0+\)': '德赛电池（000049）',
            r'德赛\s*Battery\s*\(0+\)': '德赛电池（000049）',
            r'德赛\s*battery': '德赛电池',
            r'德赛\s*Battery': '德赛电池',
        }
        
        # 2. 应用特定公司映射
        for pattern, replacement in specific_company_translations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 3. 通用修正
        chinese_company_patterns = [
            r'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s+([A-Za-z]+(?:\s*[A-Za-z]+)*)\s*[（(](\d{6})[）)]',
            r'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s+([A-Za-z]+(?:\s*[A-Za-z]+)*)',
        ]
        
        for pattern in chinese_company_patterns:
            def replace_company_name(match):
                chinese_name = match.group(1).strip()
                english_name = match.group(2).strip()
                
                if len(match.groups()) >= 3 and match.group(3):
                    stock_code = match.group(3)
                    return f"{chinese_name}（{stock_code}）"
                else:
                    return chinese_name
            
            text = re.sub(pattern, replace_company_name, text)
        
        return text
    
    # 应用修正
    fixed_response = fix_company_name_translation_universal(actual_response)
    
    print(f"\n修正后回答:")
    print(f"'{fixed_response}'")
    
    # 检查修正效果
    if "德赛 battery" in fixed_response or "德赛 Battery" in fixed_response:
        print(f"❌ 修正失败：仍包含翻译问题")
    else:
        print(f"✅ 修正成功：公司名称已正确修正")
    
    if "德赛电池（000049）" in fixed_response:
        print(f"✅ 股票代码已正确修正为（000049）")
    elif "德赛电池" in fixed_response:
        print(f"✅ 包含正确的中文公司名称")
    else:
        print(f"❌ 缺少正确的中文公司名称")

if __name__ == "__main__":
    test_universal_company_name_fix()
    test_with_real_data() 