#!/usr/bin/env python3
"""
测试 Prompt 优化效果
固定测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_prompt_optimization():
    """测试 Prompt 优化效果"""
    
    print("=== Generator LLM Prompt 优化测试 ===")
    print("测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        from xlm.components.prompt_templates.template_loader import template_loader
        
        print("1. 初始化 LLM 生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        print(f"✅ 设备: {generator.device}")
        
        # 构造测试数据
        print("\n2. 构造测试数据...")
        test_context = """
        德赛电池（000049）2021年业绩预告显示，公司预计实现归属于上市公司股东的净利润为6.5亿元至7.5亿元，
        同比增长11.02%至28.23%。业绩增长的主要原因是：
        1. iPhone 12 Pro Max等高端产品需求强劲，带动公司电池业务增长
        2. 新产品盈利能力提升，毛利率改善
        3. A客户业务持续成长，非手机业务稳步增长
        4. 并表比例增加，贡献业绩增量
        """
        
        test_summary = "德赛电池2021年业绩增长主要受益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。"
        
        test_query = "德赛电池（000049）2021年利润持续增长的主要原因是什么？"
        
        print(f"✅ 上下文长度: {len(test_context)} 字符")
        print(f"✅ 摘要长度: {len(test_summary)} 字符")
        print(f"✅ 问题长度: {len(test_query)} 字符")
        
        # 使用优化后的模板生成 Prompt
        print("\n3. 生成优化后的 Prompt...")
        prompt = template_loader.format_template(
            "multi_stage_chinese_template",
            context=test_context,
            query=test_query,
            summary=test_summary
        )
        
        if prompt is None:
            print("❌ Prompt 模板加载失败")
            return False
            
        print(f"✅ Prompt 生成成功，长度: {len(prompt)} 字符")
        print(f"✅ Prompt 预览:\n{prompt[:300]}...")
        
        # 测试格式转换
        print("\n4. 测试格式转换...")
        if "Fin-R1" in generator.model_name:
            print("🔍 检测到 Fin-R1 模型，测试格式转换...")
            
            # 测试 JSON 格式转换
            json_chat = generator.convert_to_json_chat_format(prompt)
            print(f"✅ JSON 格式转换完成，长度: {len(json_chat)} 字符")
            
            # 测试 Fin-R1 格式转换
            fin_r1_format = generator.convert_json_to_fin_r1_format(json_chat)
            print(f"✅ Fin-R1 格式转换完成，长度: {len(fin_r1_format)} 字符")
            
            # 显示转换后的格式预览
            print("\n📋 转换后格式预览:")
            print("-" * 50)
            print(fin_r1_format[:500] + "..." if len(fin_r1_format) > 500 else fin_r1_format)
            print("-" * 50)
        
        # 生成答案
        print("\n5. 生成答案...")
        print("🚀 开始生成，请稍候...")
        
        responses = generator.generate([prompt])
        answer = responses[0] if responses else "生成失败"
        
        print("\n" + "=" * 60)
        print("📝 生成结果")
        print("=" * 60)
        print(f"问题: {test_query}")
        print(f"答案: {answer}")
        print("=" * 60)
        
        # 分析结果
        print("\n6. 结果分析...")
        
        # 检查答案质量
        quality_indicators = {
            "简洁性": len(answer) <= 200,  # 控制在200字符内
            "准确性": "德赛电池" in answer or "iPhone" in answer or "需求" in answer,
            "纯粹性": not any(marker in answer for marker in ["【", "】", "回答：", "Answer:"]),
            "完整性": answer.strip().endswith(("。", "！", "？", ".", "!", "?"))
        }
        
        print("📊 质量指标:")
        for indicator, passed in quality_indicators.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {indicator}: {'通过' if passed else '需改进'}")
        
        # 计算总体评分
        score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        print(f"\n🎯 总体评分: {score:.1f}%")
        
        if score >= 75:
            print("🎉 Prompt 优化效果良好！")
        elif score >= 50:
            print("⚠️ Prompt 优化效果一般，需要进一步调整")
        else:
            print("❌ Prompt 优化效果不佳，需要重新设计")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_prompt_optimization() 