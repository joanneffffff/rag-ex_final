#!/usr/bin/env python3
"""
专门测试硬编码 Prompt 的简化脚本
使用 cuda:1 设备，专注于测试实际系统输出的 Prompt 效果
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_hardcoded_prompt():
    """测试硬编码的完整 Prompt"""
    
    print("🚀 硬编码 Prompt 测试")
    print("使用 cuda:1 设备，测试实际系统输出的 Prompt 效果")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        
        # 初始化生成器
        print("1. 初始化 LLM 生成器...")
        print(f"   配置设备: cuda:1")
        generator = LocalLLMGenerator(device="cuda:1")
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        print(f"   实际使用设备: {generator.device}")
        
        # 硬编码的完整 Prompt（从实际系统输出）
        hardcoded_prompt = """===SYSTEM===
你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告摘要】以及其所依据的【完整公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。你的回答必须**忠实于原文的语义和信息**，并进行**必要的总结、提炼和基于原文的谨慎分析**。

**请用中文作答。**

**极度重要：请严格遵守以下输出规范。你的回答：**
* **形式与内容：** **仅包含回答核心问题的最终信息**。**严禁**任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、引导语、开场白、过渡句、总结性语句或后续说明。
* **格式限制：** **严禁**使用任何格式标记，包括但不限于：加粗、斜体、下划线、列表符号（如数字列表、点号列表等）、代码块、表格、\\boxed{}。
* **引用限制：** **严禁**引用或复述Prompt内容。
* **生成停止：** **回答完成后，请立即停止生成，不要输出任何额外字符。**
* **信息来源与智能分析：**
    * 你的回答中的所有信息**必须忠实于提供的【公司财务报告摘要】和【完整公司财务报告片段】**。
    * 你可以**优先参考摘要信息**进行高层次理解，但最终回答的**细节和准确性必须以完整片段为准**。
    * 在此基础上，你可以对原文内容进行**必要的归纳、总结、措辞重组，以及基于原文事实的** **合理、谨慎的分析和趋势解读**，以直接、清晰地回答用户问题。
    * **严格禁止**引入任何原文未提及的外部信息、主观臆测或未经上下文明确、间接支持的结论。
* **处理信息缺失：** 如果提供的【公司财务报告摘要】和【完整公司财务报告片段】中**确实没有足够的信息**来明确回答用户问题（即使经过归纳、总结和合理分析也无法得出），你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。

**回答长度：请力求简洁，将回答内容限制在 2-5 句话以内，不超过 150 个汉字。**

===USER===
---
**【公司财务报告摘要】**
德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。尽管安卓业务受到H客户销量下滑的影响，但新荣耀新品的发布预计将缓解这一影响。股票近期市场数据显示，价格波动较大，但总体趋势向上。
---

**【完整公司财务报告片段】**
研报内容如下: 研报题目是:德赛电池（000049）：业绩预告超出预期，新品订单及盈利能力佳；目标价格是89.0，评分是7.0；研报摘要:事件 德赛电池发布20年业绩预告，20年营收约193 9亿元，同比增长5  净利润7 0 7 6亿元，同比增长4 5  13 5  归母净利润6 3 6 9亿元，同比增长25 5  37 4

2、21年利润持续增长，源于A客户及非手机业务成长及并表比例增加

公司20年营收约193 9亿元，同比增长5  归母净利润6 3 6 9亿元，中值6 6亿元对应同比增长31 ，超出预期

对应Q4营收66亿元，同比增长12 ，净利润中值2 9亿元，同比增长34 ，归母净利润中值2 9亿元，同比增长79

整体而言，21年我们认为手机业务营收望保持稳定，结构上A客户占比提升 非手机业务保持较佳增长趋势 利润端仍有望受益营收增长、利润率提升及子公司净利润全年并表比例增至100 而增长

这是以德赛电池（000049）：业绩预告超出预期，新品订单及盈利能力佳为题目,在2021-01-22 00:00:00日期发布的研究报告

我们认为超预期主要源于iPhone 12 Pro Max新机需求佳及新品盈利能力提升，使得综合盈利能力同比、环比提升

展望21年 1 5G iPhone周期叠加非手机业务增量 Watch  amp  AirPods需求量增长，iPad、Mac份额提升 望驱动A客户业务成长，此外新品盈利能力提升王驱动盈利水平同比提升

3 非A笔电、电动工具等亦望维持20年较高速增长的趋势

我们在此前报告 德赛电池 拟整合ATL旗下NVT ，详解消费电池格局变化趋势 中分析，此次资产重组意向的背景在于行业格局的潜在变化 1 从电芯格局看，随着动力电芯需求快速增长，以及消费电芯现有的分散格局之下，众多电芯厂商在消费电子市场不具备成本优势，并在逐步减少消费电芯领域的投入，对剩余的积极参与者而言，具有份额整合空间
---

**【用户问题】**
德赛电池（000049）2021年利润持续增长的主要原因是什么？
---

**【回答】**
==="""
        
        print(f"✅ 硬编码 Prompt 准备完成")
        print(f"   Prompt 长度: {len(hardcoded_prompt)} 字符")
        print(f"   Prompt 预览: {hardcoded_prompt[:200]}...")
        
        # 检查格式转换
        print(f"\n2. 检查格式转换...")
        if "Fin-R1" in generator.model_name:
            print("🔍 检测到 Fin-R1 模型，检查格式转换...")
            
            json_chat = generator.convert_to_json_chat_format(hardcoded_prompt)
            will_convert = json_chat != hardcoded_prompt
            
            print(f"会进行格式转换: {'是' if will_convert else '否'}")
            
            if will_convert:
                print(f"转换后长度: {len(json_chat)} 字符")
                fin_r1_format = generator.convert_json_to_fin_r1_format(json_chat)
                print(f"Fin-R1 格式长度: {len(fin_r1_format)} 字符")
                print(f"Fin-R1 格式预览: {fin_r1_format[:200]}...")
            else:
                print("⚠️ 硬编码 Prompt 不会进行格式转换")
        
        # 生成响应
        print(f"\n3. 生成响应...")
        print("🚀 开始生成，请稍候...")
        
        responses = generator.generate([hardcoded_prompt])
        response = responses[0] if responses else "生成失败"
        
        print(f"\n4. 响应结果:")
        print("=" * 50)
        print(f"问题: 德赛电池（000049）2021年利润持续增长的主要原因是什么？")
        print(f"答案: {response}")
        print("=" * 50)
        
        # 评估响应质量
        print(f"\n5. 响应质量评估:")
        length = len(response.strip())
        print(f"   响应长度: {length} 字符")
        
        # 检查关键信息
        key_terms = ["德赛电池", "iPhone", "需求", "增长", "利润", "业绩", "A客户", "新品", "盈利能力"]
        found_terms = [term for term in key_terms if term in response]
        print(f"   关键信息: {found_terms}")
        print(f"   准确性: {'✅' if len(found_terms) >= 3 else '❌'} (找到{len(found_terms)}个关键词)")
        
        # 检查语言一致性
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        has_english = any(char.isalpha() and ord(char) < 128 for char in response)
        print(f"   包含中文字符: {'是' if has_chinese else '否'}")
        print(f"   包含英文字符: {'是' if has_english else '否'}")
        
        # 检查格式标记
        unwanted_patterns = ["【", "】", "回答：", "Answer:", "---", "===", "___"]
        has_unwanted = any(pattern in response for pattern in unwanted_patterns)
        print(f"   纯粹性: {'✅' if not has_unwanted else '❌'} (无格式标记)")
        
        # 检查句子完整性
        is_complete = response.strip().endswith(("。", "！", "？", ".", "!", "?"))
        print(f"   完整性: {'✅' if is_complete else '❌'} (句子完整)")
        
        # 检查长度限制
        length_ok = 20 <= length <= 150
        print(f"   长度合适: {'✅' if length_ok else '❌'} (20-150字符)")
        
        # 总体评分
        score = 0
        if length_ok: score += 20
        if has_chinese and not has_english: score += 20
        if len(found_terms) >= 3: score += 20
        if not has_unwanted: score += 20
        if is_complete: score += 20
        
        print(f"\n🎯 硬编码 Prompt 响应评分: {score}/100 ({score}%)")
        
        if score >= 80:
            print("🎉 硬编码 Prompt 效果很好！")
        elif score >= 60:
            print("⚠️ 硬编码 Prompt 效果一般，需要优化")
        else:
            print("❌ 硬编码 Prompt 效果不佳，需要重新设计")
        
        return True
        
    except Exception as e:
        print(f"❌ 硬编码 Prompt 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hardcoded_prompt() 