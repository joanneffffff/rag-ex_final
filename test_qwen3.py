#!/usr/bin/env python3
"""
Qwen3-8B 专用测试脚本
使用本地模型缓存，禁止联网下载
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_qwen3():
    print("🚀 Qwen3-8B 测试开始...")
    
    model_path = "/users/sgjfei3/data/huggingface/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
    print(f"📋 使用本地模型: {model_path}")
    
    try:
        print("🔧 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        
        print("🔧 加载模型 (使用4bit量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,  # 使用4bit量化
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            local_files_only=True
        )
        
        # 完整的金融分析prompt
        test_prompt = """===SYSTEM=== 你是一位专业的金融分析师。你的核心任务是根据以下提供的【公司财务报告摘要】以及其所依据的【完整公司财务报告片段】，针对用户的问题提供一个**纯粹、直接且准确的回答**。你的回答必须**忠实于原文的语义和信息**，并进行**必要的总结、提炼和基于原文的谨慎分析**。 
 **请用中文作答。** 
 **极度重要：请严格遵守以下输出规范。你的回答：** * **形式与内容：** **仅包含回答核心问题的最终信息**。**严禁**任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、引导语、开场白、过渡句、总结性语句或后续说明。 * **格式限制：** **严禁**使用任何格式标记，包括但不限于：加粗、斜体、下划线、列表符号（如数字列表、点号列表等）、代码块、表格、\\boxed{}。 * **引用限制：** **严禁**引用或复述Prompt内容。 * **生成停止：** **回答完成后，请立即停止生成，不要输出任何额外字符。** * **信息来源与智能分析：**     * 你的回答中的所有信息**必须忠实于提供的【公司财务报告摘要】和【完整公司财务报告片段】**。     * 你可以**优先参考摘要信息**进行高层次理解，但最终回答的**细节和准确性必须以完整片段为准**。     * 在此基础上，你可以对原文内容进行**必要的归纳、总结、措辞重组，以及基于原文事实的** **合理、谨慎的分析和趋势解读**，以直接、清晰地回答用户问题。     * **严格禁止**引入任何原文未提及的外部信息、主观臆测或未经上下文明确、间接支持的结论。 * **处理信息缺失：** 如果提供的【公司财务报告摘要】和【完整公司财务报告片段】中**确实没有足够的信息**来明确回答用户问题（即使经过归纳、总结和合理分析也无法得出），你的回答应是"根据现有信息，无法提供此项信息。"。请直接给出这句话，不带任何其他修饰。 
 **回答长度：请力求简洁，将回答内容限制在 3-5 句话以内，不超过 300 个汉字。** 
 ===USER=== --- **【公司财务报告摘要】** 德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。尽管安卓业务受到H客户销量下滑的影响，但新荣耀新品的发布预计将缓解这一影响。股票近期市场数据显示，价格波动较大，但总体趋势向上。 --- 
 **【完整公司财务报告片段】** 研报内容如下: 研报题目是:德赛电池（000049）：业绩预告超出预期，新品订单及盈利能力佳；目标价格是89.0，评分是7.0；研报摘要:事件 德赛电池发布20年业绩预告，20年营收约193 9亿元，同比增长5  净利润7 0 7 6亿元，同比增长4 5  13 5  归母净利润6 3 6 9亿元，同比增长25 5  37 4 
 2、21年利润持续增长，源于A客户及非手机业务成长及并表比例增加 
 公司20年营收约193 9亿元，同比增长5  归母净利润6 3 6 9亿元，中值6 6亿元对应同比增长31 ，超出预期 
 对应Q4营收66亿元，同比增长12 ，净利润中值2 9亿元，同比增长34 ，归母净利润中值2 9亿元，同比增长79 
 整体而言，21年我们认为手机业务营收望保持稳定，结构上A客户占比提升 非手机业务保持较佳增长趋势 利润端仍有望受益营收增长、利润率提升及子公司净利润全年并表比例增至100 而增长 
 这是以德赛电池（000049）：业绩预告超出预期，新品订单及盈利能力佳为题目,在2021-01-22 00:00:00日期发布的研究报告 
 我们认为超预期主要源于iPhone 12 Pro Max新机需求佳及新品盈利能力提升，使得综合盈利能力同比、环比提升 
 展望21年 1 5G iPhone周期叠加非手机业务增量 Watch  amp  AirPods需求量增长，iPad、Mac份额提升 望驱动A客户业务成长，此外新品盈利能力提升王驱动盈利水平同比提升 
 3 非A笔电、电动工具等亦望维持20年较高速增长的趋势 
 我们在此前报告 德赛电池 拟整合ATL旗下NVT ，详解消费电池格局变化趋势 中分析，此次资产重组意向的背景在于行业格局的潜在变化 1 从电芯格局看，随着动力电芯需求快速增长，以及消费电芯现有的分散格局之下，众多电芯厂商在消费电子市场不具备成本优势，并在逐步减少消费电芯领域的投入，对剩余的积极参与者而言，具有份额整合空间 --- 
 **【用户问题】** 德赛电池（000049）2021年利润持续增长的主要原因是什么？ --- 
 **【回答】** ================="""
        
        print(f"\n📝 测试问题: 德赛电池（000049）2021年利润持续增长的主要原因是什么？")
        
        # 编码
        print("🔤 编码输入...")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        print("🤖 开始生成...")
        start_time = time.time()
        
        # 优化的生成参数
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # 增加到300以确保完整回答
                do_sample=False,    # 贪婪解码
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 增加重复惩罚
                no_repeat_ngram_size=3,  # 避免重复3-gram
                early_stopping=True,     # 启用早停
                num_beams=1,            # 使用单beam搜索
                length_penalty=1.0      # 长度惩罚
            )
        
        end_time = time.time()
        print(f"⏱️ 生成时间: {end_time - start_time:.2f}秒")
        
        # 解码
        print("🔤 解码输出...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        input_length = inputs.input_ids.shape[1]
        generated_text = response[input_length:]
        
        # 后处理：清理重复内容和截断句子
        print("🧹 后处理回答...")
        sentences = generated_text.strip().split('。')
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 过滤太短的句子
                # 检查是否与已有内容重复
                is_duplicate = False
                for seen in seen_content:
                    if sentence in seen or seen in sentence:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_sentences.append(sentence)
                    seen_content.add(sentence)
        
        # 重新组合，确保句子完整
        final_answer = '。'.join(unique_sentences) + '。'
        
        print(f"\n✅ Qwen3-8B 回答:")
        print(f"{'='*50}")
        print(final_answer)
        print(f"{'='*50}")
        print(f"📏 回答长度: {len(final_answer)} 字符")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen3() 