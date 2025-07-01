#!/usr/bin/env python3
"""
交互式 Prompt 测试脚本
让用户选择特定的 Prompt 变体和参数进行测试
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def show_menu(title, options):
    """显示菜单"""
    print(f"\n{title}")
    print("-" * 40)
    for i, (key, value) in enumerate(options.items(), 1):
        if isinstance(value, dict) and 'description' in value:
            print(f"{i}. {key}: {value['description']}")
        else:
            print(f"{i}. {key}")
    print("0. 退出")
    print("-" * 40)

def get_user_choice(options):
    """获取用户选择"""
    while True:
        try:
            choice = input("请选择 (输入数字): ").strip()
            if choice == "0":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return list(options.keys())[choice_num - 1]
            else:
                print(f"请输入 1-{len(options)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n退出程序")
            return None

def test_specific_combination():
    """测试特定的 Prompt 和参数组合"""
    
    print("=== 交互式 Prompt 测试 ===")
    print("测试问题：德赛电池（000049）2021年利润持续增长的主要原因是什么？")
    print("=" * 60)
    
    try:
        from xlm.components.generator.local_llm_generator import LocalLLMGenerator
        from prompt_variations_library import get_prompt_variations, get_parameter_variations, get_test_scenarios
        
        # 初始化生成器
        print("1. 初始化 LLM 生成器...")
        generator = LocalLLMGenerator()
        print(f"✅ 生成器初始化成功: {generator.model_name}")
        
        # 获取测试数据
        scenarios = get_test_scenarios()
        prompt_variations = get_prompt_variations("", "", "")
        parameter_variations = get_parameter_variations()
        
        # 选择测试场景
        show_menu("选择测试场景", scenarios)
        scenario_choice = get_user_choice(scenarios)
        if scenario_choice is None:
            return
        
        scenario = scenarios[scenario_choice]
        print(f"\n✅ 选择的场景: {scenario_choice}")
        print(f"问题: {scenario['query']}")
        
        # 选择 Prompt 变体
        show_menu("选择 Prompt 变体", prompt_variations)
        prompt_choice = get_user_choice(prompt_variations)
        if prompt_choice is None:
            return
        
        print(f"\n✅ 选择的 Prompt: {prompt_choice}")
        
        # 选择参数组合
        show_menu("选择参数组合", parameter_variations)
        param_choice = get_user_choice(parameter_variations)
        if param_choice is None:
            return
        
        print(f"\n✅ 选择的参数: {param_choice}")
        
        # 生成 Prompt
        prompt = prompt_variations[prompt_choice].format(
            context=scenario['context'],
            summary=scenario['summary'],
            query=scenario['query']
        )
        
        # 获取参数
        params = parameter_variations[param_choice]
        
        print(f"\n2. 测试配置:")
        print(f"   Prompt: {prompt_choice}")
        print(f"   参数: {param_choice}")
        print(f"   Temperature: {params['temperature']}")
        print(f"   Top-p: {params['top_p']}")
        print(f"   Max tokens: {params['max_new_tokens']}")
        print(f"   Prompt 长度: {len(prompt)} 字符")
        
        # 显示 Prompt 预览
        print(f"\n3. Prompt 预览:")
        print("-" * 50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 50)
        
        # 确认是否继续
        confirm = input("\n是否开始测试？(y/n): ").strip().lower()
        if confirm != 'y':
            print("测试已取消")
            return
        
        # 临时修改参数
        original_temp = generator.temperature
        original_top_p = generator.top_p
        original_max_tokens = generator.max_new_tokens
        
        try:
            generator.temperature = params["temperature"]
            generator.top_p = params["top_p"]
            generator.max_new_tokens = params["max_new_tokens"]
            
            # 生成响应
            print(f"\n4. 生成响应...")
            print("🚀 开始生成，请稍候...")
            
            import time
            start_time = time.time()
            responses = generator.generate([prompt])
            end_time = time.time()
            
            response = responses[0] if responses else "生成失败"
            generation_time = end_time - start_time
            
            print(f"\n5. 生成结果:")
            print("=" * 60)
            print(f"问题: {scenario['query']}")
            print(f"答案: {response}")
            print("=" * 60)
            print(f"生成时间: {generation_time:.2f}秒")
            
            # 评估结果
            print(f"\n6. 质量评估:")
            length = len(response.strip())
            print(f"   响应长度: {length} 字符")
            
            # 根据场景评估准确性
            if "德赛电池" in scenario_choice:
                key_terms = ["德赛电池", "iPhone", "需求", "增长", "利润", "业绩"]
            elif "用友网络" in scenario_choice:
                key_terms = ["用友网络", "现金流", "0.85", "增长", "12.5"]
            elif "首钢股份" in scenario_choice:
                key_terms = ["首钢股份", "业绩", "下降", "疫情", "降本增效"]
            else:
                key_terms = []
            
            found_terms = [term for term in key_terms if term in response]
            print(f"   关键信息: {found_terms}")
            print(f"   准确性: {'✅' if len(found_terms) >= 2 else '❌'} (找到{len(found_terms)}个关键词)")
            
            unwanted_patterns = ["【", "】", "回答：", "Answer:", "---", "===", "___"]
            has_unwanted = any(pattern in response for pattern in unwanted_patterns)
            print(f"   纯粹性: {'✅' if not has_unwanted else '❌'} (无格式标记)")
            
            is_complete = response.strip().endswith(("。", "！", "？", ".", "!", "?"))
            print(f"   完整性: {'✅' if is_complete else '❌'} (句子完整)")
            
            # 总体评分
            score = 0
            if 30 <= length <= 300: score += 25
            if len(found_terms) >= 2: score += 25
            if not has_unwanted: score += 25
            if is_complete: score += 25
            
            print(f"\n🎯 总体评分: {score}/100 ({score}%)")
            
            if score >= 75:
                print("🎉 效果很好！")
            elif score >= 50:
                print("⚠️ 效果一般，可以继续优化")
            else:
                print("❌ 效果不佳，需要重新设计")
            
            # 保存结果
            save_result = input("\n是否保存测试结果？(y/n): ").strip().lower()
            if save_result == 'y':
                import json
                result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario_choice,
                    "prompt": prompt_choice,
                    "parameters": param_choice,
                    "query": scenario['query'],
                    "response": response,
                    "generation_time": generation_time,
                    "score": score,
                    "length": length,
                    "found_terms": found_terms,
                    "has_unwanted": has_unwanted,
                    "is_complete": is_complete
                }
                
                filename = f"test_result_{int(time.time())}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ 结果已保存到: {filename}")
            
        finally:
            # 恢复原始参数
            generator.temperature = original_temp
            generator.top_p = original_top_p
            generator.max_new_tokens = original_max_tokens
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    while True:
        print("\n" + "=" * 60)
        print("Generator LLM Prompt 调优工具")
        print("=" * 60)
        print("1. 开始交互式测试")
        print("2. 查看 Prompt 变体库")
        print("3. 查看参数组合库")
        print("4. 查看测试场景库")
        print("0. 退出")
        print("-" * 60)
        
        choice = input("请选择: ").strip()
        
        if choice == "1":
            test_specific_combination()
        elif choice == "2":
            from prompt_variations_library import get_prompt_variations
            variations = get_prompt_variations("", "", "")
            print("\nPrompt 变体库:")
            for name in variations.keys():
                print(f"  - {name}")
        elif choice == "3":
            from prompt_variations_library import get_parameter_variations
            parameters = get_parameter_variations()
            print("\n参数组合库:")
            for name, params in parameters.items():
                print(f"  - {name}: {params['description']}")
        elif choice == "4":
            from prompt_variations_library import get_test_scenarios
            scenarios = get_test_scenarios()
            print("\n测试场景库:")
            for name, scenario in scenarios.items():
                print(f"  - {name}: {scenario['query']}")
        elif choice == "0":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 