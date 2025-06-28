#!/usr/bin/env python3
"""
测试不同max_new_tokens值的效果
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_token_lengths():
    """测试不同max_new_tokens值的效果"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH
        
        print("🧪 测试不同max_new_tokens值的效果...")
        
        # 测试问题
        test_question = "首钢股份的业绩表现如何？"
        
        # 模拟检索到的上下文
        test_context = """
        首钢股份2023年上半年业绩报告显示，公司实现营业收入1,234.56亿元，同比下降21.7%；
        净利润为-3.17亿元，同比下降77.14%，每股亏损0.06元。
        公司表示将通过注入优质资产来提升长期盈利能力，回馈股东。
        """
        
        # 构建prompt
        prompt = PROMPT_TEMPLATE_ZH.format(context=test_context, question=test_question)
        
        # 测试不同的max_new_tokens值
        token_values = [150, 200, 250, 300, 400]
        
        results = {}
        
        for max_tokens in token_values:
            print(f"\n" + "="*60)
            print(f"🔧 测试 max_new_tokens = {max_tokens}")
            print("="*60)
            
            # 临时修改配置
            config = Config()
            original_max_tokens = config.generator.max_new_tokens
            config.generator.max_new_tokens = max_tokens
            
            print(f"使用生成器模型: {config.generator.model_name}")
            print(f"生成参数: max_tokens={max_tokens}, temperature={config.generator.temperature}")
            
            # 加载生成器
            generator = load_generator(
                generator_model_name=config.generator.model_name,
                use_local_llm=True,
                use_gpu=True,
                gpu_device="cuda:1",
                cache_dir=config.generator.cache_dir
            )
            
            # 生成回答
            print(f"\n🤖 生成回答中...")
            generated_responses = generator.generate(texts=[prompt])
            answer = generated_responses[0] if generated_responses else "生成失败"
            
            print(f"\n✅ 生成的回答:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            # 分析回答
            answer_length = len(answer)
            word_count = len(answer.split())
            
            print(f"\n📊 回答分析:")
            print(f"字符数: {answer_length}")
            print(f"词数: {word_count}")
            
            # 质量评估
            quality_score = 0
            issues = []
            
            # 检查是否包含关键信息
            if "首钢" in answer and ("业绩" in answer or "利润" in answer or "收入" in answer):
                quality_score += 2
            else:
                issues.append("缺少关键信息")
            
            # 检查是否包含具体数据
            if any(char.isdigit() for char in answer):
                quality_score += 1
            else:
                issues.append("缺少具体数据")
            
            # 检查是否完整
            if answer.endswith(('.', '。', '！', '？')):
                quality_score += 1
            else:
                issues.append("回答不完整")
            
            # 检查长度是否合适
            if 50 <= answer_length <= 300:
                quality_score += 1
            elif answer_length < 50:
                issues.append("回答过短")
            else:
                issues.append("回答过长")
            
            print(f"质量评分: {quality_score}/5")
            if issues:
                print(f"问题: {', '.join(issues)}")
            
            # 保存结果
            results[max_tokens] = {
                "answer": answer,
                "length": answer_length,
                "word_count": word_count,
                "quality_score": quality_score,
                "issues": issues
            }
            
            # 恢复原始配置
            config.generator.max_new_tokens = original_max_tokens
        
        # 比较结果
        print(f"\n" + "="*60)
        print("📊 综合比较结果")
        print("="*60)
        
        print(f"{'max_tokens':<10} {'字符数':<8} {'词数':<6} {'质量评分':<8} {'状态'}")
        print("-" * 50)
        
        best_score = 0
        best_tokens = None
        
        for max_tokens in token_values:
            result = results[max_tokens]
            status = "✅ 推荐" if result["quality_score"] >= 4 else "⚠️ 一般" if result["quality_score"] >= 3 else "❌ 较差"
            
            if result["quality_score"] > best_score:
                best_score = result["quality_score"]
                best_tokens = max_tokens
            
            print(f"{max_tokens:<10} {result['length']:<8} {result['word_count']:<6} {result['quality_score']:<8} {status}")
        
        print(f"\n🏆 最佳配置: max_new_tokens = {best_tokens} (质量评分: {best_score}/5)")
        
        # 详细推荐
        print(f"\n💡 推荐配置:")
        if best_tokens == 150:
            print("✅ 推荐使用 max_new_tokens = 150")
            print("   理由: 在保持回答完整性的同时，避免生成过长内容")
        elif best_tokens == 200:
            print("✅ 推荐使用 max_new_tokens = 200")
            print("   理由: 为复杂问题提供足够的回答空间")
        elif best_tokens == 100:
            print("✅ 推荐使用 max_new_tokens = 100")
            print("   理由: 适合简洁明了的回答")
        else:
            print(f"✅ 推荐使用 max_new_tokens = {best_tokens}")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_token_lengths() 